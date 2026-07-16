"""Minimal hand kinematic model class.

Author(s):
    - Robert Jomar Malate (robert.malate@mimicrobotics.com)
"""

import numpy as np

# Pinocchio is the URDF backend only — lazy-imported so dexworld's MJCF
# training path can `import geort.trainer` (which imports this module) on
# machines without pinocchio installed. URDF users still get a clear error
# at construction time below.
try:
    import pinocchio as pin
except ImportError:
    pin = None


class HandKinematicModel:
    def __init__(
        self,
        urdf_path: str,
        joint_names: list[str] | None,
    ):
        if pin is None:
            raise ImportError(
                "pinocchio is required for the URDF kinematic backend "
                "(HandKinematicModel). Install it via `pip install pin==2.6.21`. "
                "If you're using dexworld's MJCF training path, use "
                "`MjcfHandKinematicModel` instead."
            )
        # Load model with Pinocchio
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # Setup joint indexing
        self.joint_names_user = joint_names
        self.n_user_dof = len(joint_names)

        # Setting up Pinocchio joint IDs
        self.user_to_pin_id = []
        for name in joint_names:
            if self.model.existJointName(name):
                self.user_to_pin_id.append(self.model.getJointId(name))
            else:
                error_msg = f"Joint name '{name}' not found in URDF model."
                raise ValueError(error_msg)

        # Setting up joint coupling
        self._coupling_joint_map = {}
        self._add_joint_coupling("index_mp2dp", "index_pp2mp", 1.0, 0.0)
        self._add_joint_coupling("middle_mp2dp", "middle_pp2mp", 1.0, 0.0)
        self._add_joint_coupling("ring_mp2dp", "ring_pp2mp", 1.0, 0.0)
        self._add_joint_coupling("pinky_mp2dp", "pinky_pp2mp", 1.0, 0.0)

    def _add_joint_coupling(
        self,
        child_joint_name: str,
        parent_joint_name: str,
        multiplier: float,
        offset: float,
    ) -> None:
        """
        Register that sim joint `child_joint_name` is coupled to
        `parent_joint_name` as:

            q_child = multiplier * q_parent + offset

        All indices are in *sim* joint indexing.
        """
        if self.model.existJointName(child_joint_name) and self.model.existJointName(
            parent_joint_name
        ):
            child_joint_idx_sim = self.model.getJointId(child_joint_name)
            parent_joint_idx_sim = self.model.getJointId(parent_joint_name)
            self._coupling_joint_map[child_joint_idx_sim] = (
                parent_joint_idx_sim,
                multiplier,
                offset,
            )

    def _rot_matrix_to_quat_numpy(self, R) -> np.ndarray:
        """Pure NumPy implementation to avoid pin.Quaternion C++ segfaults."""
        # Ensure input is a standard float32 array
        R = np.array(R, dtype=np.float32)

        tr = R[0, 0] + R[1, 1] + R[2, 2]

        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

        return np.array([qx, qy, qz, qw], dtype=np.float32)

    # ----------------------------------------------------------------------
    # Public API used by GeoRTTrainer
    # ----------------------------------------------------------------------
    def get_n_dof(self):
        return len(self.joint_lower_limit)

    def get_joint_limit(self):
        """
        Returns (lower_limits, upper_limits) for the USER joints.
        """
        lower = []
        upper = []

        for pin_id in self.user_to_pin_id:
            # Pinocchio stores limits in the model.
            # idx_q is the starting index of this joint in the configuration vector
            idx_q = self.model.joints[pin_id].idx_q

            # Pinocchio stores limits for the whole q vector.
            # We just grab the limit for the specific axis of this joint.
            lower_limits = self.model.lowerPositionLimit[idx_q]
            upper_limits = self.model.upperPositionLimit[idx_q]
            lower.append(lower_limits)
            upper.append(upper_limits)

        return np.array(lower, dtype=np.float32), np.array(upper, dtype=np.float32)

    def initialize_keypoint(self, keypoint_link_names, keypoint_offsets):
        self.keypoint_definitions = []
        for name, offset in zip(keypoint_link_names, keypoint_offsets):
            if self.model.existFrame(name):
                frame_id = self.model.getFrameId(name)
                self.keypoint_definitions.append((name, frame_id, np.array(offset)))
            else:
                error_msg = f"Link name '{name}' not found in URDF model."
                raise ValueError(error_msg)

    def convert_user_q_to_pinocchio_q(self, q_user):
        """
        Convert qpos from user joint order to Pinocchio joint order.
        Also applies joint coupling.
        """
        q_pin = pin.neutral(self.model)  # Creating zero-vector of correct size

        # Setting all the user joints (active joints that the user controls)
        for i, pin_id in enumerate(self.user_to_pin_id):
            # idx_q is the index in the q-vector for a specific joint ID
            q_idx = self.model.joints[pin_id].idx_q
            q_pin[q_idx] = q_user[i]

        # Setting the coupled joints
        for child_idx, (
            parent_idx,
            multiplier,
            offset,
        ) in self._coupling_joint_map.items():
            child_q_idx = self.model.joints[child_idx].idx_q
            parent_q_idx = self.model.joints[parent_idx].idx_q
            q_pin[child_q_idx] = multiplier * q_pin[parent_q_idx] + offset

        return q_pin

    def keypoint_from_qpos(self, qpos_user, ret_vec=False, ret_orientation=False):
        """
        Get keypoints from hand qpos. qpos is in *user* joint order.
        This is the ONLY thing the FK dataset & trainer actually need.
        """
        # SAFEGUARD #1: Type enforcement
        qpos_user = np.ascontiguousarray(qpos_user, dtype=np.float32)

        # Convert to Pinocchio joint order
        q_pin = self.convert_user_q_to_pinocchio_q(qpos_user)

        # SAFEGUARD #2: Check q_pin size
        if q_pin.shape[0] != self.model.nq:
            error_msg = f"Input qpos has incorrect size. Expected {self.model.nq}, got {q_pin.shape[0]}."
            raise ValueError(error_msg)

        # Compute FK
        try:
            pin.forwardKinematics(self.model, self.data, q_pin)
            pin.updateFramePlacements(self.model, self.data)
        except Exception as e:
            error_msg = f"Pinocchio forward kinematics computation failed: {e}"
            raise RuntimeError(error_msg) from e

        # Extract keypoints
        result = {}
        vec_result = []
        for name, frame_id, offset in self.keypoint_definitions:
            # Get frame placement (SE3 object)
            frame_placement = self.data.oMf[frame_id]

            # transform offset: local -> world
            safe_offset = np.ascontiguousarray(offset, dtype=np.float32)
            pos_world = frame_placement.act(safe_offset)
            vec_result.append(pos_world)

            if ret_orientation:
                rot_matrix = frame_placement.rotation
                quat_vec = self._rot_matrix_to_quat_numpy(rot_matrix)

                result[name] = (pos_world, quat_vec)
                vec_result.append(quat_vec)
            else:
                result[name] = pos_world
        if ret_vec:
            return np.array(vec_result, dtype=np.float32)

        return result

    @staticmethod
    def build_from_config(config, **kwargs):
        return HandKinematicModel(
            urdf_path=config["urdf_path"],
            joint_names=config["joint_order"],
        )
