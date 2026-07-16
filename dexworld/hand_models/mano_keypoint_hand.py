import numpy as np

import dexworld.utils.human_hands.mano_utils as mano_utils
from dexworld.types.types import Chirality, HandLandmark
from dexworld.utils import (
    LOCAL_JOINT_DOFS,
    HandKinematicsForward,
    HandKinematicsInverse,
    extract_joint_angles,
)

from .human_hand_base import HumanHandModel


class ManoKeypointHandModel(HumanHandModel):
    """Human hand driven by the 21 keypoints of the MANO joint ordering.

    "MANO" here names the **joint-ordering convention** only — wrist, then four
    joints per finger — which is what vision-based hand estimators (MediaPipe,
    WiLoR, HaMeR) emit. No parametric MANO model is loaded anywhere in dexworld.

    The input is positions-only, so joint orientations are recovered by inverse
    kinematics. Contrast with :class:`ManusHandModel`, which consumes 25 glove
    nodes that already carry real orientations and therefore needs no IK.
    """

    def __init__(
        self,
        rotation_representation: str = "matrix",
        chirality: Chirality = Chirality.RIGHT,
    ):
        super().__init__(chirality)
        self.forward_kinematics = HandKinematicsForward(
            rotation_representation=rotation_representation
        )
        self.inverse_kinematics = HandKinematicsInverse(
            rotation_representation=rotation_representation
        )

        self.pose_dim = 45

        # Single source of truth for MANO point cloud indices.
        # DP (distal phalanx base) = tip_idx - 1 in the canonical MANO 21-point topology.
        self._landmark_config: dict[HandLandmark, int] = {
            HandLandmark.WRIST: 0,
            HandLandmark.THUMB_BASE: 1,
            HandLandmark.THUMB_DP: 3,
            HandLandmark.THUMB_TIP: 4,
            HandLandmark.INDEX_BASE: 5,
            HandLandmark.INDEX_DP: 7,
            HandLandmark.INDEX_TIP: 8,
            HandLandmark.MIDDLE_BASE: 9,
            HandLandmark.MIDDLE_DP: 11,
            HandLandmark.MIDDLE_TIP: 12,
            HandLandmark.RING_BASE: 13,
            HandLandmark.RING_DP: 15,
            HandLandmark.RING_TIP: 16,
            HandLandmark.PINKY_BASE: 17,
            HandLandmark.PINKY_DP: 19,
            HandLandmark.PINKY_TIP: 20,
        }

        self.num_fingertips = 5

    def get_num_fingertips(self) -> int:
        return self.num_fingertips

    def get_qpos_joint_names(self):
        return LOCAL_JOINT_DOFS

    def get_landmarks(
        self, joints_3d: np.ndarray, **kwargs
    ) -> dict[HandLandmark, np.ndarray]:
        """Retrieve semantic landmark positions directly from a 21-point cloud."""
        pts = np.asarray(joints_3d)
        return {
            landmark: pts[..., idx, :]
            for landmark, idx in self._landmark_config.items()
        }

    def get_landmark_transforms(
        self, joints_3d: np.ndarray
    ) -> dict[HandLandmark, np.ndarray]:
        """Return 4x4 transforms for all configured landmarks.

        Parameters
        ----------
        joints_3d : (21, 3) or (N, 21, 3) array

        Returns
        -------
        dict mapping HandLandmark → ndarray of shape (4, 4) or (N, 4, 4).
        """
        joints_np = np.asarray(joints_3d, dtype=np.float32)
        is_single = joints_np.ndim == 2
        if is_single:
            joints_np = joints_np[None]

        normalized_joints = self._normalize_joints(joints_np)
        local_repr = self.inverse_kinematics(normalized_joints)
        root_pos = np.zeros((local_repr.shape[0], 3), dtype=np.float32)
        mano_frames = self.forward_kinematics.compute_kinematic_tree(
            local_repr, root_pos
        )

        result: dict[HandLandmark, np.ndarray] = {}
        for landmark, joint_idx in self._landmark_config.items():
            joint_name = mano_utils.JOINT_NAMES[joint_idx]
            T = mano_frames[joint_name]  # (B, 4, 4)
            result[landmark] = T[0] if is_single else T

        return result

    def _normalize_joints(self, joints: np.ndarray) -> np.ndarray:
        """Normalize a batch of (B, 21, 3) joint arrays in numpy."""
        return np.stack(
            [
                mano_utils.normalize_points(
                    joints_i.copy(),
                    flip_x_axis=False,
                    flip_y_axis=True,
                    add_z_rotation=np.pi / 8,
                )
                for joints_i in joints
            ]
        ).astype(np.float32)

    def to_kinematic_tree(
        self,
        joints_3d: np.ndarray,
        return_frame_dict: bool = True,
        root_pos: np.ndarray | None = None,
    ) -> tuple[np.ndarray | dict, list]:
        """Stateless computation of the kinematic tree from 3D keypoints."""
        joints_np = np.asarray(joints_3d, dtype=np.float32)
        is_single = joints_np.ndim == 2
        if is_single:
            joints_np = joints_np[None]

        normalized_joints = self._normalize_joints(joints_np)
        local_repr = self.inverse_kinematics(normalized_joints)

        if root_pos is None:
            root_pos_np = np.zeros((local_repr.shape[0], 3), dtype=np.float32)
        else:
            root_pos_np = np.asarray(root_pos, dtype=np.float32)
            if root_pos_np.ndim == 1:
                root_pos_np = root_pos_np[None]

        mano_frames = self.forward_kinematics.compute_kinematic_tree(
            local_repr, root_pos_np
        )

        links = []
        for child_idx, joint_name in enumerate(mano_utils.JOINT_NAMES):
            parent_idx = mano_utils.KINEMATIC_TREE[child_idx]
            if parent_idx == -1:
                continue
            link_start = mano_frames[joint_name][:, :3, 3]
            link_end = mano_frames[mano_utils.JOINT_NAMES[parent_idx]][:, :3, 3]
            if is_single:
                links.append((link_start[0], link_end[0]))
            else:
                links.append((link_start, link_end))

        if return_frame_dict:
            frame_dict = {k: (v[0] if is_single else v) for k, v in mano_frames.items()}
            return frame_dict, links

        frames_tensor = np.stack(list(mano_frames.values()))
        # (num_joints, B, 4, 4) → (B, num_joints, 4, 4)
        frames_out = np.swapaxes(frames_tensor, 0, 1)
        if is_single:
            frames_out = frames_out[0]
        return frames_out, links

    def to_joint_angles(self, joints_3d: np.ndarray) -> dict[str, np.ndarray]:
        """Stateless computation of internal joint angles from 3D keypoints."""
        joints_np = np.asarray(joints_3d, dtype=np.float32)
        is_single = joints_np.ndim == 2
        if is_single:
            joints_np = joints_np[None]

        normalized_joints = self._normalize_joints(joints_np)
        local_repr = self.inverse_kinematics(normalized_joints)

        batch_size = local_repr.shape[0]
        dummy_root_pos = np.zeros((batch_size, 3), dtype=np.float32)

        _, local_rotation_matrices = self.forward_kinematics(local_repr, dummy_root_pos)

        local_kin_tree: dict[str, np.ndarray] = {}
        eye4 = np.eye(4, dtype=np.float32)
        for i, joint_name in enumerate(mano_utils.JOINT_NAMES):
            joint_transform = np.broadcast_to(eye4[None], (batch_size, 4, 4)).copy()
            joint_transform[:, :3, :3] = local_rotation_matrices[:, i]
            local_kin_tree[joint_name] = joint_transform

        joint_angles = extract_joint_angles(local_kin_tree)

        if is_single:
            return {k: v[0] for k, v in joint_angles.items()}
        return joint_angles

    def local_repr_to_joints(self, local_repr: np.ndarray) -> np.ndarray:
        """Convert MANO local representation to 3D joint positions (stateless)."""
        local_repr = np.asarray(local_repr, dtype=np.float32)
        root_pos = np.zeros((local_repr.shape[0], 3), dtype=np.float32)
        joints, _ = self.forward_kinematics(local_repr, root_pos)
        return joints
