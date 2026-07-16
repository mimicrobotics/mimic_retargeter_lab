import numpy as np
from scipy.spatial.transform import Rotation as R

from dexworld.hand_models import RobotHandModel
from dexworld.types import HandLandmark
from dexworld.utils.human_hands.mano_utils import rotation_matrix_z


class WristRetargeter:
    def __init__(self, to_model: RobotHandModel, wrist_mapping):
        self.to_model = to_model
        self.tgt_key = wrist_mapping["tgt_key"]
        self.root_key = wrist_mapping["root_key"]
        self.transform = wrist_mapping["transform"]

        # --- Rotation offset ---
        rot_euler = wrist_mapping.get("rot_offset_euler", None)
        if rot_euler is not None:
            self.rot_offset = (
                R.from_euler("XYZ", rot_euler).as_matrix().astype(np.float32)
            )
        elif to_model is not None:
            # Auto-compute from robot neutral pose landmarks
            self.rot_offset = self._auto_compute_rot_offset()
        else:
            from dexworld.utils.human_hands.mano_utils import rotation_matrix_x

            self.rot_offset = (
                R.from_quat([0.5, -0.5, 0.5, 0.5]).as_matrix()
                @ rotation_matrix_x(np.pi + np.pi / 8)
            ).astype(np.float32)

        # --- Translation offset ---
        trans = wrist_mapping.get("trans_offset", None)
        if trans is not None:
            self.trans_offset = np.array(trans, dtype=np.float32)
        elif to_model is not None:
            self.trans_offset = self._auto_compute_trans_offset()
        else:
            self.trans_offset = np.array([0, 0, -0.24], dtype=np.float32)

    def _auto_compute_rot_offset(self) -> np.ndarray:
        """Compute rot_offset automatically from robot neutral-pose landmarks.

        Uses the same 3-landmark basis construction as get_hand_rotation_matrix
        (WRIST, INDEX_BASE, PINKY_BASE) to build the robot's "finger frame",
        then computes the rotation from that frame to the root body's frame.
        """
        import mujoco
        from dexworld.utils.mj_utils import get_mj_context

        # 1. Get landmarks at neutral pose (in standalone-model world frame)
        neutral_qpos = self.to_model.get_neutral_ctrl_pose()
        landmarks = self.to_model.get_landmarks(neutral_qpos)

        wrist = landmarks[HandLandmark.WRIST]
        index_base = landmarks.get(HandLandmark.INDEX_BASE)
        # Use PINKY_BASE if available, else fall back to RING_BASE (e.g. 4-finger hands)
        outer_base = landmarks.get(HandLandmark.PINKY_BASE)
        if outer_base is None:
            outer_base = landmarks.get(HandLandmark.RING_BASE)
        if index_base is None or outer_base is None:
            return np.eye(3, dtype=np.float32)

        # 2. Construct a "finger frame" identical to get_hand_rotation_matrix
        base_1 = (index_base - wrist).copy()
        base_2 = (outer_base - wrist).copy()
        normal = np.cross(base_1, base_2)
        base_2 = np.cross(normal, base_1)

        base_1 /= np.linalg.norm(base_1) + 1e-8
        base_2 /= np.linalg.norm(base_2) + 1e-8
        normal /= np.linalg.norm(normal) + 1e-8

        R_frame = np.column_stack([base_1, base_2, normal])
        R_fingers_world = R_frame @ rotation_matrix_z(-np.pi / 2)

        # 3. Get root_key body's world orientation at neutral pose
        model_path = self.to_model.get_model_path()
        model, data = get_mj_context(model_path)
        # get_landmarks already ran mj_forward with neutral qpos, so data is current
        root_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.root_key)
        R_root_world = data.xmat[root_bid].reshape(3, 3).copy()

        # 4. rot_offset = R_fingers_world^T @ R_root_world
        #    This maps: human hand frame → robot root body frame
        rot_offset = (R_fingers_world.T @ R_root_world).astype(np.float32)
        return rot_offset

    def _auto_compute_trans_offset(self) -> np.ndarray:
        """Compute trans_offset from the wrist position in the root body's frame."""
        import mujoco
        from dexworld.utils.mj_utils import get_mj_context

        neutral_qpos = self.to_model.get_neutral_ctrl_pose()
        landmarks = self.to_model.get_landmarks(neutral_qpos)

        wrist = landmarks.get(HandLandmark.WRIST)
        if wrist is None:
            return np.zeros(3, dtype=np.float32)

        # Get root body position and rotation
        model_path = self.to_model.get_model_path()
        model, data = get_mj_context(model_path)
        root_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.root_key)
        root_pos = data.xpos[root_bid].copy()
        R_root = data.xmat[root_bid].reshape(3, 3)

        # Wrist position relative to root body, in root-local coords
        wrist_in_root = R_root.T @ (wrist - root_pos)

        # trans_offset moves the root body so the wrist aligns with the target
        # i.e., offset = -wrist_in_root (in root-local frame)
        return (-wrist_in_root).astype(np.float32)

    def retarget(self, wrist_transform: np.ndarray) -> np.ndarray:
        """
        wrist_transform: (B, 4, 4) source wrist to world.
        """
        wrist_transform = np.asarray(wrist_transform, dtype=np.float32).copy()
        for i in range(wrist_transform.shape[0]):
            wrist_transform[i, :3, :3] = wrist_transform[i, :3, :3] @ self.rot_offset
            world_trans_offset = self.trans_offset @ wrist_transform[i, :3, :3].T
            wrist_transform[i, :3, 3] += world_trans_offset
        return wrist_transform
