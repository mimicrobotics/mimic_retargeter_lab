from typing import Literal

import numpy as np

from dexworld.utils.human_hands.mano_utils import KINEMATIC_TREE, JOINT_NAMES


# ---------------------------------------------------------------------------
# 6D rotation helpers (Zhou et al., CVPR 2019)
# ---------------------------------------------------------------------------


def _rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt 6D → rotation matrix.

    Args:
        d6: (..., 6)

    Returns:
        (..., 3, 3) rotation matrices
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-2)


def _matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrix → 6D (drop last row).

    Args:
        matrix: (..., 3, 3)

    Returns:
        (..., 6)
    """
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


# ---------------------------------------------------------------------------
# Kinematics classes
# ---------------------------------------------------------------------------


class HandKinematicsInverse:
    """Convert joint positions (B, 21, 3) → local rotations + translations."""

    def __init__(self, rotation_representation: Literal["matrix", "6d"] = "6d"):
        self.rotation_representation = rotation_representation

    def to(self, device):
        """No-op kept for API compatibility with callers that use .to(device)."""
        return self

    def __call__(self, joints: np.ndarray) -> np.ndarray:
        """
        Convert joint positions to local rotations and translations.
        Rotations are local (parent frame → current frame).
        Translations are in the parent's local frame.

        Args:
            joints: (batch_size, 21, 3) — accepts numpy arrays or torch tensors.

        Returns:
            (batch_size, output_dim) flat vector of local rotations + translations.
        """
        if hasattr(joints, "detach"):
            joints = joints.detach().cpu().numpy()
        joints = np.asarray(joints, dtype=np.float32)

        batch_size, num_joints, _ = joints.shape

        # Identity rotations, zero translations
        local_rotations = np.broadcast_to(
            np.eye(3, dtype=np.float32)[None, None], (batch_size, num_joints, 3, 3)
        ).copy()
        local_translations = np.zeros((batch_size, num_joints, 3), dtype=np.float32)
        global_rotations_cache = local_rotations.copy()

        for i in range(1, num_joints):
            parent_idx = KINEMATIC_TREE[i]
            parent_global_R = global_rotations_cache[:, parent_idx]  # (B, 3, 3)

            # Bone vector expressed in parent's local frame
            bone_global = joints[:, i] - joints[:, parent_idx]  # (B, 3)
            # T_local = R_parent^T @ T_global
            bone_local = (
                parent_global_R.swapaxes(-2, -1) @ bone_global[..., None]
            ).squeeze(-1)
            local_translations[:, i] = bone_local

            local_R = self._compute_rotation_matrices_y_aligned(bone_local)
            local_rotations[:, i] = local_R

            # G_current = G_parent @ L_current
            global_rotations_cache[:, i] = parent_global_R @ local_R

        if self.rotation_representation == "matrix":
            rotations_flat = local_rotations.reshape(batch_size, -1)
        elif self.rotation_representation == "6d":
            rotations_flat = _matrix_to_rotation_6d(local_rotations).reshape(
                batch_size, -1
            )
        else:
            raise ValueError(
                f"Unsupported rotation representation: {self.rotation_representation}"
            )

        return np.concatenate(
            [rotations_flat, local_translations.reshape(batch_size, -1)], axis=1
        )

    @staticmethod
    def _compute_rotation_matrices_y_aligned(y_axes: np.ndarray) -> np.ndarray:
        """Compute rotation matrices for a batch of y-aligned vectors (robust)."""
        batch_size = y_axes.shape[0]

        y_axes = y_axes / (np.linalg.norm(y_axes, axis=-1, keepdims=True) + 1e-8)

        x_ref = np.broadcast_to(
            np.array([1.0, 0.0, 0.0], dtype=y_axes.dtype), (batch_size, 3)
        )
        z_axes = np.cross(x_ref, y_axes)
        z_norm = np.linalg.norm(z_axes, axis=-1)
        collinear_mask = z_norm < 1e-6

        # Fallback when y is parallel to x_ref
        x_ref_fb = np.broadcast_to(
            np.array([0.0, 0.0, 1.0], dtype=y_axes.dtype), (batch_size, 3)
        )
        z_axes_fb = np.cross(x_ref_fb, y_axes)
        z_axes = np.where(collinear_mask[:, None], z_axes_fb, z_axes)
        z_axes = z_axes / (np.linalg.norm(z_axes, axis=-1, keepdims=True) + 1e-8)

        x_axes = np.cross(y_axes, z_axes)
        x_axes = x_axes / (np.linalg.norm(x_axes, axis=-1, keepdims=True) + 1e-8)

        return np.stack([x_axes, y_axes, z_axes], axis=-1)


class HandKinematicsForward:
    """Reconstruct joint positions from local rotations + translations."""

    def __init__(self, rotation_representation: Literal["matrix", "6d"] = "6d"):
        self.rotation_representation = rotation_representation

    def to(self, device):
        """No-op kept for API compatibility with callers that use .to(device)."""
        return self

    def __call__(
        self,
        local_transforms: np.ndarray,
        root_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            local_transforms: (batch_size, input_dim)
            root_positions:   (batch_size, 3)

        Returns:
            joint_positions:    (batch_size, 21, 3)
            rotation_matrices:  (batch_size, 21, 3, 3)  — local rotations
        """
        if hasattr(local_transforms, "detach"):
            local_transforms = local_transforms.detach().cpu().numpy()
        if hasattr(root_positions, "detach"):
            root_positions = root_positions.detach().cpu().numpy()

        local_transforms = np.asarray(local_transforms, dtype=np.float32)
        root_positions = np.asarray(root_positions, dtype=np.float32)

        batch_size = local_transforms.shape[0]
        num_joints = 21

        if self.rotation_representation == "6d":
            rot_dim = num_joints * 6
        elif self.rotation_representation == "matrix":
            rot_dim = num_joints * 9
        else:
            raise ValueError(
                f"Unsupported rotation representation: {self.rotation_representation}"
            )

        rotations = local_transforms[:, :rot_dim]
        translations = local_transforms[:, rot_dim:].reshape(batch_size, num_joints, 3)

        if self.rotation_representation == "6d":
            rotation_matrices = _rotation_6d_to_matrix(
                rotations.reshape(batch_size, num_joints, 6)
            )
        else:
            rotation_matrices = rotations.reshape(batch_size, num_joints, 3, 3)

        joint_positions = np.zeros((batch_size, num_joints, 3), dtype=np.float32)
        joint_positions[:, 0] = root_positions

        for i in range(1, num_joints):
            parent_idx = KINEMATIC_TREE[i]
            parent_global_R = self._compute_global_rotations(
                rotation_matrices, parent_idx
            )
            # T_global = G_parent @ T_local
            global_t = (parent_global_R @ translations[:, i, :, None]).squeeze(-1)
            joint_positions[:, i] = joint_positions[:, parent_idx] + global_t

        return joint_positions, rotation_matrices

    def compute_kinematic_tree(
        self,
        local_transforms: np.ndarray,
        root_positions: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return dict[joint_name → (B, 4, 4) global transform matrices]."""
        joint_positions, rotation_matrices = self(local_transforms, root_positions)
        batch_size = joint_positions.shape[0]

        # Compute global rotations for each joint
        global_Rs = rotation_matrices.copy()
        for i in range(1, 21):
            global_Rs[:, i] = self._compute_global_rotations(rotation_matrices, i)

        transforms = np.broadcast_to(
            np.eye(4, dtype=np.float32)[None, None],
            (batch_size, 21, 4, 4),
        ).copy()
        transforms[:, :, :3, 3] = joint_positions
        transforms[:, :, :3, :3] = global_Rs

        return {name: transforms[:, JOINT_NAMES.index(name)] for name in JOINT_NAMES}

    @staticmethod
    def _compute_global_rotations(
        local_rotations: np.ndarray, joint_idx: int
    ) -> np.ndarray:
        """G_i = L_0 @ L_1 @ ... @ L_i  (iterative, parent-to-root)."""
        global_R = local_rotations[:, joint_idx].copy()
        parent_idx = KINEMATIC_TREE[joint_idx]
        while parent_idx != -1:
            global_R = local_rotations[:, parent_idx] @ global_R
            parent_idx = KINEMATIC_TREE[parent_idx]
        return global_R
