import numpy as np

import mimic_retargeter_lab.utils.human_hands.mano_utils as mano_utils
from .hand_kinematics import HandKinematicsInverse


class ManoPreprocessor:
    def __init__(self, local_rot_representation="matrix"):
        """
        ManoPreprocessor converts raw 21-keypoint hand positions into the
        normalised representation used for retargeting (joints, keyvectors and
        a local per-joint rotation representation).

        "MANO" here refers only to the canonical 21-keypoint ordering (see
        ``mano_utils``), not to the parametric MANO model — no hand model is
        loaded.

        Args:
            local_rot_representation: Rotation representation for the IK pass.
        """
        self.hand_kinematics_inverse = HandKinematicsInverse(
            rotation_representation=local_rot_representation
        )

    def mano_joints_to_keyvectors(self, mano_joints):
        mano_joints = np.asarray(mano_joints, dtype=np.float32)
        mano_joints_dict = mano_utils.get_mano_joints_dict(
            mano_joints, batch_processing=True
        )
        mano_fingertips = mano_utils.get_mano_fingertips_batch(mano_joints_dict)
        mano_pps = mano_utils.get_mano_pps_batch(mano_joints_dict)

        pps_and_wrist = np.concatenate(
            [mano_joints[:, [0], :], mano_pps["index"], mano_pps["pinky"]], axis=1
        )  # (B, 3, 3)
        mano_palm = np.mean(pps_and_wrist, axis=1, keepdims=True)  # (B, 1, 3)

        return mano_utils.get_keyvectors(mano_fingertips, mano_palm)

    def convert_from_joints(self, joints, add_normalization=False, convert_units=False):
        """
        joints: (N, 21, 3) joint positions.

        Returns a dict with keys: pose, shape, joints, keyvectors, local_representation.
        """
        joints = np.asarray(joints, dtype=np.float32)

        if add_normalization:
            joints = np.stack(
                [
                    mano_utils.normalize_points(
                        joints_i.copy(),
                        flip_x_axis=False,
                        flip_y_axis=True,
                        add_z_rotation=np.pi / 16,
                    )
                    for joints_i in joints
                ]
            ).astype(np.float32)

        if convert_units:
            joints = joints / 1000.0

        local_representation = self.hand_kinematics_inverse(joints)
        keyvectors_mano = self.mano_joints_to_keyvectors(joints)

        return {
            "pose": None,
            "shape": None,
            "joints": joints,
            "keyvectors": keyvectors_mano,
            "local_representation": local_representation,
        }
