from __future__ import annotations

from pathlib import Path

import numpy as np

from dexworld.types import Chirality, HandLandmark, MujocoLandmark
from .robot_hand_base import RobotHandModel


class ShadowDexeeHandModel(RobotHandModel):
    """Shadow DEX-EE: 3-fingered Shadow hand (F0, F1, F2) with 4 DOFs per finger.

    The DEX-EE MJCF is mirror-symmetric, so both chiralities load the same
    ``shadow_dexee.xml`` file. ``chirality`` is still meaningful — it indicates
    which hand this *instance* represents in a (possibly bimanual) scene.

    Finger-to-landmark mapping (3 fingers → 3 canonical fingertips):
        F0 → thumb,  F1 → index,  F2 → middle.
    """

    _mjcf_is_symmetric = True
    _symmetric_mjcf_name = "shadow_dexee.xml"

    def __init__(self, robot_base_path: Path, chirality: Chirality):
        super().__init__(robot_base_path, chirality)

        self.num_fingertips = 3
        self.num_qpos_dofs = len(self.get_qpos_joint_names())
        self.num_actuated_dofs = len(self.get_actuated_joint_names())

        # The single source of truth for all landmarks.
        self._landmark_config: dict[HandLandmark, MujocoLandmark] = {
            HandLandmark.ARM_ATTACHMENT: MujocoLandmark(
                name="hand_base", object_type="body"
            ),
            HandLandmark.WRIST: MujocoLandmark(
                name="attachment_site", object_type="site"
            ),
            # Fingertips (sites placed at the tip of each distal link)
            HandLandmark.THUMB_TIP: MujocoLandmark(
                name="F0/distal_site", object_type="site"
            ),
            HandLandmark.INDEX_TIP: MujocoLandmark(
                name="F1/distal_site", object_type="site"
            ),
            HandLandmark.MIDDLE_TIP: MujocoLandmark(
                name="F2/distal_site", object_type="site"
            ),
            # Finger bases (abduction joints — the pivot point at the knuckle)
            HandLandmark.THUMB_BASE: MujocoLandmark(
                name="F0/j0_site", object_type="site"
            ),
            HandLandmark.INDEX_BASE: MujocoLandmark(
                name="F1/j0_site", object_type="site"
            ),
            HandLandmark.MIDDLE_BASE: MujocoLandmark(
                name="F2/j0_site", object_type="site"
            ),
            # Distal phalanx sites (J3 anchor — base of the distal link).
            HandLandmark.THUMB_DP: MujocoLandmark(
                name="F0/j3_site", object_type="site"
            ),
            HandLandmark.INDEX_DP: MujocoLandmark(
                name="F1/j3_site", object_type="site"
            ),
            HandLandmark.MIDDLE_DP: MujocoLandmark(
                name="F2/j3_site", object_type="site"
            ),
        }

        self.joint_map = self.compute_joint_map()

        # Compile MJX immediately
        self.create_mjx_kinematic_model()

    # ------------------------------------------------------------------
    # Joint names and coupling
    # ------------------------------------------------------------------
    def get_qpos_joint_names(self) -> list[str]:
        return [
            "F0/J0",
            "F0/J1",
            "F0/J2",
            "F0/J3",
            "F1/J0",
            "F1/J1",
            "F1/J2",
            "F1/J3",
            "F2/J0",
            "F2/J1",
            "F2/J2",
            "F2/J3",
        ]

    def get_actuated_joint_names(self) -> list[str]:
        return self.get_qpos_joint_names()

    def compute_joint_map(self) -> np.ndarray:
        return np.eye(self.num_qpos_dofs, dtype=np.float32)
