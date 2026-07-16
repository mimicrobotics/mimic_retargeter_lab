from __future__ import annotations

from pathlib import Path

import numpy as np

from dexworld.types import Chirality, HandLandmark, MujocoLandmark
from .robot_hand_base import RobotHandModel


class WonikAllegroHandModel(RobotHandModel):
    def __init__(self, robot_base_path: Path, chirality: Chirality):
        super().__init__(robot_base_path, chirality)
        self.ch_prefix = "R" if chirality == Chirality.RIGHT else "L"

        self.num_fingertips = 4
        self.num_qpos_dofs = len(self.get_qpos_joint_names())
        self.num_actuated_dofs = len(self.get_actuated_joint_names())

        # The single source of truth for all landmarks
        self._landmark_config: dict[HandLandmark, tuple[str, str]] = {
            HandLandmark.ARM_ATTACHMENT: MujocoLandmark(
                name="arm_attachment", object_type="body"
            ),
            HandLandmark.WRIST: MujocoLandmark(name="wrist", object_type="body"),
            HandLandmark.PALM: MujocoLandmark(name="palm", object_type="body"),
            # Fingertips
            HandLandmark.THUMB_TIP: MujocoLandmark(name="th_tip", object_type="body"),
            HandLandmark.INDEX_TIP: MujocoLandmark(name="ff_tip", object_type="body"),
            HandLandmark.MIDDLE_TIP: MujocoLandmark(name="mf_tip", object_type="body"),
            HandLandmark.RING_TIP: MujocoLandmark(name="rf_tip", object_type="body"),
            # Finger Bases
            HandLandmark.THUMB_BASE: MujocoLandmark(name="th_base", object_type="body"),
            HandLandmark.INDEX_BASE: MujocoLandmark(name="ff_base", object_type="body"),
            HandLandmark.MIDDLE_BASE: MujocoLandmark(
                name="mf_base", object_type="body"
            ),
            HandLandmark.RING_BASE: MujocoLandmark(name="rf_base", object_type="body"),
            # Distal phalanx bodies (origin sits at the base of the distal link).
            # Allegro has 4 fingers; no pinky DP.
            HandLandmark.THUMB_DP: MujocoLandmark(name="th_distal", object_type="body"),
            HandLandmark.INDEX_DP: MujocoLandmark(name="ff_distal", object_type="body"),
            HandLandmark.MIDDLE_DP: MujocoLandmark(
                name="mf_distal", object_type="body"
            ),
            HandLandmark.RING_DP: MujocoLandmark(name="rf_distal", object_type="body"),
        }

        self.joint_map = self.compute_joint_map()

        # Compile MJX immediately
        self.create_mjx_kinematic_model()

    def get_qpos_joint_names(self) -> list[str]:
        return [
            "ffj0",
            "ffj1",
            "ffj2",
            "ffj3",
            "mfj0",
            "mfj1",
            "mfj2",
            "mfj3",
            "rfj0",
            "rfj1",
            "rfj2",
            "rfj3",
            "thj0",
            "thj1",
            "thj2",
            "thj3",
        ]

    def _joint_name_from_actuated_name(self, actuated_joint_name: str) -> str:
        """Map actuator name (e.g. 'ffa0') back to qpos joint name ('ffj0')."""
        return actuated_joint_name[:2] + "j" + actuated_joint_name[3:]

    def get_actuated_joint_names(self) -> list[str]:
        return [
            "ffa0",
            "ffa1",
            "ffa2",
            "ffa3",
            "mfa0",
            "mfa1",
            "mfa2",
            "mfa3",
            "rfa0",
            "rfa1",
            "rfa2",
            "rfa3",
            "tha0",
            "tha1",
            "tha2",
            "tha3",
        ]

    def compute_joint_map(self) -> np.ndarray:
        return np.eye(self.num_qpos_dofs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Keyvectors & Kinematic tree overrides
    # ------------------------------------------------------------------

    # def _extra_keyvector_frame_names(self) -> list[str]:
    #     # Tell the base class to explicitly track the 'palm' string in MJX
    #     return [self._landmark_config[HandLandmark.PALM][0]]
