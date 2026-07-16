from __future__ import annotations

from pathlib import Path

import numpy as np

from dexworld.types import Chirality, HandLandmark, MujocoLandmark
from .robot_hand_base import RobotHandModel


class ShadowHandModel(RobotHandModel):
    def __init__(self, robot_base_path: Path, chirality: Chirality):
        super().__init__(robot_base_path, chirality)
        self.ch_prefix = "rh" if chirality == Chirality.RIGHT else "lh"

        self.num_fingertips = 5
        self.num_actuated_dofs = len(self.get_actuated_joint_names())
        self.num_qpos_dofs = len(self.get_qpos_joint_names())

        # The single source of truth for all landmarks
        self._landmark_config: dict[HandLandmark, tuple[str, str]] = {
            HandLandmark.ARM_ATTACHMENT: MujocoLandmark(
                name="arm_attachment", object_type="body"
            ),
            # Wrist
            HandLandmark.WRIST: MujocoLandmark(
                name=f"{self.ch_prefix}_wrist", object_type="body"
            ),
            # Fingertips (Bodies)
            HandLandmark.THUMB_TIP: MujocoLandmark(
                name=f"{self.ch_prefix}_thtip", object_type="body"
            ),
            HandLandmark.INDEX_TIP: MujocoLandmark(
                name=f"{self.ch_prefix}_fftip", object_type="body"
            ),
            HandLandmark.MIDDLE_TIP: MujocoLandmark(
                name=f"{self.ch_prefix}_mftip", object_type="body"
            ),
            HandLandmark.RING_TIP: MujocoLandmark(
                name=f"{self.ch_prefix}_rftip", object_type="body"
            ),
            HandLandmark.PINKY_TIP: MujocoLandmark(
                name=f"{self.ch_prefix}_lftip", object_type="body"
            ),
            # Finger Bases (Joints)
            HandLandmark.THUMB_BASE: MujocoLandmark(
                name=f"{self.ch_prefix}_THJ5", object_type="joint"
            ),
            HandLandmark.INDEX_BASE: MujocoLandmark(
                name=f"{self.ch_prefix}_FFJ4", object_type="joint"
            ),
            HandLandmark.MIDDLE_BASE: MujocoLandmark(
                name=f"{self.ch_prefix}_MFJ4", object_type="joint"
            ),
            HandLandmark.RING_BASE: MujocoLandmark(
                name=f"{self.ch_prefix}_RFJ4", object_type="joint"
            ),
            HandLandmark.PINKY_BASE: MujocoLandmark(
                name=f"{self.ch_prefix}_LFJ4", object_type="joint"
            ),
            # Distal phalanx bodies (origin sits at the base of the distal link).
            HandLandmark.THUMB_DP: MujocoLandmark(
                name=f"{self.ch_prefix}_thdistal", object_type="body"
            ),
            HandLandmark.INDEX_DP: MujocoLandmark(
                name=f"{self.ch_prefix}_ffdistal", object_type="body"
            ),
            HandLandmark.MIDDLE_DP: MujocoLandmark(
                name=f"{self.ch_prefix}_mfdistal", object_type="body"
            ),
            HandLandmark.RING_DP: MujocoLandmark(
                name=f"{self.ch_prefix}_rfdistal", object_type="body"
            ),
            HandLandmark.PINKY_DP: MujocoLandmark(
                name=f"{self.ch_prefix}_lfdistal", object_type="body"
            ),
        }

        self.joint_map = self.compute_joint_map()

        # Compile MJX immediately
        self.create_mjx_kinematic_model()

    # ------------------------------------------------------------------
    # Joint names and coupling (Shadow-specific)
    # ------------------------------------------------------------------

    def get_qpos_joint_names(self) -> list[str]:
        return [
            f"{self.ch_prefix}_FFJ4",
            f"{self.ch_prefix}_FFJ3",
            f"{self.ch_prefix}_FFJ2",
            f"{self.ch_prefix}_FFJ1",
            f"{self.ch_prefix}_MFJ4",
            f"{self.ch_prefix}_MFJ3",
            f"{self.ch_prefix}_MFJ2",
            f"{self.ch_prefix}_MFJ1",
            f"{self.ch_prefix}_RFJ4",
            f"{self.ch_prefix}_RFJ3",
            f"{self.ch_prefix}_RFJ2",
            f"{self.ch_prefix}_RFJ1",
            f"{self.ch_prefix}_LFJ5",
            f"{self.ch_prefix}_LFJ4",
            f"{self.ch_prefix}_LFJ3",
            f"{self.ch_prefix}_LFJ2",
            f"{self.ch_prefix}_LFJ1",
            f"{self.ch_prefix}_THJ5",
            f"{self.ch_prefix}_THJ4",
            f"{self.ch_prefix}_THJ3",
            f"{self.ch_prefix}_THJ2",
            f"{self.ch_prefix}_THJ1",
        ]

    def get_actuated_joint_names(self) -> list[str]:
        return [
            f"{self.ch_prefix}_A_THJ5",
            f"{self.ch_prefix}_A_THJ4",
            f"{self.ch_prefix}_A_THJ3",
            f"{self.ch_prefix}_A_THJ2",
            f"{self.ch_prefix}_A_THJ1",
            f"{self.ch_prefix}_A_FFJ4",
            f"{self.ch_prefix}_A_FFJ3",
            f"{self.ch_prefix}_A_FFJ0",
            f"{self.ch_prefix}_A_MFJ4",
            f"{self.ch_prefix}_A_MFJ3",
            f"{self.ch_prefix}_A_MFJ0",
            f"{self.ch_prefix}_A_RFJ4",
            f"{self.ch_prefix}_A_RFJ3",
            f"{self.ch_prefix}_A_RFJ0",
            f"{self.ch_prefix}_A_LFJ5",
            f"{self.ch_prefix}_A_LFJ4",
            f"{self.ch_prefix}_A_LFJ3",
            f"{self.ch_prefix}_A_LFJ0",
        ]

    def compute_joint_map(self) -> np.ndarray:
        return self._build_joint_map_from_couplings(self._joint_couplings())

    def _joint_name_from_actuated_name(self, actuated_joint_name: str) -> str:
        return actuated_joint_name.replace("_A", "")

    def _joint_couplings(self) -> list[dict[str, dict[str, dict[str, float]]]]:
        return [
            {
                "parent": f"{self.ch_prefix}_A_FFJ0",
                "children": {
                    f"{self.ch_prefix}_FFJ1": {"mult": 0.5},
                    f"{self.ch_prefix}_FFJ2": {"mult": 0.5},
                },
            },
            {
                "parent": f"{self.ch_prefix}_A_MFJ0",
                "children": {
                    f"{self.ch_prefix}_MFJ1": {"mult": 0.5},
                    f"{self.ch_prefix}_MFJ2": {"mult": 0.5},
                },
            },
            {
                "parent": f"{self.ch_prefix}_A_RFJ0",
                "children": {
                    f"{self.ch_prefix}_RFJ1": {"mult": 0.5},
                    f"{self.ch_prefix}_RFJ2": {"mult": 0.5},
                },
            },
            {
                "parent": f"{self.ch_prefix}_A_LFJ0",
                "children": {
                    f"{self.ch_prefix}_LFJ1": {"mult": 0.5},
                    f"{self.ch_prefix}_LFJ2": {"mult": 0.5},
                },
            },
        ]

    def get_neutral_qpos_pose(self) -> np.ndarray:
        return np.zeros(self.num_qpos_dofs, dtype=np.float32)

    def get_neutral_ctrl_pose(self) -> np.ndarray:
        return np.zeros(self.num_actuated_dofs, dtype=np.float32)
