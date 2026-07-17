from __future__ import annotations

from pathlib import Path

import numpy as np

from mimic_retargeter_lab.types import Chirality, HandLandmark, MujocoLandmark
from .robot_hand_base import RobotHandModel


class MimicP050HandModel(RobotHandModel):
    def __init__(self, robot_hand_base_path: Path, chirality: Chirality):
        super().__init__(robot_hand_base_path, chirality)

        self.num_fingertips = 5
        self.num_actuated_dofs = len(self.get_actuated_joint_names())
        self.num_qpos_dofs = len(self.get_qpos_joint_names())

        # Defines the landmark link names and their sources
        self._landmark_config: dict[HandLandmark, tuple[str, str]] = {
            # Format: HandLandmark: ("mujoco_name", "body" | "joint" | "site")
            HandLandmark.ARM_ATTACHMENT: MujocoLandmark(
                name="arm_attachment", object_type="body"
            ),
            HandLandmark.WRIST: MujocoLandmark(
                name="wrist_link_virtual", object_type="body"
            ),
            # Fingertips (Bodies)
            HandLandmark.THUMB_TIP: MujocoLandmark(
                name="thumb_fingertip", object_type="body"
            ),
            HandLandmark.INDEX_TIP: MujocoLandmark(
                name="index_fingertip", object_type="body"
            ),
            HandLandmark.MIDDLE_TIP: MujocoLandmark(
                name="middle_fingertip", object_type="body"
            ),
            HandLandmark.RING_TIP: MujocoLandmark(
                name="ring_fingertip", object_type="body"
            ),
            HandLandmark.PINKY_TIP: MujocoLandmark(
                name="pinky_fingertip", object_type="body"
            ),
            # Finger Bases (Joints)
            HandLandmark.THUMB_BASE: MujocoLandmark(
                name="thumb_base2cmc", object_type="joint"
            ),
            HandLandmark.INDEX_BASE: MujocoLandmark(
                name="index_base2mcp", object_type="joint"
            ),
            HandLandmark.PINKY_BASE: MujocoLandmark(
                name="pinky_base2mcp", object_type="joint"
            ),
            # Finger DP (Bodies)
            HandLandmark.THUMB_DP: MujocoLandmark(name="thumb_dp", object_type="body"),
            HandLandmark.INDEX_DP: MujocoLandmark(name="index_dp", object_type="body"),
            HandLandmark.MIDDLE_DP: MujocoLandmark(
                name="middle_dp", object_type="body"
            ),
            HandLandmark.RING_DP: MujocoLandmark(name="ring_dp", object_type="body"),
            HandLandmark.PINKY_DP: MujocoLandmark(name="pinky_dp", object_type="body"),
        }

        self.joint_map = self.compute_joint_map()

        # Compile MJX immediately
        self.create_mjx_kinematic_model()

    # ------------------------------------------------------------------
    # Joint names and coupling
    # ------------------------------------------------------------------
    def get_qpos_joint_names(self) -> list[str]:
        return [
            "thumb_base2cmc",
            "thumb_cmc2mcp",
            "thumb_mcp2pp",
            "thumb_pp2dp_actuated",
            "index_base2mcp",
            "index_mcp2pp",
            "index_pp2mp",
            "index_mp2dp",
            "middle_base2mcp",
            "middle_mcp2pp",
            "middle_pp2mp",
            "middle_mp2dp",
            "ring_base2mcp",
            "ring_mcp2pp",
            "ring_pp2mp",
            "ring_mp2dp",
            "pinky_base2mcp",
            "pinky_mcp2pp",
            "pinky_pp2mp",
            "pinky_mp2dp",
        ]

    def get_actuated_joint_names(self) -> list[str]:
        return [
            "A_thumb_base2cmc",
            "A_thumb_cmc2mcp",
            "A_thumb_mcp2pp",
            "A_thumb_pp2dp_actuated",
            "A_index_base2mcp",
            "A_index_mcp2pp",
            "A_index_pp2mp",
            "A_middle_base2mcp",
            "A_middle_mcp2pp",
            "A_middle_pp2mp",
            "A_ring_base2mcp",
            "A_ring_mcp2pp",
            "A_ring_pp2mp",
            "A_pinky_base2mcp",
            "A_pinky_mcp2pp",
            "A_pinky_pp2mp",
        ]

    def _joint_name_from_actuated_name(self, actuated_joint_name: str) -> str:
        """Strip the 'A_' actuator prefix to get the qpos joint name."""
        if actuated_joint_name.startswith("A_"):
            return actuated_joint_name[2:]
        return actuated_joint_name

    def compute_joint_map(self) -> np.ndarray:
        return self._build_joint_map_from_couplings(self._joint_couplings())

    def _joint_couplings(self) -> list[dict[str, dict[str, dict[str, float]]]]:
        return [
            {"parent": "A_index_pp2mp", "children": {"index_mp2dp": {"mult": 1.0}}},
            {"parent": "A_middle_pp2mp", "children": {"middle_mp2dp": {"mult": 1.0}}},
            {"parent": "A_ring_pp2mp", "children": {"ring_mp2dp": {"mult": 1.0}}},
            {"parent": "A_pinky_pp2mp", "children": {"pinky_mp2dp": {"mult": 1.0}}},
        ]

    # ------------------------------------------------------------------
    # Neutral poses & Setup
    # ------------------------------------------------------------------

    def get_neutral_qpos_pose(self) -> np.ndarray:
        return np.array(
            [
                0.523599,
                -0.349066,
                0.872665,
                0.872665,
                0.148353,
                0.261800,
                0.261800,
                0.261800,
                0.218166,
                0.261800,
                0.261800,
                0.261800,
                0.305433,
                0.261800,
                0.261800,
                0.261800,
                0.392699,
                0.261800,
                0.261800,
                0.261800,
            ],
            dtype=np.float32,
        )

    def get_neutral_ctrl_pose(self) -> np.ndarray:
        return np.array(
            [
                0.523599,
                -0.349066,
                0.872665,
                0.872665,
                0.148353,
                0.261800,
                0.261800,
                0.218166,
                0.261800,
                0.261800,
                0.305433,
                0.261800,
                0.261800,
                0.392699,
                0.261800,
                0.261800,
            ],
            dtype=np.float32,
        )
