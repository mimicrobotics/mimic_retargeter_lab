from pathlib import Path

import numpy as np

from mimic_retargeter_lab.types import Chirality, HandLandmark, MujocoLandmark
from .robot_hand_base import RobotHandModel


class LeapHandModel(RobotHandModel):
    def __init__(self, robot_base_path: Path, chirality: Chirality):
        super().__init__(robot_base_path, chirality)
        self.ch_prefix = "R" if chirality == Chirality.RIGHT else "L"

        self.num_fingertips = 4
        self.num_qpos_dofs = len(self.get_qpos_joint_names())
        self.num_actuated_dofs = len(self.get_actuated_joint_names())

        self._landmark_config: dict[HandLandmark, MujocoLandmark] = {
            HandLandmark.ARM_ATTACHMENT: MujocoLandmark(
                name="arm_attachment", object_type="body"
            ),
            HandLandmark.WRIST: MujocoLandmark(name="base", object_type="body"),
            # Fingertips (tip_head bodies from URDF)
            HandLandmark.THUMB_TIP: MujocoLandmark(
                name="thumb_tip_head", object_type="body"
            ),
            HandLandmark.INDEX_TIP: MujocoLandmark(
                name="index_tip_head", object_type="body"
            ),
            HandLandmark.MIDDLE_TIP: MujocoLandmark(
                name="middle_tip_head", object_type="body"
            ),
            HandLandmark.RING_TIP: MujocoLandmark(
                name="ring_tip_head", object_type="body"
            ),
            # Finger bases (first phalanx of each finger)
            HandLandmark.THUMB_BASE: MujocoLandmark(
                name="thumb_temp_base", object_type="body"
            ),
            HandLandmark.INDEX_BASE: MujocoLandmark(
                name="mcp_joint", object_type="body"
            ),
            HandLandmark.MIDDLE_BASE: MujocoLandmark(
                name="mcp_joint_2", object_type="body"
            ),
            HandLandmark.RING_BASE: MujocoLandmark(
                name="mcp_joint_3", object_type="body"
            ),
            # Distal phalanx bodies (one joint before tip)
            HandLandmark.THUMB_DP: MujocoLandmark(name="thumb_dip", object_type="body"),
            HandLandmark.INDEX_DP: MujocoLandmark(name="dip", object_type="body"),
            HandLandmark.MIDDLE_DP: MujocoLandmark(name="dip_2", object_type="body"),
            HandLandmark.RING_DP: MujocoLandmark(name="dip_3", object_type="body"),
        }

        self.joint_map = self.compute_joint_map()

        # Compile MJX kinematic model
        self.create_mjx_kinematic_model()

    def get_qpos_joint_names(self) -> list[str]:
        """Joint names in body-tree traversal order (qpos order)."""
        return [
            "1",  # index mcp
            "0",  # index rot
            "2",  # index pip
            "3",  # index dip
            "5",  # middle mcp
            "4",  # middle rot
            "6",  # middle pip
            "7",  # middle dip
            "9",  # ring mcp
            "8",  # ring rot
            "10",  # ring pip
            "11",  # ring dip
            "12",  # thumb cmc
            "13",  # thumb axl
            "14",  # thumb mcp
            "15",  # thumb ipl
        ]

    def _joint_name_from_actuated_name(self, actuated_joint_name: str) -> str:
        """Strip '_ctrl' suffix to get the qpos joint name."""
        if actuated_joint_name.endswith("_ctrl"):
            return actuated_joint_name[:-5]
        return actuated_joint_name

    def get_actuated_joint_names(self) -> list[str]:
        """Actuator names matching the MJCF actuator declaration order."""
        return [
            "1_ctrl",  # index mcp
            "0_ctrl",  # index rot
            "2_ctrl",  # index pip
            "3_ctrl",  # index dip
            "5_ctrl",  # middle mcp
            "4_ctrl",  # middle rot
            "6_ctrl",  # middle pip
            "7_ctrl",  # middle dip
            "9_ctrl",  # ring mcp
            "8_ctrl",  # ring rot
            "10_ctrl",  # ring pip
            "11_ctrl",  # ring dip
            "12_ctrl",  # thumb cmc
            "13_ctrl",  # thumb axl
            "14_ctrl",  # thumb mcp
            "15_ctrl",  # thumb ipl
        ]

    def compute_joint_map(self) -> np.ndarray:
        return np.eye(self.num_qpos_dofs, dtype=np.float32)
