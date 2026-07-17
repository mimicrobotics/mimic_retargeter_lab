from pathlib import Path

import numpy as np

from mimic_retargeter_lab.types import Chirality, HandLandmark, MujocoLandmark
from .robot_hand_base import RobotHandModel


class OrcaV2HandModel(RobotHandModel):
    def __init__(self, robot_base_path: Path, chirality: Chirality):
        super().__init__(robot_base_path, chirality)
        self.ch_prefix = "right" if chirality == Chirality.RIGHT else "left"

        self.num_fingertips = 5
        self.num_qpos_dofs = len(self.get_qpos_joint_names())
        self.num_actuated_dofs = len(self.get_actuated_joint_names())

        self._landmark_config: dict[HandLandmark, MujocoLandmark] = {
            HandLandmark.ARM_ATTACHMENT: MujocoLandmark(
                name="arm_attachment", object_type="body"
            ),
            HandLandmark.PALM: MujocoLandmark(
                name="right_R-Carpals_8d1f1041", object_type="body"
            ),
            HandLandmark.WRIST: MujocoLandmark(
                name="right_R-Carpals_8d1f1041", object_type="body"
            ),
            # Fingertips (sites at the actual tip of the mesh)
            HandLandmark.THUMB_TIP: MujocoLandmark(
                name="right_thumb_tip", object_type="site"
            ),
            HandLandmark.INDEX_TIP: MujocoLandmark(
                name="right_index_tip", object_type="site"
            ),
            HandLandmark.MIDDLE_TIP: MujocoLandmark(
                name="right_middle_tip", object_type="site"
            ),
            HandLandmark.RING_TIP: MujocoLandmark(
                name="right_ring_tip", object_type="site"
            ),
            HandLandmark.PINKY_TIP: MujocoLandmark(
                name="right_pinky_tip", object_type="site"
            ),
            # Finger bases (first actuated body of each finger)
            HandLandmark.THUMB_BASE: MujocoLandmark(
                name="right_T-TP-R_1c2b802d", object_type="body"
            ),
            HandLandmark.INDEX_BASE: MujocoLandmark(
                name="right_i-abd", object_type="joint"
            ),
            HandLandmark.MIDDLE_BASE: MujocoLandmark(
                name="right_m-abd", object_type="joint"
            ),
            HandLandmark.RING_BASE: MujocoLandmark(
                name="right_r-abd", object_type="joint"
            ),
            HandLandmark.PINKY_BASE: MujocoLandmark(
                name="right_p-abd", object_type="joint"
            ),
            # Distal phalanx bodies (one joint before tip)
            HandLandmark.THUMB_DP: MujocoLandmark(
                name="right_T-DP_b7429e50", object_type="body"
            ),
            HandLandmark.INDEX_DP: MujocoLandmark(
                name="right_I-FingerTipAssembly_ec49c16c", object_type="body"
            ),
            HandLandmark.MIDDLE_DP: MujocoLandmark(
                name="right_M-FingerTipAssembly_34afb748", object_type="body"
            ),
            HandLandmark.RING_DP: MujocoLandmark(
                name="right_M-FingerTipAssembly_424a8e75", object_type="body"
            ),
            HandLandmark.PINKY_DP: MujocoLandmark(
                name="right_P-FingerTipAssembly_cd219176", object_type="body"
            ),
        }

        self.joint_map = self.compute_joint_map()
        self.create_mjx_kinematic_model()

    def get_qpos_joint_names(self) -> list[str]:
        return [
            "right_t-cmc",
            "right_t-abd",
            "right_t-mcp",
            "right_t-pip",
            "right_i-abd",
            "right_i-mcp",
            "right_i-pip",
            "right_m-abd",
            "right_m-mcp",
            "right_m-pip",
            "right_r-abd",
            "right_r-mcp",
            "right_r-pip",
            "right_p-abd",
            "right_p-mcp",
            "right_p-pip",
        ]

    def _joint_name_from_actuated_name(self, actuated_joint_name: str) -> str:
        """Strip '_actuator' suffix to get the qpos joint name."""
        if actuated_joint_name.endswith("_actuator"):
            return actuated_joint_name[: -len("_actuator")]
        return actuated_joint_name

    def get_actuated_joint_names(self) -> list[str]:
        return [
            "right_t-cmc_actuator",
            "right_t-abd_actuator",
            "right_t-mcp_actuator",
            "right_t-pip_actuator",
            "right_i-abd_actuator",
            "right_i-mcp_actuator",
            "right_i-pip_actuator",
            "right_m-abd_actuator",
            "right_m-mcp_actuator",
            "right_m-pip_actuator",
            "right_r-abd_actuator",
            "right_r-mcp_actuator",
            "right_r-pip_actuator",
            "right_p-abd_actuator",
            "right_p-mcp_actuator",
            "right_p-pip_actuator",
        ]

    def compute_joint_map(self) -> np.ndarray:
        return np.eye(self.num_qpos_dofs, dtype=np.float32)
