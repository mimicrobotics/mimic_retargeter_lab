from pathlib import Path

import numpy as np

from dexworld.types import Chirality, HandLandmark, MujocoLandmark
from .robot_hand_base import RobotHandModel


class WujiHandModel(RobotHandModel):
    def __init__(self, robot_base_path: Path, chirality: Chirality):
        super().__init__(robot_base_path, chirality)

        self.num_fingertips = 5
        self.num_qpos_dofs = len(self.get_qpos_joint_names())
        self.num_actuated_dofs = len(self.get_actuated_joint_names())

        self._landmark_config: dict[HandLandmark, MujocoLandmark] = {
            HandLandmark.ARM_ATTACHMENT: MujocoLandmark(
                name="arm_attachment", object_type="body"
            ),
            HandLandmark.WRIST: MujocoLandmark(
                name="right_palm_link", object_type="body"
            ),
            # Fingertips (tip link bodies from URDF fixed joints)
            HandLandmark.THUMB_TIP: MujocoLandmark(
                name="right_finger1_tip_link", object_type="body"
            ),
            HandLandmark.INDEX_TIP: MujocoLandmark(
                name="right_finger2_tip_link", object_type="body"
            ),
            HandLandmark.MIDDLE_TIP: MujocoLandmark(
                name="right_finger3_tip_link", object_type="body"
            ),
            HandLandmark.RING_TIP: MujocoLandmark(
                name="right_finger4_tip_link", object_type="body"
            ),
            HandLandmark.PINKY_TIP: MujocoLandmark(
                name="right_finger5_tip_link", object_type="body"
            ),
            # Finger bases (first link of each finger)
            HandLandmark.THUMB_BASE: MujocoLandmark(
                name="right_finger1_link1", object_type="body"
            ),
            HandLandmark.INDEX_BASE: MujocoLandmark(
                name="right_finger2_link1", object_type="body"
            ),
            HandLandmark.MIDDLE_BASE: MujocoLandmark(
                name="right_finger3_link1", object_type="body"
            ),
            HandLandmark.RING_BASE: MujocoLandmark(
                name="right_finger4_link1", object_type="body"
            ),
            HandLandmark.PINKY_BASE: MujocoLandmark(
                name="right_finger5_link1", object_type="body"
            ),
            # Distal phalanx bodies (one joint before tip)
            HandLandmark.THUMB_DP: MujocoLandmark(
                name="right_finger1_link4", object_type="body"
            ),
            HandLandmark.INDEX_DP: MujocoLandmark(
                name="right_finger2_link4", object_type="body"
            ),
            HandLandmark.MIDDLE_DP: MujocoLandmark(
                name="right_finger3_link4", object_type="body"
            ),
            HandLandmark.RING_DP: MujocoLandmark(
                name="right_finger4_link4", object_type="body"
            ),
            HandLandmark.PINKY_DP: MujocoLandmark(
                name="right_finger5_link4", object_type="body"
            ),
        }

        self.joint_map = self.compute_joint_map()

        # Compile MJX kinematic model
        self.create_mjx_kinematic_model()

    def get_qpos_joint_names(self) -> list[str]:
        """Joint names in body-tree traversal order (qpos order)."""
        return [
            # Thumb (finger1)
            "right_finger1_joint1",
            "right_finger1_joint2",
            "right_finger1_joint3",
            "right_finger1_joint4",
            # Index (finger2)
            "right_finger2_joint1",
            "right_finger2_joint2",
            "right_finger2_joint3",
            "right_finger2_joint4",
            # Middle (finger3)
            "right_finger3_joint1",
            "right_finger3_joint2",
            "right_finger3_joint3",
            "right_finger3_joint4",
            # Ring (finger4)
            "right_finger4_joint1",
            "right_finger4_joint2",
            "right_finger4_joint3",
            "right_finger4_joint4",
            # Pinky (finger5)
            "right_finger5_joint1",
            "right_finger5_joint2",
            "right_finger5_joint3",
            "right_finger5_joint4",
        ]

    def _joint_name_from_actuated_name(self, actuated_joint_name: str) -> str:
        """Strip '_actuator' suffix to get the qpos joint name."""
        if actuated_joint_name.endswith("_actuator"):
            return actuated_joint_name[: -len("_actuator")]
        return actuated_joint_name

    def get_actuated_joint_names(self) -> list[str]:
        """Actuator names matching the MJCF actuator declaration order."""
        return [
            # Thumb (finger1)
            "right_finger1_joint1_actuator",
            "right_finger1_joint2_actuator",
            "right_finger1_joint3_actuator",
            "right_finger1_joint4_actuator",
            # Index (finger2)
            "right_finger2_joint1_actuator",
            "right_finger2_joint2_actuator",
            "right_finger2_joint3_actuator",
            "right_finger2_joint4_actuator",
            # Middle (finger3)
            "right_finger3_joint1_actuator",
            "right_finger3_joint2_actuator",
            "right_finger3_joint3_actuator",
            "right_finger3_joint4_actuator",
            # Ring (finger4)
            "right_finger4_joint1_actuator",
            "right_finger4_joint2_actuator",
            "right_finger4_joint3_actuator",
            "right_finger4_joint4_actuator",
            # Pinky (finger5)
            "right_finger5_joint1_actuator",
            "right_finger5_joint2_actuator",
            "right_finger5_joint3_actuator",
            "right_finger5_joint4_actuator",
        ]

    def compute_joint_map(self) -> np.ndarray:
        return np.eye(self.num_qpos_dofs, dtype=np.float32)
