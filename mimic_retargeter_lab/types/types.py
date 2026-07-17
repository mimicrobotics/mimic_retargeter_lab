import enum
from typing import NamedTuple, Literal


class HumanHandType(str, enum.Enum):
    MANO_KEYPOINT_HAND = "mano_keypoint_hand"
    MANUS_HAND = "manus_hand"


class RobotHandType(str, enum.Enum):
    SHADOW_HAND = "shadow_hand"
    SHADOW_DEXEE_HAND = "shadow_dexee_hand"
    WONIK_ALLEGRO_HAND = "wonik_allegro_hand"
    MIMIC_P050_HAND = "mimic_p050_hand"
    LEAP_HAND = "leap_hand"
    WUJI_HAND = "wuji_hand"
    ORCA_V2_HAND = "orca_v2_hand"


class Chirality(enum.Enum):
    LEFT = "left"
    RIGHT = "right"


class Simulator(enum.Enum):
    MUJOCO = "mujoco"


class Scene(enum.Enum):
    HAND_ONLY = "hand_only"
    OFFLINE_RETARGET = "offline_retarget"


class HandDataset(enum.Enum):
    DEXYCB = "dexycb"
    WILOR_TEST_LONG = "wilor_test_long"
    WILOR_TEST_IMAGES = "wilor_test_images"
    WILOR_TEST_INDEX_MCP = "wilor_test_index_mcp"
    WILOR_TEST_THUMB = "wilor_thumb_test"
    KEYVECTOR_MATCHING_TEST = "keyvector_matching_test"
    PINCH_GRASPS_TEST = "pinch_grasps_test"
    HUMAN_HAND_WORKSPACE = "human_hand_workspace"


class Retargeter(enum.Enum):
    KEYVECTOR = "keyvector"
    JOINT_ANGLE = "joint_angle"
    DEXPILOT = "dexpilot"
    AKO = "ako"
    SAMPLING_BASED = "sampling_based"
    HYBRID = "hybrid"
    GEORT = "geort"


class HandLandmark(str, enum.Enum):
    WRIST = "wrist"
    PALM = "palm"
    THUMB_BASE = "thumb_base"
    INDEX_BASE = "index_base"
    MIDDLE_BASE = "middle_base"
    RING_BASE = "ring_base"
    PINKY_BASE = "pinky_base"

    # Distal phalanx bases (one joint proximal to the tip; pairs with *_TIP
    # to give a fingertip-orientation direction).
    THUMB_DP = "thumb_dp"
    INDEX_DP = "index_dp"
    MIDDLE_DP = "middle_dp"
    RING_DP = "ring_dp"
    PINKY_DP = "pinky_dp"

    # Fingertips
    THUMB_TIP = "thumb_tip"
    INDEX_TIP = "index_tip"
    MIDDLE_TIP = "middle_tip"
    RING_TIP = "ring_tip"
    PINKY_TIP = "pinky_tip"

    # Point which will be controlled to follow the human wrist pose command.
    ARM_ATTACHMENT = "arm_attachment"

    @classmethod
    def valid_choices(cls) -> list[str]:
        """Returns a list of valid hand landmarks."""
        return [e.value for e in cls] + [e.value.upper() for e in cls]

    def __str__(self) -> str:
        """Returns the string representation of the hand landmark."""
        return self.value


class MujocoLandmark(NamedTuple):
    name: str
    object_type: Literal["joint", "body", "site"]
