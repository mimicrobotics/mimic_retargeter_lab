import pytest

from mimic_retargeter_lab.hand_models import WujiHandModel
from mimic_retargeter_lab.types import HandLandmark
from tests.hand_models.test_robot_hand_models_base import BaseHandModelRegressionSuite


@pytest.fixture
def model(wuji_hand_path, right):
    return WujiHandModel(wuji_hand_path, right)


@pytest.fixture
def golden(load_golden):
    return load_golden("wuji_hand_right_golden.npz")


class TestWujiHandModel(BaseHandModelRegressionSuite):
    EXPECTED_QPOS_DOFS = 20
    EXPECTED_ACTUATED_DOFS = 20
    EXPECTED_FINGERTIPS = 5
    EXPECTED_LANDMARK_ORDER = [
        HandLandmark.THUMB_TIP,
        HandLandmark.INDEX_TIP,
        HandLandmark.MIDDLE_TIP,
        HandLandmark.RING_TIP,
        HandLandmark.PINKY_TIP,
    ]
    EXPECTED_JOINT_MAP_SEMANTIC_COUPLINGS = [
        ("right_finger1_joint1", "right_finger1_joint1_actuator", 1.0),
        ("right_finger1_joint2", "right_finger1_joint2_actuator", 1.0),
        ("right_finger1_joint3", "right_finger1_joint3_actuator", 1.0),
        ("right_finger1_joint4", "right_finger1_joint4_actuator", 1.0),
        ("right_finger2_joint1", "right_finger2_joint1_actuator", 1.0),
        ("right_finger2_joint2", "right_finger2_joint2_actuator", 1.0),
        ("right_finger2_joint3", "right_finger2_joint3_actuator", 1.0),
        ("right_finger2_joint4", "right_finger2_joint4_actuator", 1.0),
        ("right_finger3_joint1", "right_finger3_joint1_actuator", 1.0),
        ("right_finger3_joint2", "right_finger3_joint2_actuator", 1.0),
        ("right_finger3_joint3", "right_finger3_joint3_actuator", 1.0),
        ("right_finger3_joint4", "right_finger3_joint4_actuator", 1.0),
        ("right_finger4_joint1", "right_finger4_joint1_actuator", 1.0),
        ("right_finger4_joint2", "right_finger4_joint2_actuator", 1.0),
        ("right_finger4_joint3", "right_finger4_joint3_actuator", 1.0),
        ("right_finger4_joint4", "right_finger4_joint4_actuator", 1.0),
        ("right_finger5_joint1", "right_finger5_joint1_actuator", 1.0),
        ("right_finger5_joint2", "right_finger5_joint2_actuator", 1.0),
        ("right_finger5_joint3", "right_finger5_joint3_actuator", 1.0),
        ("right_finger5_joint4", "right_finger5_joint4_actuator", 1.0),
    ]
