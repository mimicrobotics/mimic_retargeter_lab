import pytest

from dexworld.hand_models import LeapHandModel
from dexworld.types import HandLandmark
from tests.hand_models.test_robot_hand_models_base import BaseHandModelRegressionSuite


@pytest.fixture
def model(leap_hand_path, right):
    return LeapHandModel(leap_hand_path, right)


@pytest.fixture
def golden(load_golden):
    return load_golden("leap_hand_right_golden.npz")


class TestLeapHandModel(BaseHandModelRegressionSuite):
    EXPECTED_QPOS_DOFS = 16
    EXPECTED_ACTUATED_DOFS = 16
    EXPECTED_FINGERTIPS = 4
    EXPECTED_LANDMARK_ORDER = [
        HandLandmark.THUMB_TIP,
        HandLandmark.INDEX_TIP,
        HandLandmark.MIDDLE_TIP,
        HandLandmark.RING_TIP,
    ]
    EXPECTED_JOINT_MAP_SEMANTIC_COUPLINGS = [
        ("1", "1_ctrl", 1.0),
        ("0", "0_ctrl", 1.0),
        ("2", "2_ctrl", 1.0),
        ("3", "3_ctrl", 1.0),
        ("5", "5_ctrl", 1.0),
        ("4", "4_ctrl", 1.0),
        ("6", "6_ctrl", 1.0),
        ("7", "7_ctrl", 1.0),
        ("9", "9_ctrl", 1.0),
        ("8", "8_ctrl", 1.0),
        ("10", "10_ctrl", 1.0),
        ("11", "11_ctrl", 1.0),
        ("12", "12_ctrl", 1.0),
        ("13", "13_ctrl", 1.0),
        ("14", "14_ctrl", 1.0),
        ("15", "15_ctrl", 1.0),
    ]
