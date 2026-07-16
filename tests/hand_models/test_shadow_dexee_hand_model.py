import pytest

from dexworld.hand_models import ShadowDexeeHandModel
from dexworld.types import HandLandmark
from tests.hand_models.test_robot_hand_models_base import BaseHandModelRegressionSuite


@pytest.fixture
def model(shadow_dexee_hand_path, right):
    return ShadowDexeeHandModel(shadow_dexee_hand_path, right)


@pytest.fixture
def golden(load_golden):
    return load_golden("shadow_dexee_hand_right_golden.npz")


class TestShadowDexeeHandModel(BaseHandModelRegressionSuite):
    EXPECTED_QPOS_DOFS = 12
    EXPECTED_ACTUATED_DOFS = 12
    EXPECTED_FINGERTIPS = 3
    EXPECTED_LANDMARK_ORDER = [
        HandLandmark.THUMB_TIP,
        HandLandmark.INDEX_TIP,
        HandLandmark.MIDDLE_TIP,
    ]
    EXPECTED_JOINT_MAP_SEMANTIC_COUPLINGS = [
        ("F0/J0", "F0/J0", 1.0),
        ("F0/J1", "F0/J1", 1.0),
        ("F0/J2", "F0/J2", 1.0),
        ("F0/J3", "F0/J3", 1.0),
        ("F1/J0", "F1/J0", 1.0),
        ("F1/J1", "F1/J1", 1.0),
        ("F1/J2", "F1/J2", 1.0),
        ("F1/J3", "F1/J3", 1.0),
        ("F2/J0", "F2/J0", 1.0),
        ("F2/J1", "F2/J1", 1.0),
        ("F2/J2", "F2/J2", 1.0),
        ("F2/J3", "F2/J3", 1.0),
    ]
