import pytest

from mimic_retargeter_lab.hand_models import ShadowHandModel
from mimic_retargeter_lab.types import HandLandmark
from tests.hand_models.test_robot_hand_models_base import BaseHandModelRegressionSuite


@pytest.fixture
def model(shadow_hand_path, right):
    return ShadowHandModel(shadow_hand_path, right)


@pytest.fixture
def golden(load_golden):
    return load_golden("shadow_hand_right_golden.npz")


class TestShadowHandModel(BaseHandModelRegressionSuite):
    EXPECTED_QPOS_DOFS = 22
    EXPECTED_ACTUATED_DOFS = 18
    EXPECTED_FINGERTIPS = 5
    EXPECTED_LANDMARK_ORDER = [
        HandLandmark.THUMB_TIP,
        HandLandmark.INDEX_TIP,
        HandLandmark.MIDDLE_TIP,
        HandLandmark.RING_TIP,
        HandLandmark.PINKY_TIP,
    ]
    EXPECTED_JOINT_MAP_SEMANTIC_COUPLINGS = [
        ("rh_THJ5", "rh_A_THJ5", 1.0),
        ("rh_THJ4", "rh_A_THJ4", 1.0),
        ("rh_THJ3", "rh_A_THJ3", 1.0),
        ("rh_THJ2", "rh_A_THJ2", 1.0),
        ("rh_THJ1", "rh_A_THJ1", 1.0),
        ("rh_FFJ4", "rh_A_FFJ4", 1.0),
        ("rh_FFJ3", "rh_A_FFJ3", 1.0),
        ("rh_FFJ2", "rh_A_FFJ0", 0.5),
        ("rh_FFJ1", "rh_A_FFJ0", 0.5),
        ("rh_MFJ4", "rh_A_MFJ4", 1.0),
        ("rh_MFJ3", "rh_A_MFJ3", 1.0),
        ("rh_MFJ2", "rh_A_MFJ0", 0.5),
        ("rh_MFJ1", "rh_A_MFJ0", 0.5),
        ("rh_RFJ4", "rh_A_RFJ4", 1.0),
        ("rh_RFJ3", "rh_A_RFJ3", 1.0),
        ("rh_RFJ2", "rh_A_RFJ0", 0.5),
        ("rh_RFJ1", "rh_A_RFJ0", 0.5),
        ("rh_LFJ5", "rh_A_LFJ5", 1.0),
        ("rh_LFJ4", "rh_A_LFJ4", 1.0),
        ("rh_LFJ3", "rh_A_LFJ3", 1.0),
        ("rh_LFJ2", "rh_A_LFJ0", 0.5),
        ("rh_LFJ1", "rh_A_LFJ0", 0.5),
    ]
