import pytest

from dexworld.hand_models import WonikAllegroHandModel
from dexworld.types import HandLandmark
from tests.hand_models.test_robot_hand_models_base import BaseHandModelRegressionSuite


@pytest.fixture
def model(allegro_hand_path, right):
    return WonikAllegroHandModel(allegro_hand_path, right)


@pytest.fixture
def golden(load_golden):
    return load_golden("wonik_allegro_hand_right_golden.npz")


class TestWonikAllegroHandModel(BaseHandModelRegressionSuite):
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
        ("ffj0", "ffa0", 1.0),
        ("ffj1", "ffa1", 1.0),
        ("ffj2", "ffa2", 1.0),
        ("ffj3", "ffa3", 1.0),
        ("mfj0", "mfa0", 1.0),
        ("mfj1", "mfa1", 1.0),
        ("mfj2", "mfa2", 1.0),
        ("mfj3", "mfa3", 1.0),
        ("rfj0", "rfa0", 1.0),
        ("rfj1", "rfa1", 1.0),
        ("rfj2", "rfa2", 1.0),
        ("rfj3", "rfa3", 1.0),
        ("thj0", "tha0", 1.0),
        ("thj1", "tha1", 1.0),
        ("thj2", "tha2", 1.0),
        ("thj3", "tha3", 1.0),
    ]
