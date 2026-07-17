import pytest

from mimic_retargeter_lab.hand_models import MimicP050HandModel
from mimic_retargeter_lab.types import HandLandmark
from tests.hand_models.test_robot_hand_models_base import BaseHandModelRegressionSuite


@pytest.fixture
def model(mimic_hand_path, right):
    return MimicP050HandModel(mimic_hand_path, right)


@pytest.fixture
def golden(load_golden):
    return load_golden("mimic_p050_hand_right_golden.npz")


class TestMimicP050HandModel(BaseHandModelRegressionSuite):
    EXPECTED_QPOS_DOFS = 20
    EXPECTED_ACTUATED_DOFS = 16
    EXPECTED_FINGERTIPS = 5
    EXPECTED_LANDMARK_ORDER = [
        HandLandmark.THUMB_TIP,
        HandLandmark.INDEX_TIP,
        HandLandmark.MIDDLE_TIP,
        HandLandmark.RING_TIP,
        HandLandmark.PINKY_TIP,
    ]
    EXPECTED_JOINT_MAP_SEMANTIC_COUPLINGS = [
        ("thumb_base2cmc", "A_thumb_base2cmc", 1.0),
        ("thumb_cmc2mcp", "A_thumb_cmc2mcp", 1.0),
        ("thumb_mcp2pp", "A_thumb_mcp2pp", 1.0),
        ("thumb_pp2dp_actuated", "A_thumb_pp2dp_actuated", 1.0),
        ("index_base2mcp", "A_index_base2mcp", 1.0),
        ("index_mcp2pp", "A_index_mcp2pp", 1.0),
        ("index_pp2mp", "A_index_pp2mp", 1.0),
        ("index_mp2dp", "A_index_pp2mp", 1.0),
        ("middle_base2mcp", "A_middle_base2mcp", 1.0),
        ("middle_mcp2pp", "A_middle_mcp2pp", 1.0),
        ("middle_pp2mp", "A_middle_pp2mp", 1.0),
        ("middle_mp2dp", "A_middle_pp2mp", 1.0),
        ("ring_base2mcp", "A_ring_base2mcp", 1.0),
        ("ring_mcp2pp", "A_ring_mcp2pp", 1.0),
        ("ring_pp2mp", "A_ring_pp2mp", 1.0),
        ("ring_mp2dp", "A_ring_pp2mp", 1.0),
        ("pinky_base2mcp", "A_pinky_base2mcp", 1.0),
        ("pinky_mcp2pp", "A_pinky_mcp2pp", 1.0),
        ("pinky_pp2mp", "A_pinky_pp2mp", 1.0),
        ("pinky_mp2dp", "A_pinky_pp2mp", 1.0),
    ]
