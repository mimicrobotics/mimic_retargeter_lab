import pytest

from mimic_retargeter_lab.hand_models import OrcaV2HandModel
from mimic_retargeter_lab.types import HandLandmark
from tests.hand_models.test_robot_hand_models_base import BaseHandModelRegressionSuite


@pytest.fixture
def model(orca_v2_hand_path, right):
    return OrcaV2HandModel(orca_v2_hand_path, right)


@pytest.fixture
def golden(load_golden):
    return load_golden("orca_v2_hand_right_golden.npz")


class TestOrcaV2HandModel(BaseHandModelRegressionSuite):
    EXPECTED_QPOS_DOFS = 16
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
        ("right_t-cmc", "right_t-cmc_actuator", 1.0),
        ("right_t-abd", "right_t-abd_actuator", 1.0),
        ("right_t-mcp", "right_t-mcp_actuator", 1.0),
        ("right_t-pip", "right_t-pip_actuator", 1.0),
        ("right_i-abd", "right_i-abd_actuator", 1.0),
        ("right_i-mcp", "right_i-mcp_actuator", 1.0),
        ("right_i-pip", "right_i-pip_actuator", 1.0),
        ("right_m-abd", "right_m-abd_actuator", 1.0),
        ("right_m-mcp", "right_m-mcp_actuator", 1.0),
        ("right_m-pip", "right_m-pip_actuator", 1.0),
        ("right_r-abd", "right_r-abd_actuator", 1.0),
        ("right_r-mcp", "right_r-mcp_actuator", 1.0),
        ("right_r-pip", "right_r-pip_actuator", 1.0),
        ("right_p-abd", "right_p-abd_actuator", 1.0),
        ("right_p-mcp", "right_p-mcp_actuator", 1.0),
        ("right_p-pip", "right_p-pip_actuator", 1.0),
    ]
