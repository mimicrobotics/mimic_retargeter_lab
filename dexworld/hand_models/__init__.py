from .wonik_allegro_hand import WonikAllegroHandModel
from .base_hand import BaseHandModel
from .mano_keypoint_hand import ManoKeypointHandModel
from .manus_hand import ManusHandModel
from .human_hand_base import HumanHandModel
from .leap_hand import LeapHandModel
from .wuji_hand import WujiHandModel
from .orca_v2_hand import OrcaV2HandModel
from .mimic_p050_hand import MimicP050HandModel
from .robot_hand_base import RobotHandModel
from .shadow_dexee_hand import ShadowDexeeHandModel
from .shadow_hand import ShadowHandModel

from dexworld.types import HumanHandType, RobotHandType


# --- HUMAN HANDS ---
HUMAN_HAND_REGISTRY = {
    HumanHandType.MANO_KEYPOINT_HAND: ManoKeypointHandModel,
    HumanHandType.MANUS_HAND: ManusHandModel,
}


def create_human_hand(hand_type: HumanHandType, chirality, **kwargs) -> HumanHandModel:
    """Factory function to create human hand models based on the specified type."""
    model_class = HUMAN_HAND_REGISTRY.get(hand_type)
    if not model_class:
        raise ValueError(f"Unsupported human hand type: {hand_type}")
    return model_class(chirality=chirality, **kwargs)


# --- ROBOT HANDS ---
ROBOT_HAND_REGISTRY = {
    RobotHandType.SHADOW_HAND: ShadowHandModel,
    RobotHandType.SHADOW_DEXEE_HAND: ShadowDexeeHandModel,
    RobotHandType.MIMIC_P050_HAND: MimicP050HandModel,
    RobotHandType.WONIK_ALLEGRO_HAND: WonikAllegroHandModel,
    RobotHandType.LEAP_HAND: LeapHandModel,
    RobotHandType.WUJI_HAND: WujiHandModel,
    RobotHandType.ORCA_V2_HAND: OrcaV2HandModel,
}


def create_robot_hand(hand_type: RobotHandType, hand_path, chirality) -> RobotHandModel:
    """Factory function to create robot hand models based on the specified type."""
    model_class = ROBOT_HAND_REGISTRY.get(hand_type)
    if not model_class:
        raise ValueError(f"Unsupported robot hand type: {hand_type}")
    return model_class(hand_path, chirality)
