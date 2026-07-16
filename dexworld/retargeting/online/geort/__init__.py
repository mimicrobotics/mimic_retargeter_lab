"""DexWorld-side training adapter for the GeoRT learned retargeter.

Provides an MJCF-native replacement for `geort.env.hand_min.HandKinematicModel`
so GeoRT's trainer can be driven from a `RobotHandModel` (MJCF) rather than
from a URDF + Pinocchio. The IK MLP, FK MLP, and training losses are
inherited from upstream untouched — only the kinematic backend changes.
"""

from .mjcf_kinematic_model import MjcfHandKinematicModel
from .trainer import MjcfGeoRTTrainer

__all__ = [
    "MjcfGeoRTTrainer",
    "MjcfHandKinematicModel",
]
