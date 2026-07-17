from .base_online_retargeter import BaseOnlineRetargeter
from .joint_angle_retargeter import JointAngleRetargeter
from .keyvector_retargeter import KeyvectorRetargeter
from .dexpilot_retargeter import DexPilotRetargeter
from .ako_retargeter import AKORetargeter
from .sampling_based_retargeter import SamplingBasedRetargeter
from .hybrid_retargeter import HybridRetargeter
from .geort_retargeter import GeortRetargeter

from mimic_retargeter_lab.types import Retargeter


RETARGETER_REGISTRY = {
    Retargeter.KEYVECTOR: KeyvectorRetargeter,
    Retargeter.JOINT_ANGLE: JointAngleRetargeter,
    Retargeter.DEXPILOT: DexPilotRetargeter,
    Retargeter.AKO: AKORetargeter,
    Retargeter.SAMPLING_BASED: SamplingBasedRetargeter,
    Retargeter.HYBRID: HybridRetargeter,
    Retargeter.GEORT: GeortRetargeter,
}


def create_retargeter(retargeter_type: Retargeter, **kwargs) -> BaseOnlineRetargeter:
    """Factory function to create online retargeters based on the specified type.

    Kwargs (including ``from_model`` and ``to_model``) are forwarded to the
    chosen retargeter's constructor.
    """
    cls = RETARGETER_REGISTRY.get(retargeter_type)
    if cls is None:
        raise ValueError(f"Unsupported retargeter type: {retargeter_type}")
    return cls(**kwargs)
