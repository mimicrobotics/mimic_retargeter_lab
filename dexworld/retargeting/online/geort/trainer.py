"""DexWorld-side GeoRT trainer subclass.

Subclasses upstream `geort.trainer.GeoRTTrainer` to swap out a single
construction site: instead of loading a URDF via Pinocchio, it instantiates
the MJCF-backed `MjcfHandKinematicModel`. Every other aspect of training
(FK MLP architecture, IK MLP architecture, all loss functions, training
loop, checkpoint format) is inherited verbatim — `MjcfGeoRTTrainer` itself
contains no training logic.

`to_model` is intentionally kept *outside* `self.config` because the parent
class persists `self.config` to `config.json` per checkpoint. A
`RobotHandModel` is not JSON-serializable; it lives only on the trainer
instance and is passed through `MjcfHandKinematicModel.build_from_config`
via kwargs.
"""

from __future__ import annotations

from geort.trainer import GeoRTTrainer

from dexworld.hand_models import RobotHandModel

from .mjcf_kinematic_model import MjcfHandKinematicModel


class MjcfGeoRTTrainer(GeoRTTrainer):
    def __init__(self, config: dict, to_model: RobotHandModel):
        # Skip super().__init__ — it would call HandKinematicModel.build_from_config
        # which expects a URDF path. Re-implement the two assignments with the
        # MJCF backend instead.
        self.config = config
        self.hand = MjcfHandKinematicModel.build_from_config(
            self.config, to_model=to_model
        )
