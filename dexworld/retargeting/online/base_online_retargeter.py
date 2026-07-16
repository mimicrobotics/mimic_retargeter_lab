"""Base class for online retargeters."""

from abc import ABC, abstractmethod

import jax.numpy as jnp
import numpy as np

from dexworld.hand_models import HumanHandModel, RobotHandModel
from .wrist_retargeter import WristRetargeter


class BaseOnlineRetargeter(ABC):
    def __init__(
        self,
        from_model: HumanHandModel,
        to_model: RobotHandModel,
        wrist_mapping: dict,
    ):
        self.from_model = from_model
        self.to_model = to_model
        self.wrist_retargeter = WristRetargeter(to_model, wrist_mapping)
        self._actuated_names = self.to_model.get_actuated_joint_names()
        self._qpos_prev = jnp.asarray(
            self.to_model.get_neutral_ctrl_pose(), dtype=jnp.float32
        )

    def _init_bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Per-DoF lower/upper bounds in ctrl space, ordered by actuated joints."""
        act_limits = self.to_model.get_actuated_joint_limits()
        mins, maxs = [], []
        for name in self._actuated_names:
            lo, hi = act_limits[name]
            mins.append(float(lo))
            maxs.append(float(hi))
        return (
            jnp.asarray(mins, dtype=jnp.float32),
            jnp.asarray(maxs, dtype=jnp.float32),
        )

    def reset(self):
        """Reset optimizer state to neutral pose.

        Call between independent episodes so each starts from the same
        initial guess rather than warm-starting from the previous episode.
        """
        self._qpos_prev = jnp.asarray(
            self.to_model.get_neutral_ctrl_pose(), dtype=jnp.float32
        )

    @abstractmethod
    def retarget(
        self, pcloud: np.ndarray, **kwargs
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Retarget the hand from source point cloud (pcloud) to the target.

        Other arguments are possible depending on the retargeter. However,
        it is expected that a pcloud is passed.
        """
        pass
