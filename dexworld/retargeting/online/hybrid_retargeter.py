"""Hybrid (warmstart) retargeter.

Composes a `JointAngleRetargeter` (analytical) with a `KeyvectorRetargeter`
(SLSQP optimization). Each frame the analytical pass produces a qpos that
is injected as the keyvector solver's initial guess, replacing its usual
``_qpos_prev`` warmstart. The keyvector pass then refines fingertip
placement from that closer-to-optimum starting point.
"""

from __future__ import annotations

import types
from typing import Any

import jax.numpy as jnp
import numpy as np

from dexworld.hand_models import HumanHandModel, RobotHandModel

from .base_online_retargeter import BaseOnlineRetargeter
from .joint_angle_retargeter import JointAngleRetargeter
from .keyvector_retargeter import KeyvectorRetargeter


class HybridRetargeter(BaseOnlineRetargeter):
    def __init__(
        self,
        from_model: HumanHandModel,
        to_model: RobotHandModel,
        wrist_mapping: dict,
        joint_angle: dict,
        keyvector: dict,
        device: str = "cpu",
    ):
        super().__init__(from_model, to_model, wrist_mapping)

        # The top-level `device` controls both children, overriding any
        # `device` set inside the nested `joint_angle` / `keyvector` dicts —
        # there's only one device for the whole hybrid pipeline.
        joint_angle = {k: v for k, v in joint_angle.items() if k != "device"}
        keyvector = {k: v for k, v in keyvector.items() if k != "device"}

        self._joint_angle = JointAngleRetargeter(
            from_model=from_model,
            to_model=to_model,
            wrist_mapping=wrist_mapping,
            device=device,
            **joint_angle,
        )
        self._keyvector = KeyvectorRetargeter(
            from_model=from_model,
            to_model=to_model,
            wrist_mapping=wrist_mapping,
            device=device,
            **keyvector,
        )

        # `_device` is read by the Latency metric
        # (utils/retarget_utils.py:184) to label per-episode timings.
        self._device = types.SimpleNamespace(platform=device)

    def reset(self):
        super().reset()
        self._joint_angle.reset()
        self._keyvector.reset()

    @property
    def debug_visualization_data(self) -> dict[str, np.ndarray] | None:
        return self._keyvector.debug_visualization_data

    def retarget(
        self,
        pcloud: np.ndarray,
        wrist_transform: Any | None = None,
    ) -> tuple[Any, Any | None]:
        # Analytical pass: produces a (1, num_actuated_dofs) qpos in ctrl space.
        qpos_warmstart, _ = self._joint_angle.retarget(pcloud, wrist_transform=None)
        qpos_init = jnp.asarray(np.asarray(qpos_warmstart)[0], dtype=jnp.float32)

        # Keyvector pass: SLSQP from the analytical guess (not _qpos_prev).
        qpos_out, _ = self._keyvector.retarget(
            pcloud, wrist_transform=None, qpos_init=qpos_init
        )

        tgt_wrist_transform = None
        if wrist_transform is not None:
            tgt_wrist_transform = self.wrist_retargeter.retarget(wrist_transform)

        return qpos_out, tgt_wrist_transform
