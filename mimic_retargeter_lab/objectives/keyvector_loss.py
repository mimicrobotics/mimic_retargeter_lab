from __future__ import annotations

import jax.numpy as jnp

from mimic_retargeter_lab.utils import get_logger


class KeyvectorLoss:
    def __init__(
        self,
        keyvectors_cfg: list[dict[str, str | float | bool]],
    ):
        self.keyvectors_cfg = keyvectors_cfg
        self._logger = get_logger(__name__)

    def __call__(
        self,
        keyvectors_from: dict[str, jnp.ndarray],
        keyvectors_to: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        loss = jnp.array(0.0, dtype=jnp.float32)
        for keyvector_cfg in self.keyvectors_cfg:
            # Get the keyvector from the source key
            try:
                keyvector_from = keyvectors_from[keyvector_cfg["src_key"]]
            except KeyError as e:
                msg = f"Looking for src_key {keyvector_cfg['src_key']} but not found. Available keys: {list(keyvectors_from.keys())}"
                self._logger.error(msg)
                raise e

            # Get the keyvector from the target key
            try:
                keyvector_to = keyvectors_to[keyvector_cfg["tgt_key"]]
            except KeyError as e:
                msg = f"Looking for tgt_key {keyvector_cfg['tgt_key']} but not found. Available keys: {list(keyvectors_to.keys())}"
                self._logger.error(msg)
                raise e

            dist = jnp.linalg.norm(keyvector_from - keyvector_to)
            loss = loss + jnp.asarray(keyvector_cfg["loss_coef"], dtype=jnp.float32) * (
                dist**2
            )

        return loss
