from __future__ import annotations

from typing import Any
import time

import numpy as np
import jax
import jax.numpy as jnp
import jaxopt

from dexworld.hand_models import BaseHandModel, RobotHandModel
from dexworld.utils import (
    align_pcloud_kabsch_umeyama,
    device_put_attrs,
    get_logger,
    rebuild_mjx_fk_on_device,
    resolve_jax_device,
)
from dexworld.types import HandLandmark
from .base_online_retargeter import BaseOnlineRetargeter


class DexPilotRetargeter(BaseOnlineRetargeter):
    """DexPilot geometric retargeter (JAX / jaxopt SLSQP).

    Matches three groups of vectors between the source hand and the
    target robot hand: wrist->fingertip, thumb->fingertip, and primary
    finger<->finger. Thumb/finger vectors switch between a "reach"
    target (scaled source vector) and a "grasp" target (fixed small
    distance along the source direction) via hysteresis on the source
    distance.

    References:
      - DexPilot: Handa et al., 2019 (https://sites.google.com/view/dex-pilot)
      - dex-retargeting: https://github.com/dexsuite/dex-retargeting
    """

    def __init__(
        self,
        from_model: BaseHandModel,
        to_model: RobotHandModel,
        wrist_mapping: dict[str, str | np.ndarray],
        alignment_landmarks: list[HandLandmark],
        keyvectors_cfg: list[dict[str, Any]],
        retargeter_params: dict[str, Any],
        solver_params: dict[str, Any],
        debug_mode: bool,
        debug_every_n_frames: int,
        precomputed_scale: float | None = None,
        regularized_joints: dict[str, dict[str, float]] | None = None,
    ):
        super().__init__(from_model, to_model, wrist_mapping)
        self._logger = get_logger(__name__)

        # DexPilot algorithm parameters.
        self._project_distance = float(retargeter_params["project_distance"])
        self._escape_distance = float(retargeter_params["escape_distance"])
        self._scale_factor = float(retargeter_params["scale_factor"])
        self._gamma = float(retargeter_params["gamma"])
        self._beta = float(retargeter_params["beta"])

        # SLSQP solver parameters.
        self.tolerance = float(solver_params["tolerance"])
        self.iterations = int(solver_params["iterations"])

        device_str = str(solver_params.get("device", "cpu")).lower()
        self._device = resolve_jax_device(device_str, logger=self._logger)
        self._logger.info(
            f"DexPilotRetargeter using JAX device: {self._device} "
            f"(requested: {device_str!r})"
        )

        # When on GPU, rebuild the hand model's MJX FK on that device so the
        # JIT'd loss doesn't close over CPU-pinned MJX arrays. No-op on CPU.
        rebuild_mjx_fk_on_device(self.to_model, self._device, logger=self._logger)

        # Accepted for YAML parity with KeyvectorRetargeter; DexPilot's joint
        # smoothness comes from `gamma * sum(qpos^2)` in the loss, so this is unused.
        self.regularized_joints = regularized_joints or {}

        self.precomputed_scale = precomputed_scale
        self.alignment_landmarks = alignment_landmarks

        self.keyvectors_cfg = self._normalize_keyvectors_cfg(keyvectors_cfg)
        self._keyvector_names: list[str] = [kv["name"] for kv in self.keyvectors_cfg]
        self._src_keys: list[str] = [kv["src_key"] for kv in self.keyvectors_cfg]
        self._tgt_keys: list[str] = [kv["tgt_key"] for kv in self.keyvectors_cfg]

        # Per-entry static arrays: eta (grasp target distance), active/inactive weights,
        # and per-entry scale. eta = 0.0 means the vector never engages a grasp.
        self._etas = jnp.asarray(
            [kv["eta"] for kv in self.keyvectors_cfg], dtype=jnp.float32
        )
        self._active_weights = jnp.asarray(
            [kv["active_loss_coef"] for kv in self.keyvectors_cfg], dtype=jnp.float32
        )
        self._inactive_weights = jnp.asarray(
            [kv["loss_coef"] for kv in self.keyvectors_cfg], dtype=jnp.float32
        )
        self._scaling_coefs = jnp.asarray(
            [kv["scaling_coef"] for kv in self.keyvectors_cfg], dtype=jnp.float32
        )

        # Hysteresis state for every vector that can engage a grasp (eta > 0 and role == thumb).
        # Finger<->finger engagement is derived from the two adjacent thumb entries each step.
        self._grasp_states: dict[str, bool] = {
            kv["name"]: False
            for kv in self.keyvectors_cfg
            if kv["role"] == "thumb" and kv["eta"] > 0.0
        }

        self._num_dofs = int(self.to_model.num_actuated_dofs)

        self._lower_bounds, self._upper_bounds = self._init_bounds()
        self._bounds_tuple = (
            np.asarray(self._lower_bounds, dtype=np.float64),
            np.asarray(self._upper_bounds, dtype=np.float64),
        )

        # Pin every persistent JAX array we hand to the JIT'd loss on the
        # target device — otherwise per-frame calls trigger host->device
        # copies that dominate wall time.
        device_put_attrs(
            self,
            (
                "_lower_bounds",
                "_upper_bounds",
                "_qpos_prev",
                "_etas",
                "_active_weights",
                "_inactive_weights",
                "_scaling_coefs",
            ),
            self._device,
        )

        with jax.default_device(self._device):
            self._loss_fn = self._init_loss_fn()
            self._solver = jaxopt.ScipyBoundedMinimize(
                fun=self._loss_fn,
                method="SLSQP",
                tol=self.tolerance,
                maxiter=self.iterations,
                jit=True,
            )

        self.debug_mode = bool(debug_mode)
        self.debug_every_n_frames = max(1, int(debug_every_n_frames))
        self._retarget_step_idx = 0
        self._debug_visualization_data: dict[str, np.ndarray] | None = None

    def _normalize_keyvectors_cfg(
        self, keyvectors_cfg: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Fill in per-entry defaults and auto-detect role from tgt_key prefix.

        Role decides grasp behavior:
          - "wrist":  never engages a grasp (eta ignored).
          - "thumb":  tracks its own hysteresis on the source distance.
          - "finger": engaged iff both the two THUMB_TIP_to_<X>_TIP entries
                      that share its endpoints are currently engaged.
        """
        out: list[dict[str, Any]] = []
        for kv in keyvectors_cfg:
            entry = dict(kv)
            tgt_key = entry["tgt_key"]

            role = entry.get("role")
            if role is None:
                if tgt_key.startswith("WRIST_to_"):
                    role = "wrist"
                elif tgt_key.startswith("THUMB_TIP_to_"):
                    role = "thumb"
                else:
                    role = "finger"
            entry["role"] = role

            eta = entry.get("eta")
            entry["eta"] = 0.0 if eta is None or role == "wrist" else float(eta)

            entry["loss_coef"] = float(entry.get("loss_coef", 1.0))
            entry["active_loss_coef"] = float(
                entry.get("active_loss_coef", entry["loss_coef"])
            )
            entry["scaling_coef"] = float(entry.get("scaling_coef", 1.0))
            out.append(entry)
        return out

    def _init_loss_fn(self) -> Any:
        """Compile the DexPilot weighted smooth-L1 loss plus qpos regularizer."""
        keyvector_names = tuple(self._tgt_keys)
        gamma = self._gamma
        beta = self._beta

        def smooth_l1(x: jnp.ndarray) -> jnp.ndarray:
            abs_x = jnp.abs(x)
            return jnp.where(abs_x < beta, 0.5 * x * x / beta, abs_x - 0.5 * beta)

        def loss_fn(
            qpos: jnp.ndarray,
            targets: jnp.ndarray,
            weights: jnp.ndarray,
        ) -> jnp.ndarray:
            qpos_batched = qpos[None, :]
            tgt_kv = self.to_model.compute_keyvectors_jax(
                qpos_batched, joint_space="ctrl"
            )
            tgt_vecs = jnp.stack([tgt_kv[name][0] for name in keyvector_names], axis=0)
            per_vec = jnp.sum(smooth_l1(tgt_vecs - targets), axis=-1)
            loss = jnp.sum(weights * per_vec)
            loss = loss + gamma * jnp.sum(qpos * qpos)
            return loss

        return jax.jit(loss_fn)

    def _optimize_controls(
        self,
        qpos_init_guess: jnp.ndarray,
        targets: jnp.ndarray,
        weights: jnp.ndarray,
    ) -> jnp.ndarray:
        with jax.default_device(self._device):
            # Per-frame inputs from the numpy/CPU path; push to the target
            # device so the JIT'd loss doesn't host->device copy each call.
            targets = jax.device_put(targets, self._device)
            weights = jax.device_put(weights, self._device)
            qpos_clipped = jnp.clip(
                qpos_init_guess, self._lower_bounds, self._upper_bounds
            )
            result = self._solver.run(
                qpos_clipped,
                bounds=self._bounds_tuple,
                targets=targets,
                weights=weights,
            )
            qpos_optimized = jnp.clip(
                result.params, self._lower_bounds, self._upper_bounds
            )
        self._logger.debug(f"Loss: {float(result.state.fun_val)}")
        return qpos_optimized

    def _update_grasp_states(self, src_distances: np.ndarray) -> None:
        """Hysteresis update of per-thumb-finger grasp states."""
        for i, kv in enumerate(self.keyvectors_cfg):
            name = kv["name"]
            if name not in self._grasp_states:
                continue
            dist = float(src_distances[i])
            if not self._grasp_states[name] and dist < self._project_distance:
                self._grasp_states[name] = True
                self._logger.debug(f"Engaging grasp: {name}")
            elif self._grasp_states[name] and dist > self._escape_distance:
                self._grasp_states[name] = False
                self._logger.debug(f"Releasing grasp: {name}")

    def _finger_pair_grasp_active(self, tgt_key: str) -> bool:
        """Finger<->finger engagement is the AND of the two adjacent thumb grasps."""
        a_name, b_name = tgt_key.split("_to_")
        thumb_prefix = f"{HandLandmark.THUMB_TIP.name}_to_"
        return self._grasp_states.get(
            self._thumb_name_for_tgt(thumb_prefix + a_name), False
        ) and self._grasp_states.get(
            self._thumb_name_for_tgt(thumb_prefix + b_name), False
        )

    def _thumb_name_for_tgt(self, thumb_tgt_key: str) -> str:
        """Resolve the `name` of the thumb entry whose tgt_key matches `thumb_tgt_key`."""
        for kv in self.keyvectors_cfg:
            if kv["role"] == "thumb" and kv["tgt_key"] == thumb_tgt_key:
                return kv["name"]
        return thumb_tgt_key  # miss -> .get() returns False

    def _build_targets_and_weights(
        self, src_keyvectors: dict[str, np.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Per-vector targets and weights fed into the jitted loss."""
        src_vecs = np.stack(
            [
                np.asarray(src_keyvectors[key], dtype=np.float32)
                for key in self._src_keys
            ],
            axis=0,
        )
        dists = np.linalg.norm(src_vecs, axis=-1)
        dirs = src_vecs / (dists[..., None] + 1e-6)

        self._update_grasp_states(dists)

        grasp_mask = np.zeros(len(self.keyvectors_cfg), dtype=bool)
        for i, kv in enumerate(self.keyvectors_cfg):
            if kv["eta"] <= 0.0 or kv["role"] == "wrist":
                continue
            if kv["role"] == "thumb":
                grasp_mask[i] = self._grasp_states.get(kv["name"], False)
            elif kv["role"] == "finger":
                grasp_mask[i] = self._finger_pair_grasp_active(kv["tgt_key"])

        etas = np.asarray(self._etas, dtype=np.float32)
        active_w = np.asarray(self._active_weights, dtype=np.float32)
        inactive_w = np.asarray(self._inactive_weights, dtype=np.float32)
        scales = np.asarray(self._scaling_coefs, dtype=np.float32)

        targets_np = np.where(
            grasp_mask[:, None],
            dirs * etas[:, None],
            src_vecs * (scales * self._scale_factor)[:, None],
        )
        weights_np = np.where(grasp_mask, active_w, inactive_w)

        return (
            jnp.asarray(targets_np, dtype=jnp.float32),
            jnp.asarray(weights_np, dtype=jnp.float32),
        )

    def _generate_debug_visualization_data(
        self,
        pcloud: np.ndarray,
        qpos: np.ndarray,
    ) -> dict[str, np.ndarray]:
        src_landmarks = self.from_model.get_landmarks(pcloud)
        tgt_landmarks = self.to_model.get_landmarks(qpos)

        src_segments = self._keyvector_segments(src_landmarks)
        tgt_segments = self._keyvector_segments(tgt_landmarks)

        grasp_flags = np.array(
            [
                self._grasp_states.get(kv["name"], False)
                if kv["role"] == "thumb"
                else self._finger_pair_grasp_active(kv["tgt_key"])
                if kv["role"] == "finger"
                else False
                for kv in self.keyvectors_cfg
            ],
            dtype=bool,
        )
        return {
            "src_raw_points": pcloud.copy(),
            "src_keyvectors": src_segments,
            "tgt_keyvectors": tgt_segments,
            "grasp_active": grasp_flags,
        }

    def _keyvector_segments(
        self, landmarks: dict[HandLandmark, np.ndarray]
    ) -> np.ndarray:
        segments: list[np.ndarray] = []
        for tgt_key in self._tgt_keys:
            if "_to_" not in tgt_key:
                continue
            start_name, end_name = tgt_key.split("_to_", 1)
            try:
                start_lm = HandLandmark[start_name]
                end_lm = HandLandmark[end_name]
            except KeyError:
                continue
            if start_lm not in landmarks or end_lm not in landmarks:
                continue
            start = np.asarray(landmarks[start_lm], dtype=np.float32)
            end = np.asarray(landmarks[end_lm], dtype=np.float32)
            segments.append(np.stack([start, end], axis=0))
        if not segments:
            return np.zeros((0, 2, 3), dtype=np.float32)
        return np.stack(segments, axis=0)

    @property
    def debug_visualization_data(self) -> dict[str, np.ndarray] | None:
        return self._debug_visualization_data

    def retarget(
        self,
        pcloud: np.ndarray,
        wrist_transform: Any | None = None,
    ) -> tuple[Any, Any | None]:
        """Retarget the hand from source point cloud (pcloud) to the target."""
        start_time = time.time()

        if pcloud.ndim == 3 and pcloud.shape[0] == 1:
            pcloud = pcloud[0]
        elif pcloud.ndim != 2:
            raise ValueError(
                f"pcloud must have shape (N, 3), got {tuple(pcloud.shape)}"
            )

        src_landmarks_all = self.from_model.get_landmarks(pcloud)
        tgt_landmarks_all = self.to_model.get_landmarks(
            qpos=np.asarray(self._qpos_prev, dtype=np.float32)
        )
        src_landmarks = np.stack(
            [src_landmarks_all[lm] for lm in self.alignment_landmarks]
        )
        tgt_landmarks = np.stack(
            [tgt_landmarks_all[lm] for lm in self.alignment_landmarks]
        )
        pcloud_aligned, _, _ = align_pcloud_kabsch_umeyama(
            points=pcloud,
            source_landmarks=src_landmarks,
            target_landmarks=tgt_landmarks,
            precomputed_scale=self.precomputed_scale,
        )

        src_keyvectors = self.from_model.compute_keyvectors(pcloud_aligned)
        targets, weights = self._build_targets_and_weights(src_keyvectors)

        qpos_optimized = self._optimize_controls(
            qpos_init_guess=self._qpos_prev,
            targets=targets,
            weights=weights,
        )
        self._qpos_prev = qpos_optimized

        tgt_wrist_transform = None
        if wrist_transform is not None:
            tgt_wrist_transform = self.wrist_retargeter.retarget(wrist_transform)

        if self.debug_mode and (
            self._retarget_step_idx % self.debug_every_n_frames == 0
        ):
            self._debug_visualization_data = self._generate_debug_visualization_data(
                pcloud=pcloud_aligned,
                qpos=np.asarray(qpos_optimized, dtype=np.float32),
            )
        self._retarget_step_idx += 1

        end_time = time.time()
        self._logger.debug(f"Retargeting time: {end_time - start_time:.4f} seconds")

        # Scene/dataset pipeline still expects a leading batch dim; match KeyvectorRetargeter.
        return qpos_optimized[None, :], tgt_wrist_transform
