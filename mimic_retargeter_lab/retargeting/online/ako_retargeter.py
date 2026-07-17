from __future__ import annotations

from typing import Any
import time

import numpy as np
import jax
import jax.numpy as jnp
import jaxopt

from mimic_retargeter_lab.hand_models import BaseHandModel, RobotHandModel
from mimic_retargeter_lab.utils import (
    align_pcloud_kabsch_umeyama,
    device_put_attrs,
    get_logger,
    rebuild_mjx_fk_on_device,
    resolve_jax_device,
)
from mimic_retargeter_lab.types import HandLandmark
from .base_online_retargeter import BaseOnlineRetargeter


class AKORetargeter(BaseOnlineRetargeter):
    """Analyzing Key Objectives (AKO) geometric retargeter (JAX / jaxopt SLSQP).

    Matches weighted keyvector pairs between the source hand and the target
    robot hand with a smooth-L1 (Huber) loss on vector-magnitude errors, plus
    two mechanisms that make contact-oriented retargeting feel more natural:

      - Switching weights: sigmoids on source pinch distances that gate
        wrist->tip vectors OFF and thumb->tip ("pinch") vectors ON as the
        source hand closes. Derived each frame from the source — no
        hysteresis state.
      - Pinch source rescaling: thumb->tip source vectors are smoothly
        collapsed to zero between `epsilon_2` and `epsilon_1`, so the
        optimizer pulls tips into contact instead of chasing a residual gap.

    A separate "orientation" role matches per-finger DP->tip vectors (short
    stubs along the distal phalanx) to pin down fingertip orientation,
    which the position-only wrist/pinch/finger losses leave underconstrained.

    Joint-position regularization (toward a per-joint reference pose) and
    joint-velocity regularization (between consecutive frames) are added to
    the loss to keep the solution smooth.

    References:
      - Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous
        Manipulation: Yu et al., 2025 (https://arxiv.org/abs/2506.09384)
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

        # AKO algorithm parameters.
        self._epsilon_1 = float(retargeter_params["epsilon_1"])
        self._epsilon_2 = float(retargeter_params["epsilon_2"])
        self._huber_delta = float(retargeter_params["huber_delta"])
        self._weight_joint_vel = float(retargeter_params.get("weight_joint_vel", 0.0))

        # SLSQP solver parameters.
        self.tolerance = float(solver_params["tolerance"])
        self.iterations = int(solver_params["iterations"])

        device_str = str(solver_params.get("device", "cpu")).lower()
        self._device = resolve_jax_device(device_str, logger=self._logger)
        self._logger.info(
            f"AKORetargeter using JAX device: {self._device} "
            f"(requested: {device_str!r})"
        )

        # When on GPU, rebuild the hand model's MJX FK on that device so the
        # JIT'd loss doesn't close over CPU-pinned MJX arrays. No-op on CPU.
        rebuild_mjx_fk_on_device(self.to_model, self._device, logger=self._logger)

        self.precomputed_scale = precomputed_scale
        self.alignment_landmarks = alignment_landmarks

        self.keyvectors_cfg = self._normalize_keyvectors_cfg(keyvectors_cfg)
        self._keyvector_names: list[str] = [kv["name"] for kv in self.keyvectors_cfg]
        self._src_keys: list[str] = [kv["src_key"] for kv in self.keyvectors_cfg]
        self._tgt_keys: list[str] = [kv["tgt_key"] for kv in self.keyvectors_cfg]
        self._n_vectors = len(self.keyvectors_cfg)

        # Per-entry static arrays.
        self._loss_weights = np.asarray(
            [kv["loss_coef"] for kv in self.keyvectors_cfg], dtype=np.float32
        )
        self._is_wrist = np.array(
            [kv["role"] == "wrist" for kv in self.keyvectors_cfg], dtype=bool
        )
        self._is_pinch = np.array(
            [kv["role"] == "pinch" for kv in self.keyvectors_cfg], dtype=bool
        )

        self._num_dofs = int(self.to_model.num_actuated_dofs)

        # Per-actuated-joint reference and weight for joint-position regularization.
        # Joints not listed get weight 0 (effectively disabled).
        self._reg_ref, self._reg_weights = self._init_joint_pos_regularization(
            regularized_joints or {}
        )

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
                "_reg_ref",
                "_reg_weights",
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

        Role decides switching-weight and rescaling behavior:
          - "wrist":       wrist->tip. Weight is gated OFF as the source hand closes.
          - "pinch":       thumb->tip. Weight is gated ON as the source hand closes;
                           the source vector is additionally rescaled between
                           epsilon_2 and epsilon_1 so the target collapses to 0.
          - "orientation": per-finger DP->tip. Plain smooth-L1 on a short stub;
                           constrains fingertip orientation.
          - "finger":      everything else (finger tip<->tip pairs). Plain smooth-L1.
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
                    role = "pinch"
                elif "_DP_to_" in tgt_key and tgt_key.endswith("_TIP"):
                    role = "orientation"
                else:
                    role = "finger"
            entry["role"] = role

            entry["loss_coef"] = float(entry.get("loss_coef", 1.0))
            out.append(entry)
        return out

    def _init_joint_pos_regularization(
        self, regularized_joints: dict[str, dict[str, float]]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Build (num_dofs,) reference pose and weight arrays from the YAML dict.

        Expected shape: ``{joint_name: {target: float, weight: float}}``. Joints
        that aren't actuated are warned about and skipped; joints not listed get
        weight 0 so they don't contribute to the regularizer.
        """
        ref = np.zeros(self._num_dofs, dtype=np.float32)
        weights = np.zeros(self._num_dofs, dtype=np.float32)
        for name, spec in regularized_joints.items():
            if name not in self._actuated_names:
                self._logger.warning(
                    f"Regularized joint '{name}' not in actuated joints; skipping."
                )
                continue
            idx = self._actuated_names.index(name)
            ref[idx] = float(spec.get("target", 0.0))
            weights[idx] = float(spec.get("weight", 0.0))
        return (
            jnp.asarray(ref, dtype=jnp.float32),
            jnp.asarray(weights, dtype=jnp.float32),
        )

    def _init_loss_fn(self) -> Any:
        """Compile the AKO weighted smooth-L1 loss + joint regularizers."""
        keyvector_names = tuple(self._tgt_keys)
        delta = self._huber_delta
        vel_w = self._weight_joint_vel
        reg_ref = self._reg_ref
        reg_weights = self._reg_weights

        def smooth_l1(x: jnp.ndarray) -> jnp.ndarray:
            abs_x = jnp.abs(x)
            return jnp.where(abs_x < delta, 0.5 * x * x / delta, abs_x - 0.5 * delta)

        def loss_fn(
            qpos: jnp.ndarray,
            targets: jnp.ndarray,
            per_vec_weights: jnp.ndarray,
            qpos_prev: jnp.ndarray,
        ) -> jnp.ndarray:
            qpos_batched = qpos[None, :]
            tgt_kv = self.to_model.compute_keyvectors_jax(
                qpos_batched, joint_space="ctrl"
            )
            tgt_vecs = jnp.stack([tgt_kv[name][0] for name in keyvector_names], axis=0)

            # Magnitude error per keyvector; safe sqrt to keep gradients finite at 0.
            diffs = tgt_vecs - targets
            errors = jnp.sqrt(jnp.sum(diffs * diffs, axis=-1) + 1e-12)
            vec_loss = jnp.sum(smooth_l1(per_vec_weights * errors))

            reg_loss = jnp.sum(smooth_l1(reg_weights * (qpos - reg_ref)))
            vel_loss = jnp.sum(smooth_l1(vel_w * (qpos - qpos_prev)))

            return vec_loss + reg_loss + vel_loss

        return jax.jit(loss_fn)

    def _optimize_controls(
        self,
        qpos_init_guess: jnp.ndarray,
        targets: jnp.ndarray,
        per_vec_weights: jnp.ndarray,
        qpos_prev: jnp.ndarray,
    ) -> jnp.ndarray:
        with jax.default_device(self._device):
            # Per-frame inputs from the numpy/CPU path; push to the target
            # device so the JIT'd loss doesn't host->device copy each call.
            targets = jax.device_put(targets, self._device)
            per_vec_weights = jax.device_put(per_vec_weights, self._device)
            qpos_prev_d = jax.device_put(qpos_prev, self._device)
            qpos_clipped = jnp.clip(
                qpos_init_guess, self._lower_bounds, self._upper_bounds
            )
            result = self._solver.run(
                qpos_clipped,
                bounds=self._bounds_tuple,
                targets=targets,
                per_vec_weights=per_vec_weights,
                qpos_prev=qpos_prev_d,
            )
            qpos_optimized = jnp.clip(
                result.params, self._lower_bounds, self._upper_bounds
            )
        self._logger.debug(f"Loss: {float(result.state.fun_val)}")
        return qpos_optimized

    def _build_targets_and_weights(
        self, src_keyvectors: dict[str, np.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Per-vector targets and combined (loss_coef * switching_weight) weights.

        Pinch switching weight: s(d) = 1 / (1 + exp(10 * (d - eps1)))  (high when d << eps1)
        Wrist switching weight: 1 / (1 + exp(-10 * (d - eps1)))        (low  when d << eps1)
          - d here is the min source-pinch distance, per the reference implementation,
            so the wrist anchor releases as soon as any pinch is imminent.
        Pinch source rescaling (per-entry):
          d_new = 0                                 if d < eps2
          d_new = eps1/(eps1-eps2) * (d - eps2)     if eps2 <= d <= eps1
          d_new = d                                 if d > eps1
        """
        src_vecs = np.stack(
            [
                np.asarray(src_keyvectors[key], dtype=np.float32)
                for key in self._src_keys
            ],
            axis=0,
        )
        dists = np.linalg.norm(src_vecs, axis=-1)
        dirs = src_vecs / (dists[..., None] + 1e-8)

        sw = np.ones(self._n_vectors, dtype=np.float32)
        if self._is_pinch.any():
            sw[self._is_pinch] = 1.0 / (
                1.0 + np.exp(10.0 * (dists[self._is_pinch] - self._epsilon_1))
            )
        if self._is_wrist.any():
            d_wrist = (
                float(np.min(dists[self._is_pinch]))
                if self._is_pinch.any()
                else self._epsilon_1
            )
            sw[self._is_wrist] = 1.0 / (
                1.0 + np.exp(-10.0 * (d_wrist - self._epsilon_1))
            )

        per_vec_weights = self._loss_weights * sw

        targets_np = src_vecs.copy()
        if self._is_pinch.any():
            k = self._epsilon_1 / (self._epsilon_1 - self._epsilon_2)
            for i in np.where(self._is_pinch)[0]:
                d = float(dists[i])
                if d < self._epsilon_2:
                    d_new = 0.0
                elif d > self._epsilon_1:
                    d_new = d
                else:
                    d_new = k * (d - self._epsilon_2)
                targets_np[i] = dirs[i] * d_new

        return (
            jnp.asarray(targets_np, dtype=jnp.float32),
            jnp.asarray(per_vec_weights, dtype=jnp.float32),
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

        return {
            "src_raw_points": pcloud.copy(),
            "src_keyvectors": src_segments,
            "tgt_keyvectors": tgt_segments,
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
        targets, per_vec_weights = self._build_targets_and_weights(src_keyvectors)

        qpos_optimized = self._optimize_controls(
            qpos_init_guess=self._qpos_prev,
            targets=targets,
            per_vec_weights=per_vec_weights,
            qpos_prev=self._qpos_prev,
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

        # Scene/dataset pipeline still expects a leading batch dim; match existing retargeters.
        return qpos_optimized[None, :], tgt_wrist_transform
