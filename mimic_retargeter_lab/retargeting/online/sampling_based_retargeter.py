"""Sampling-based (iCEM / softmax-CE) retargeter with DexPilot target switching.

Mirrors :class:`KeyvectorRetargeter` and :class:`DexPilotRetargeter` end-to-end
for the per-frame pipeline (Kabsch alignment -> DexPilot target switching ->
debug viz), but replaces the SLSQP inner loop with a JAX-jitted iCEM step
sampling several thousand qpos candidates per call.

Loss = weighted smooth-L1 keyvector error + ``gamma*||q - q_neutral||^2``
       + ``weight_loss_velocity*||q - q_prev||^2``.

DexPilot target / weight selection follows the group-based scheme (parameters
``weight_default``, ``weight_vector_set1``, ``weight_vector_set2``,
``eta1``, ``eta2``, ``scale_factor``) — not :class:`DexPilotRetargeter`'s
per-entry ``eta``/``active_loss_coef`` schema.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mimic_retargeter_lab.hand_models import BaseHandModel, RobotHandModel
from mimic_retargeter_lab.types import HandLandmark
from mimic_retargeter_lab.utils import (
    align_pcloud_kabsch_umeyama,
    device_put_attrs,
    get_logger,
    rebuild_mjx_fk_on_device,
    resolve_jax_device,
)

from .base_online_retargeter import BaseOnlineRetargeter


def _smooth_l1_jax(x: jnp.ndarray, beta: float = 0.02) -> jnp.ndarray:
    abs_x = jnp.abs(x)
    return jnp.where(abs_x < beta, 0.5 * (x**2) / beta, abs_x - 0.5 * beta)


class SamplingBasedRetargeter(BaseOnlineRetargeter):
    """Sampling-based retargeter (iCEM with DexPilot targets, JAX/MJX backend)."""

    def __init__(
        self,
        from_model: BaseHandModel,
        to_model: RobotHandModel,
        wrist_mapping: dict[str, str | np.ndarray],
        alignment_landmarks: list[HandLandmark],
        keyvectors_cfg: list[dict[str, Any]],
        dexpilot_params: dict[str, float],
        sampling_params: dict[str, Any],
        precomputed_scale: float | None = None,
        debug_mode: bool = False,
        debug_every_n_frames: int = 1,
    ):
        super().__init__(from_model, to_model, wrist_mapping)
        self._logger = get_logger(__name__)

        self.precomputed_scale = precomputed_scale
        self.alignment_landmarks = alignment_landmarks

        self._project_distance = float(dexpilot_params["project_distance"])
        self._escape_distance = float(dexpilot_params["escape_distance"])
        self._scale_factor = float(dexpilot_params["scale_factor"])
        self._gamma = float(dexpilot_params["gamma"])
        self._weight_default = float(dexpilot_params["weight_default"])
        self._weight_vector_set1 = float(dexpilot_params["weight_vector_set1"])
        self._weight_vector_set2 = float(dexpilot_params["weight_vector_set2"])
        self._eta1 = float(dexpilot_params["eta1"])
        self._eta2 = float(dexpilot_params["eta2"])
        self._weight_loss_velocity = float(dexpilot_params["weight_loss_velocity"])
        self._loss_smooth_l1_beta = float(dexpilot_params.get("beta", 0.02))

        self._num_samples = int(sampling_params["num_samples"])
        self._num_samples_elite = int(sampling_params["num_samples_elite"])
        self._sigma = sampling_params["sigma"]
        self._lambda = float(sampling_params["lambda"])
        self._learning_rate = float(sampling_params["learning_rate"])
        self._update_cycle = int(sampling_params["update_cycle"])
        self._jax_compilation_cache_enable = bool(
            sampling_params.get("jax_compilation_cache_enable", True)
        )
        self._jax_compilation_cache_dir = str(
            sampling_params.get(
                "jax_compilation_cache_dir",
                "~/.cache/mimic_retargeter_lab/jax_compilation",
            )
        )
        rng_seed = int(sampling_params.get("rng_seed", 42))

        device_str = str(sampling_params.get("device", "cuda")).lower()
        self._device = resolve_jax_device(device_str, logger=self._logger)
        self._logger.info(
            f"SamplingBasedRetargeter using JAX device: {self._device} "
            f"(requested: {device_str!r})"
        )

        # When on GPU, rebuild the hand model's MJX FK on that device so the
        # JIT'd loss doesn't close over CPU-pinned MJX arrays. No-op on CPU.
        rebuild_mjx_fk_on_device(self.to_model, self._device, logger=self._logger)

        self.keyvectors_cfg = self._normalize_keyvectors_cfg(keyvectors_cfg)
        self._tgt_keys: list[str] = [kv["tgt_key"] for kv in self.keyvectors_cfg]
        self._src_keys: list[str] = [kv["src_key"] for kv in self.keyvectors_cfg]
        self._num_keyvectors = len(self.keyvectors_cfg)

        # Hysteresis state for thumb->fingertip and finger<->finger entries only.
        # Wrist->fingertip entries are always inactive (no grasp engagement).
        self._dexpilot_grasp_states: dict[str, bool] = {
            kv["name"]: False for kv in self.keyvectors_cfg if kv["group"] != "wrist"
        }

        self._num_dofs = int(self.to_model.num_actuated_dofs)
        self._lower_bounds, self._upper_bounds = self._init_bounds()

        self._q_neutral = jnp.asarray(
            self.to_model.get_neutral_ctrl_pose(), dtype=jnp.float32
        )

        self._configure_jax_compilation_cache()

        # Pin every persistent JAX array we hand to the JIT'd step on the
        # target device — otherwise per-frame calls trigger host->device
        # copies that dominate wall time. ``self._qpos_prev`` is initialized
        # by the base class on the JAX default; ``device_put_attrs`` re-places
        # it. The RNG key is created inline since ``device_put_attrs`` only
        # moves existing attributes.
        self._jax_rng_key = jax.random.PRNGKey(rng_seed)
        device_put_attrs(
            self,
            (
                "_lower_bounds",
                "_upper_bounds",
                "_q_neutral",
                "_qpos_prev",
                "_jax_rng_key",
            ),
            self._device,
        )

        with jax.default_device(self._device):
            self._batched_loss_fn = self._build_batched_loss_fn()
            self._monte_carlo_step_fn = self._build_monte_carlo_step_fn()
            self._jax_step_compilation_reported = False
            self._warmup_jax_compilation()

        self.debug_mode = bool(debug_mode)
        self.debug_every_n_frames = max(1, int(debug_every_n_frames))
        self._retarget_step_idx = 0
        self._debug_visualization_data: dict[str, np.ndarray] | None = None

    def _normalize_keyvectors_cfg(
        self, keyvectors_cfg: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Add the ``group`` field to each entry by inspecting ``tgt_key`` prefix."""
        out: list[dict[str, Any]] = []
        for kv in keyvectors_cfg:
            entry = dict(kv)
            tgt_key = entry["tgt_key"]
            if tgt_key.startswith("WRIST_to_"):
                group = "wrist"
            elif tgt_key.startswith("THUMB_TIP_to_"):
                group = "thumb"
            else:
                group = "finger"
            entry["group"] = group
            out.append(entry)
        return out

    def _configure_jax_compilation_cache(self) -> None:
        """Enable persistent JAX compilation cache when requested."""
        if not self._jax_compilation_cache_enable:
            self._logger.info("JAX compilation cache disabled by configuration.")
            return
        try:
            cache_dir_path = Path(self._jax_compilation_cache_dir).expanduser()
            cache_dir_path.mkdir(parents=True, exist_ok=True)
            self._logger.info(f"Configuring JAX compilation cache at: {cache_dir_path}")
            from jax.experimental.compilation_cache import (
                compilation_cache as jax_compilation_cache,
            )

            jax_compilation_cache.set_cache_dir(str(cache_dir_path))
            self._logger.info("JAX compilation cache enabled.")
        except Exception as exc:
            self._logger.warning(
                f"Failed to enable JAX compilation cache; continuing without "
                f"persistent cache. Error: {exc}"
            )

    def _thumb_name_for_tgt(self, thumb_tgt_key: str) -> str:
        """Resolve cfg ``name`` whose ``tgt_key`` matches ``thumb_tgt_key``."""
        for kv in self.keyvectors_cfg:
            if kv["group"] == "thumb" and kv["tgt_key"] == thumb_tgt_key:
                return kv["name"]
        return thumb_tgt_key

    def _update_dexpilot_grasp_states(self, src_distances: np.ndarray) -> None:
        """Hysteresis update: thumb->tip grasps update independently;
        finger<->finger derives from the AND of the two adjacent thumb grasps."""
        for i, kv in enumerate(self.keyvectors_cfg):
            if kv["group"] != "thumb":
                continue
            name = kv["name"]
            if name not in self._dexpilot_grasp_states:
                continue
            dist = float(src_distances[i])
            is_active = self._dexpilot_grasp_states[name]
            if not is_active and dist < self._project_distance:
                self._dexpilot_grasp_states[name] = True
                self._logger.debug(f"Engaging grasp: {name}")
            elif is_active and dist > self._escape_distance:
                self._dexpilot_grasp_states[name] = False
                self._logger.debug(f"Releasing grasp: {name}")

        for kv in self.keyvectors_cfg:
            if kv["group"] != "finger":
                continue
            name = kv["name"]
            if name not in self._dexpilot_grasp_states:
                continue
            a, b = kv["tgt_key"].split("_to_")
            thumb_a = self._thumb_name_for_tgt(f"THUMB_TIP_to_{a}")
            thumb_b = self._thumb_name_for_tgt(f"THUMB_TIP_to_{b}")
            self._dexpilot_grasp_states[name] = self._dexpilot_grasp_states.get(
                thumb_a, False
            ) and self._dexpilot_grasp_states.get(thumb_b, False)

    def _compute_dexpilot_loss_terms(
        self, src_keyvectors: dict[str, np.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Per-vector ``(targets, weights)`` for the JAX step (group-based scheme)."""
        src_vecs = np.stack(
            [
                np.asarray(src_keyvectors[k], dtype=np.float32).reshape(3)
                for k in self._src_keys
            ],
            axis=0,
        )
        src_dists = np.linalg.norm(src_vecs, axis=-1)
        src_dirs = src_vecs / (src_dists[..., None] + 1e-8)

        self._update_dexpilot_grasp_states(src_dists)

        targets = np.zeros((self._num_keyvectors, 3), dtype=np.float32)
        weights = np.zeros((self._num_keyvectors,), dtype=np.float32)

        for i, kv in enumerate(self.keyvectors_cfg):
            name = kv["name"]
            group = kv["group"]
            is_active = self._dexpilot_grasp_states.get(name, False)
            if is_active:
                if group == "thumb":
                    targets[i] = src_dirs[i] * self._eta1
                    weights[i] = self._weight_vector_set1
                elif group == "finger":
                    targets[i] = src_dirs[i] * self._eta2
                    weights[i] = self._weight_vector_set2
                else:
                    targets[i] = src_vecs[i] * self._scale_factor
                    weights[i] = self._weight_default
            else:
                targets[i] = src_vecs[i] * self._scale_factor
                weights[i] = self._weight_default

        return (
            jnp.asarray(targets, dtype=jnp.float32),
            jnp.asarray(weights, dtype=jnp.float32),
        )

    def _build_batched_loss_fn(self) -> Any:
        """Compile the per-sample weighted smooth-L1 + regularizers loss."""
        tgt_keys = tuple(self._tgt_keys)
        gamma = jnp.float32(self._gamma)
        weight_loss_velocity = jnp.float32(self._weight_loss_velocity)
        beta = jnp.float32(self._loss_smooth_l1_beta)
        q_neutral = self._q_neutral

        def loss_fn(
            qpos_batch: jnp.ndarray,  # (S, dof) ctrl space
            targets: jnp.ndarray,  # (K, 3)
            weights: jnp.ndarray,  # (K,)
            qpos_current: jnp.ndarray,  # (dof,)
        ) -> jnp.ndarray:
            tgt_kv_dict = self.to_model.compute_keyvectors_jax(
                qpos_batch, joint_space="ctrl"
            )
            tgt_vecs = jnp.stack(
                [tgt_kv_dict[k] for k in tgt_keys], axis=1
            )  # (S, K, 3)
            err = tgt_vecs - targets[None, :, :]
            kv_losses = jnp.sum(_smooth_l1_jax(err, beta=beta), axis=-1)  # (S, K)
            kinematic_loss = jnp.sum(kv_losses * weights[None, :], axis=-1)  # (S,)
            reg_loss = gamma * jnp.sum((qpos_batch - q_neutral) ** 2, axis=-1)
            velocity_loss = weight_loss_velocity * jnp.sum(
                (qpos_batch - qpos_current) ** 2, axis=-1
            )
            return kinematic_loss + reg_loss + velocity_loss

        return jax.jit(loss_fn)

    def _build_monte_carlo_step_fn(self) -> Any:
        """Compile the iCEM / softmax-CE update loop."""
        num_samples = self._num_samples
        num_samples_elite = self._num_samples_elite
        inv_lambda = jnp.float32(1.0 / self._lambda)
        learning_rate = jnp.float32(self._learning_rate)
        update_cycle = self._update_cycle
        eps = jnp.float32(1e-6)
        lower_bounds = self._lower_bounds
        upper_bounds = self._upper_bounds
        batched_loss_fn = self._batched_loss_fn

        sigma_np = np.asarray(self._sigma, dtype=np.float32)

        def step_fn(
            rng_key: jax.Array,
            qpos_current: jnp.ndarray,
            targets: jnp.ndarray,
            weights: jnp.ndarray,
        ):
            dof = qpos_current.shape[0]
            eye = jnp.eye(dof, dtype=jnp.float32)

            if sigma_np.ndim == 0:
                sigma_diag = jnp.full((dof,), jnp.float32(sigma_np), dtype=jnp.float32)
            else:
                sigma_diag = jnp.asarray(sigma_np, dtype=jnp.float32)
            covariance = jnp.diag(sigma_diag**2) + eps * eye

            mean_update = qpos_current
            losses_last = jnp.full((num_samples,), jnp.inf, dtype=jnp.float32)

            split_keys = jax.random.split(rng_key, update_cycle + 1)
            rng_next = split_keys[0]
            loop_subkeys = split_keys[1:]

            for i in range(update_cycle):
                noises = jax.random.multivariate_normal(
                    key=loop_subkeys[i],
                    mean=jnp.zeros((dof,), dtype=jnp.float32),
                    cov=covariance,
                    shape=(num_samples,),
                    method="cholesky",
                )
                q_samples = mean_update[None, :] + noises
                q_samples = jnp.clip(q_samples, lower_bounds, upper_bounds)

                losses = batched_loss_fn(q_samples, targets, weights, qpos_current)
                losses_last = losses

                if i < update_cycle - 1:
                    elite_idxs = jnp.argsort(losses)[:num_samples_elite]
                    q_elite = q_samples[elite_idxs, :]
                    elite_losses = losses[elite_idxs]

                    min_elite = elite_losses[0]
                    exp_weights = jnp.exp(-inv_lambda * (elite_losses - min_elite))
                    weights_norm = exp_weights / (jnp.sum(exp_weights) + eps)

                    weighted_mean = jnp.einsum("k,ki->i", weights_norm, q_elite)
                    mean_update = (
                        1.0 - learning_rate
                    ) * mean_update + learning_rate * weighted_mean

                    centered = q_elite - mean_update[None, :]
                    weighted_outer = jnp.einsum(
                        "k,ki,kj->ij", weights_norm, centered, centered
                    )
                    covariance = (
                        1.0 - learning_rate
                    ) * covariance + learning_rate * weighted_outer
                    covariance = covariance + eps * eye

            qpos_next = jnp.clip(mean_update, lower_bounds, upper_bounds)
            return rng_next, qpos_next, losses_last

        return jax.jit(step_fn)

    def _warmup_jax_compilation(self) -> None:
        """Run one fake step to force JIT compilation before the first real frame.

        Uses the exact same input objects the real ``retarget()`` call will
        use (``self._qpos_prev`` and ``self._jax_rng_key``, both already
        device-pinned in ``__init__``) plus targets/weights pushed through
        ``jax.device_put`` — matching the real call's commitments byte-for-byte
        so frame 1 is a cache hit instead of a 50s recompile.
        """
        self._logger.info(
            f"JAX warmup: compiling sampling kernels "
            f"(dof={self._num_dofs}, samples={self._num_samples}, "
            f"update_cycle={self._update_cycle})..."
        )
        warmup_start = time.perf_counter()

        targets_warm = jax.device_put(
            jnp.zeros((self._num_keyvectors, 3), dtype=jnp.float32),
            self._device,
        )
        weights_warm = jax.device_put(
            jnp.ones((self._num_keyvectors,), dtype=jnp.float32),
            self._device,
        )

        # Discard the returned ``rng_next``: frame 1 deliberately reuses
        # the seeded ``self._jax_rng_key`` so warmup and the first real
        # call produce byte-identical inputs (same shape, dtype, device,
        # *and* value), guaranteeing the JIT cache hit. Don't "fix" this
        # to advance the key — you'll trade determinism for nothing.
        _, qpos_next, losses = self._monte_carlo_step_fn(
            self._jax_rng_key, self._qpos_prev, targets_warm, weights_warm
        )
        qpos_next.block_until_ready()
        losses.block_until_ready()

        warmup_elapsed = time.perf_counter() - warmup_start
        self._logger.info(
            f"JAX warmup complete in {warmup_elapsed:.2f}s (device: {self._device})."
        )

    def _keyvector_segments_from_config(
        self, landmarks: dict[HandLandmark, np.ndarray]
    ) -> np.ndarray:
        """Translate keyvector cfg entries into ``(N, 2, 3)`` segments for viz."""
        segments: list[np.ndarray] = []
        for kv in self.keyvectors_cfg:
            tgt_key = kv["tgt_key"]
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

    def _generate_debug_visualization_data(
        self,
        pcloud_raw: np.ndarray,
        pcloud_aligned: np.ndarray,
        qpos: np.ndarray,
        tgt_landmarks_alignment: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Match the shape of :class:`KeyvectorRetargeter`'s debug payload."""
        src_landmarks_aligned = self.from_model.get_landmarks(pcloud_aligned)
        tgt_landmarks = self.to_model.get_landmarks(qpos)

        src_segments = self._keyvector_segments_from_config(src_landmarks_aligned)
        tgt_segments = self._keyvector_segments_from_config(tgt_landmarks)

        grasp_flags = np.array(
            [
                self._dexpilot_grasp_states.get(kv["name"], False)
                for kv in self.keyvectors_cfg
            ],
            dtype=bool,
        )

        return {
            "tgt_landmarks": tgt_landmarks_alignment,
            "src_raw_points": pcloud_raw.copy(),
            "src_aligned_points": pcloud_aligned.copy(),
            "src_keyvectors": src_segments,
            "tgt_keyvectors": tgt_segments,
            "grasp_active": grasp_flags,
        }

    @property
    def debug_visualization_data(self) -> dict[str, np.ndarray] | None:
        return self._debug_visualization_data

    def retarget(
        self,
        pcloud: np.ndarray,
        wrist_transform: Any | None = None,
    ) -> tuple[Any, Any | None]:
        """Retarget the hand from source point cloud to the target."""
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
        targets, weights = self._compute_dexpilot_loss_terms(src_keyvectors)

        # Per-frame inputs from the dexpilot path land on the JAX default
        # device (CPU). Push them to ``self._device`` explicitly so the JIT'd
        # step doesn't trigger a host->device copy on every call. Persistent
        # state (``_qpos_prev``, ``_jax_rng_key``, bounds) is already on
        # ``self._device`` from ``__init__``.
        targets = jax.device_put(targets, self._device)
        weights = jax.device_put(weights, self._device)

        with jax.default_device(self._device):
            self._jax_rng_key, qpos_next_j, losses_j = self._monte_carlo_step_fn(
                self._jax_rng_key, self._qpos_prev, targets, weights
            )
            qpos_optimized = jnp.clip(
                qpos_next_j, self._lower_bounds, self._upper_bounds
            )

        if not self._jax_step_compilation_reported:
            losses_j.block_until_ready()
            self._logger.info("SamplingBasedRetargeter JAX compilation done.")
            self._jax_step_compilation_reported = True

        self._qpos_prev = qpos_optimized

        tgt_wrist_transform = None
        if wrist_transform is not None:
            tgt_wrist_transform = self.wrist_retargeter.retarget(wrist_transform)

        if self.debug_mode and (
            self._retarget_step_idx % self.debug_every_n_frames == 0
        ):
            self._debug_visualization_data = self._generate_debug_visualization_data(
                pcloud_raw=pcloud,
                pcloud_aligned=pcloud_aligned,
                qpos=np.asarray(qpos_optimized, dtype=np.float32),
                tgt_landmarks_alignment=tgt_landmarks,
            )
        self._retarget_step_idx += 1

        # Force the GPU work to finish before we log timing — without this,
        # the JIT call returns immediately (async dispatch) and the time below
        # would only measure dispatch overhead, hiding any real compute cost.
        qpos_optimized.block_until_ready()
        end_time = time.time()
        if self._retarget_step_idx <= 5 or self._retarget_step_idx % 50 == 0:
            self._logger.debug(
                f"Retargeting frame {self._retarget_step_idx} "
                f"({(end_time - start_time) * 1000:.1f} ms)"
            )

        return qpos_optimized[None, :], tgt_wrist_transform
