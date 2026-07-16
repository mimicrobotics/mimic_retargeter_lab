from __future__ import annotations

from typing import Any
import time

import numpy as np
import jax
import jax.numpy as jnp
import jaxopt

from dexworld.hand_models import BaseHandModel, RobotHandModel
from dexworld.objectives import KeyvectorLoss
from dexworld.utils import (
    align_pcloud_kabsch_umeyama,
    device_put_attrs,
    get_logger,
    rebuild_mjx_fk_on_device,
    resolve_jax_device,
)
from dexworld.types import HandLandmark
from .base_online_retargeter import BaseOnlineRetargeter


class KeyvectorRetargeter(BaseOnlineRetargeter):
    def __init__(
        self,
        from_model: BaseHandModel,
        to_model: RobotHandModel,
        keyvectors_cfg: list[dict[str, str | float | bool]],
        wrist_mapping: dict[str, str | np.ndarray],
        alignment_landmarks: list[HandLandmark],
        regularized_joints: dict[str, dict[str, float]] | None = None,
        precomputed_scale: float | None = None,
        iterations: int = 50,
        debug_mode: bool = False,
        debug_every_n_frames: int = 1,
        device: str = "cpu",
    ):
        super().__init__(from_model, to_model, wrist_mapping)
        self._logger = get_logger(__name__)
        self.iterations = iterations
        self.loss = KeyvectorLoss(keyvectors_cfg)
        self.regularized_joints = regularized_joints or {}

        # Kabsch-Umeyama alignment parameters
        self.precomputed_scale = precomputed_scale
        self.alignment_landmarks = alignment_landmarks
        self._num_dofs = int(self.to_model.num_actuated_dofs)

        self._device = resolve_jax_device(device, logger=self._logger)
        self._logger.info(
            f"KeyvectorRetargeter using JAX device: {self._device} "
            f"(requested: {device!r})"
        )

        # When on GPU, rebuild the hand model's MJX FK on that device so the
        # JIT'd loss doesn't close over CPU-pinned MJX arrays. No-op on CPU.
        rebuild_mjx_fk_on_device(self.to_model, self._device, logger=self._logger)

        # Precompile/cache optimization components
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
            ("_lower_bounds", "_upper_bounds", "_qpos_prev"),
            self._device,
        )

        with jax.default_device(self._device):
            self._loss_fn = self._init_loss_fn()
            self._solver = jaxopt.ScipyBoundedMinimize(
                fun=self._loss_fn,
                method="SLSQP",
                tol=1e-6,
                maxiter=self.iterations,
                jit=True,
            )

        # Optional debug payload for interface-side visualization (e.g. MuJoCo overlay).
        self.debug_mode = bool(debug_mode)
        self.debug_every_n_frames = max(1, int(debug_every_n_frames))
        self._retarget_step_idx = 0
        self._debug_visualization_data: dict[str, np.ndarray] | None = None

    def _init_loss_fn(self) -> Any:
        """Initialize and compile the static JAX loss function."""

        def loss_fn(
            qpos: jnp.ndarray,
            src_keyvectors_jax: dict[str, jnp.ndarray],
        ) -> jnp.ndarray:
            # Add batch dimension for JAX FK engine
            qpos_batched = qpos[None, :]
            tgt_keyvectors = self.to_model.compute_keyvectors_jax(
                qpos_batched, joint_space="ctrl"
            )
            return self.loss(src_keyvectors_jax, tgt_keyvectors)

        return jax.jit(loss_fn)

    def _optimize_controls(
        self,
        qpos_init_guess: jnp.ndarray,
        src_keyvectors_jax: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Optimize the controls to minimize the keyvector loss."""
        with jax.default_device(self._device):
            # Per-frame inputs from the numpy/CPU path; push to the target
            # device so the JIT'd loss doesn't host->device copy each call.
            src_keyvectors_jax = {
                k: jax.device_put(v, self._device)
                for k, v in src_keyvectors_jax.items()
            }
            qpos_init_guess_clipped = jnp.clip(
                qpos_init_guess, self._lower_bounds, self._upper_bounds
            )

            result = self._solver.run(
                qpos_init_guess_clipped,
                bounds=self._bounds_tuple,
                src_keyvectors_jax=src_keyvectors_jax,
            )
            qpos_optimized = jnp.clip(
                result.params, self._lower_bounds, self._upper_bounds
            )

        self._logger.debug(f"Loss: {float(result.state.fun_val)}")
        return qpos_optimized

    def _generate_debug_visualization_data(
        self,
        src_landmarks_all: dict[HandLandmark, np.ndarray],
        tgt_landmarks_all: dict[HandLandmark, np.ndarray],
        pcloud_raw: np.ndarray,
        pcloud_aligned: np.ndarray,
        qpos: np.ndarray,
    ) -> dict[str, np.ndarray] | None:
        """Generate debug data for visualization.

        ``src_keyvectors`` / ``tgt_keyvectors`` are emitted as ``(N, 2, 3)``
        arrays of (start, end) 3D pairs — one segment per entry in
        ``keyvectors_cfg``. This lets the viewer render arbitrary pairs
        (wrist→tip, tip↔tip, base→tip, …) rather than only a wrist fan.
        """
        tgt_landmarks_alignment = np.array(
            [tgt_landmarks_all[landmark] for landmark in self.alignment_landmarks]
        )

        # Dynamically fetch src landmarks from the ALIGNED point cloud.
        src_landmarks_aligned = self.from_model.get_landmarks(pcloud_aligned)
        # Dynamically fetch tgt landmarks from the optimized qpos.
        tgt_landmarks = self.to_model.get_landmarks(qpos)

        src_segments = self._keyvector_segments_from_config(src_landmarks_aligned)
        tgt_segments = self._keyvector_segments_from_config(tgt_landmarks)

        debug_visualization_data = {
            "tgt_landmarks": tgt_landmarks_alignment,
            "src_raw_points": pcloud_raw.copy(),
            "src_aligned_points": pcloud_aligned.copy(),
            "src_keyvectors": src_segments,
            "tgt_keyvectors": tgt_segments,
        }

        return debug_visualization_data

    def _keyvector_segments_from_config(
        self, landmarks: dict[HandLandmark, np.ndarray]
    ) -> np.ndarray:
        """Translate the keyvector config into concrete line segments for the viewer.

        The keyvector config names pairs of landmarks by string (e.g. "THUMB_TIP_to_INDEX_TIP");
        the viewer needs concrete 3D coordinates to draw capsules. This method is the
        adapter between the two: for each entry in self.loss.keyvectors_cfg, it splits
        the tgt_key into its two landmark names, looks up their positions in the
        supplied landmarks snapshot, and packs them as a segment.

        Called once per pose (source and target) so the viewer can render both the
        human (red) and robot (blue) versions of the same keyvector set.

        Args:
            landmarks: Map from HandLandmark to world-frame 3D positions for the hand
                in its current pose (either the aligned source hand or the optimized
                target hand).

        Returns:
            (N, 2, 3) array where out[i] is the (start, end) pair for keyvector config
            entry i. Entries whose tgt_key references a landmark that doesn't exist on
            the current hand (e.g. PINKY_TIP on a 3-finger DEX-EE) are silently dropped,
            so N may be smaller than len(keyvectors_cfg). Returns (0, 2, 3) when no
            keyvector entries are renderable, so callers don't need a special path.
        """
        segments: list[np.ndarray] = []
        for kv in self.loss.keyvectors_cfg:
            tgt_key = kv.get("tgt_key")
            if not isinstance(tgt_key, str) or "_to_" not in tgt_key:
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
        """Debug visualization data."""
        return self._debug_visualization_data

    def retarget(
        self,
        pcloud: np.ndarray,
        wrist_transform: Any | None = None,
        qpos_init: jnp.ndarray | None = None,
    ) -> tuple[Any, Any | None]:
        """Retarget the hand from source point cloud (pcloud) to the target.

        ``qpos_init`` overrides ``self._qpos_prev`` as the SLSQP initial guess
        for this frame (used by HybridRetargeter to inject a per-frame
        analytical warmstart). When None (default), warmstart from the
        previous solution as before.
        """
        start_time = time.time()

        # Enforce the (N, 3) pcloud shape
        if pcloud.ndim == 3 and pcloud.shape[0] == 1:
            pcloud = pcloud[
                0
            ]  # Auto-squeeze if the user accidentally passes a batch of 1
        elif pcloud.ndim != 2:
            raise ValueError(
                f"pcloud must have shape (N, 3), got {tuple(pcloud.shape)}"
            )

        # Extract info for the retargeting
        qpos_prev = self._qpos_prev

        # Extract the landmarks for the alignment
        src_landmarks_all = self.from_model.get_landmarks(pcloud)
        tgt_landmarks_all = self.to_model.get_landmarks(
            qpos=np.asarray(qpos_prev, dtype=np.float32)
        )

        # Extract the landmarks for the alignment
        src_landmarks = np.stack(
            [src_landmarks_all[landmark] for landmark in self.alignment_landmarks]
        )
        tgt_landmarks = np.stack(
            [tgt_landmarks_all[landmark] for landmark in self.alignment_landmarks]
        )

        # Align the pcloud
        pcloud_aligned, _, _ = align_pcloud_kabsch_umeyama(
            points=pcloud,
            source_landmarks=src_landmarks,
            target_landmarks=tgt_landmarks,
            precomputed_scale=self.precomputed_scale,
        )

        src_keyvectors = self.from_model.compute_keyvectors(pcloud_aligned)
        src_keyvectors_jax = {
            k: jnp.asarray(v, dtype=jnp.float32) for k, v in src_keyvectors.items()
        }

        # Run the optimization (Clean, no batch math or explicit bounds!)
        init_guess = qpos_init if qpos_init is not None else self._qpos_prev
        qpos_optimized = self._optimize_controls(
            qpos_init_guess=init_guess, src_keyvectors_jax=src_keyvectors_jax
        )
        self._qpos_prev = qpos_optimized

        # Retarget Wrist
        tgt_wrist_transform = None
        if wrist_transform is not None:
            tgt_wrist_transform = self.wrist_retargeter.retarget(wrist_transform)

        # Generate debug visualization data
        if self.debug_mode and (
            self._retarget_step_idx % self.debug_every_n_frames == 0
        ):
            # Note: The debug method needs the pcloud to be flat (N, 3), not nested in a batch
            # If your alignment returns (1, N, 3), ensure you pass pcloud_aligned[0]
            self._debug_visualization_data = self._generate_debug_visualization_data(
                src_landmarks_all=src_landmarks_all,
                tgt_landmarks_all=tgt_landmarks_all,
                pcloud_raw=pcloud,
                pcloud_aligned=pcloud_aligned,
                qpos=qpos_optimized,
            )
        self._retarget_step_idx += 1

        end_time = time.time()
        self._logger.debug(f"Retargeting time: {end_time - start_time:.4f} seconds")

        # TODO: The downstream scene/dataset pipeline (kinematic_retargeting.py)
        # still expects a batch dimension (B, D). We add it back here to prevent
        # the .squeeze(0) crash. Remove this once the scene pipeline is refactored!
        return qpos_optimized[None, :], tgt_wrist_transform
