"""Retargeting utilities."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

# Third-party
import numpy as np
import jax.numpy as jnp
from scipy.spatial.transform import Rotation
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Number of untimed JIT-warmup retargets run before the timed pass during
# RetargetCache populate. Picked to cover JAX's first-call compilation; if
# a retargeter retraces past this many frames, bump the constant.
_DEFAULT_WARMUP_FRAMES = 10


def _human_joints_to_jax(points_3d: Any) -> jnp.ndarray:
    """Convert human joint sequence to float32 JAX array ``(T, J, 3)``.

    Accepts NumPy, JAX arrays, or torch tensors (no ``torch`` import; uses ``.detach()``).
    """
    detach = getattr(points_3d, "detach", None)
    if callable(detach):
        arr = detach().cpu().numpy()
    else:
        arr = np.asarray(points_3d)
    return jnp.asarray(arr, dtype=jnp.float32)


def compute_kabsch_umeyama_transform(
    source_points: np.ndarray, target_points: np.ndarray
) -> tuple[np.ndarray, float]:
    """Calculate the optimal Kabsch rotation matrix and true Umeyama scale.

    Args:
        source_points (np.ndarray): An (N, 3) array of centered points.
        target_points (np.ndarray): An (N, 3) array of corresponding centered points.

    Returns:
        tuple containing:
            - np.ndarray: The (3, 3) rotation matrix.
            - float: The optimal Umeyama scale factor.
    """
    # Calculate covariance matrix
    H = source_points.T @ target_points

    try:
        # Use SVD to find optimal rotation
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Auto handle reflection case
        d = 1.0
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
            d = -1.0  # Track reflection for the scale calculation

        # Calculate strict Umeyama scale
        # scale = trace(S * D) / variance(X)
        var_src = np.sum(source_points**2)
        if var_src < 1e-8:  # Safe fallback to avoid division by zero
            scale = 1.0
        else:
            # S is sorted descending, so S[2] is the smallest singular value
            scale = (S[0] + S[1] + d * S[2]) / var_src

        return R, scale

    except np.linalg.LinAlgError:
        print("Warning: SVD failed to converge. Returning identity transform.")
        return np.eye(3), 1.0


def align_pcloud_kabsch_umeyama(
    points: np.ndarray,
    source_landmarks: np.ndarray,
    target_landmarks: np.ndarray,
    precomputed_scale: float | None = None,
) -> tuple[np.ndarray, Rotation, float]:
    """
    Aligns points by estimating the similarity transformation (Scale, Rotation, Translation)
    between source and target landmarks.

    Args:
        points: (N, 3) Array of points to transform.
        source_landmarks: (M, 3) Corresponding points in the source frame.
        target_landmarks: (M, 3) Corresponding points in the target frame.
        precomputed_scale: Optional float. If provided, this scale is applied.
                           If None, the optimal Umeyama scale is estimated from landmarks.

    Returns:
        tuple containing:
            - Aligned points (N, 3)
            - Rotation object
            - Scale factor used
    """
    points = points.copy()

    source_centroid = np.mean(source_landmarks, axis=0)
    target_centroid = np.mean(target_landmarks, axis=0)

    # Center clouds
    src_centered = source_landmarks - source_centroid
    tgt_centered = target_landmarks - target_centroid

    # Compute optimal rotation and scale together
    R_matrix, optimal_scale = compute_kabsch_umeyama_transform(
        src_centered, tgt_centered
    )
    R_kabsch = Rotation.from_matrix(R_matrix)

    # Use precomputed scale if provided, otherwise use the calculated Umeyama scale
    scale_factor = precomputed_scale if precomputed_scale is not None else optimal_scale

    # Apply transformation to the full point cloud
    points_centered = points - source_centroid
    points_scaled = points_centered * scale_factor
    points_rotated = points_scaled @ R_matrix.T
    aligned_points = points_rotated + target_centroid

    return aligned_points, R_kabsch, scale_factor


def retarget_points_sequence(retargeter, points_3d: Any) -> jnp.ndarray:
    """
    Run the retargeter once per time step so the batch dimension is always a single frame.

    Args:
        retargeter: Online retargeter (e.g. KeyvectorRetargeter).
        points_3d: Human joints ``(T, J, 3)`` (NumPy, JAX, or torch tensor).

    Returns:
        Robot actuated joint commands ``(T, num_actuated_dofs)`` as a JAX array.
    """
    pts = _human_joints_to_jax(points_3d)
    if int(pts.ndim) != 3:
        raise ValueError(
            f"Expected human joints with shape (T, J, 3); got {tuple(pts.shape)}"
        )
    t_max = int(pts.shape[0])
    frames: list[jnp.ndarray] = []
    for t in tqdm(range(t_max), desc="Retarget", leave=False):
        frame = pts[t : t + 1]
        frame = np.asarray(frame.squeeze(0), dtype=np.float32)
        q_cmd, _ = retargeter.retarget(frame)
        frames.append(jnp.asarray(q_cmd).squeeze(0))
    return jnp.stack(frames, axis=0)


def _retarget_with_timings(
    retargeter,
    points_3d: Any,
    warmup_frames: int = _DEFAULT_WARMUP_FRAMES,
) -> tuple[jnp.ndarray, list[float], str]:
    """Per-frame retarget that also records per-frame latencies.

    Phase 1 — JIT warmup: run ``warmup_frames`` untimed retargets on the
    first frames of the stream so JAX has compiled its graph before timing
    begins. Outputs are discarded. ``retargeter.reset()`` is called
    afterward so the optimizer's warm-start state matches what episode 1
    of a fresh run would see.

    Phase 2 — timed pass over ALL T frames from index 0. Every frame's
    angles are kept (the cache's primary output) and every frame's latency
    is recorded. ``jax.block_until_ready(q_cmd)`` after each call ensures
    the timing reflects real compute, not JAX dispatch.

    Returns ``(angles[T, num_actuated_dofs], latencies_ms[T], device_str)``.
    """
    pts = _human_joints_to_jax(points_3d)
    if int(pts.ndim) != 3:
        raise ValueError(
            f"Expected human joints with shape (T, J, 3); got {tuple(pts.shape)}"
        )

    device = getattr(retargeter, "_device", None)
    device_str = device.platform if device is not None else "unknown"

    try:
        import jax  # type: ignore

        sync_enabled = device is not None
    except ImportError:
        jax = None  # type: ignore
        sync_enabled = False

    def _sync(q_cmd) -> None:
        if not sync_enabled:
            return
        try:
            jax.block_until_ready(q_cmd)
        except Exception:
            pass

    t_max = int(pts.shape[0])

    # Phase 1: untimed JIT warmup, then reset retargeter state so the
    # timed pass starts from a clean baseline.
    warmup = max(0, min(warmup_frames, t_max))
    for t in range(warmup):
        frame = np.asarray(pts[t : t + 1].squeeze(0), dtype=np.float32)
        q_cmd, _ = retargeter.retarget(frame)
        _sync(q_cmd)
    retargeter.reset()

    # Phase 2: timed pass over all T frames.
    frames: list[jnp.ndarray] = []
    latencies_ms: list[float] = []
    for t in tqdm(range(t_max), desc="Retarget", leave=False):
        frame = np.asarray(pts[t : t + 1].squeeze(0), dtype=np.float32)
        t0 = time.perf_counter_ns()
        q_cmd, _ = retargeter.retarget(frame)
        _sync(q_cmd)
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / 1e6)
        frames.append(jnp.asarray(q_cmd).squeeze(0))
    return jnp.stack(frames, axis=0), latencies_ms, device_str


class RetargetCache:
    """Per-episode retargeting cache.

    Single-retargeter only. ``get(key, joints)`` returns a cached
    ``(T, num_actuated_dofs)`` JAX array of joint angles. On a cache miss
    it runs the retargeter on the supplied joints and stores both the
    angles and per-frame retarget latencies (see ``_retarget_with_timings``);
    ``timings(key)`` exposes the latencies so ``LatencyMetric`` can read
    them without re-iterating the dataset.

    The cache does not reset the retargeter between ``get`` calls —
    callers control reset cadence so that warm-starting between
    consecutive episodes is preserved (important for static reference
    poses where the prior pose is a good initial guess).
    """

    def __init__(self, retargeter):
        self._retargeter = retargeter
        # key -> {"angles": jnp.ndarray, "latencies_ms": list[float],
        #         "device": str, "warmup_frames": int}
        self._cache: dict[tuple, dict] = {}

    def get(self, key: tuple, human_joints_3d: Any) -> jnp.ndarray:
        label = self._format_key(key)
        if key not in self._cache:
            logger.info(f"Retargeting {label} (caching for reuse)")
            angles, latencies_ms, device = _retarget_with_timings(
                self._retargeter, human_joints_3d
            )
            self._cache[key] = {
                "angles": angles,
                "latencies_ms": latencies_ms,
                "device": device,
                "warmup_frames": _DEFAULT_WARMUP_FRAMES,
            }
        else:
            logger.info(f"Using cached retargeting for {label}")
        return self._cache[key]["angles"]

    def timings(self, key: tuple) -> dict:
        """Per-frame retarget latencies + device for a populated key.

        Raises KeyError if the key has not been populated yet — caller
        should call ``get(...)`` first to ensure population.
        """
        if key not in self._cache:
            raise KeyError(f"No cached retargeting for {self._format_key(key)}")
        entry = self._cache[key]
        return {
            "latencies_ms": entry["latencies_ms"],
            "device": entry["device"],
            "warmup_frames": entry["warmup_frames"],
        }

    @staticmethod
    def _format_key(key: tuple) -> str:
        if len(key) >= 2:
            return f"{Path(str(key[0])).stem}/{key[1]}"
        return str(key)
