"""Unit tests for SamplingBasedRetargeter with fakes and a stubbed JAX step."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as SciRotation

from mimic_retargeter_lab.retargeting.online import base_online_retargeter as bor_mod
from mimic_retargeter_lab.retargeting.online import sampling_based_retargeter as sbr_mod
from mimic_retargeter_lab.retargeting.online.sampling_based_retargeter import (
    SamplingBasedRetargeter,
)
from mimic_retargeter_lab.types import HandLandmark


_FINGERTIP_LANDMARKS = [
    HandLandmark.THUMB_TIP,
    HandLandmark.INDEX_TIP,
    HandLandmark.MIDDLE_TIP,
    HandLandmark.RING_TIP,
    HandLandmark.PINKY_TIP,
]


def _landmark_dict() -> dict[HandLandmark, np.ndarray]:
    d: dict[HandLandmark, np.ndarray] = {
        HandLandmark.WRIST: np.array([0.0, 0.0, 0.0], dtype=np.float32),
        HandLandmark.THUMB_BASE: np.array([0.0, 0.0, 0.0], dtype=np.float32),
        HandLandmark.INDEX_BASE: np.array([1.2, 4.7, 9.1], dtype=np.float32),
        HandLandmark.PINKY_BASE: np.array([0.2, 3.1, -5.3], dtype=np.float32),
    }
    for lm in _FINGERTIP_LANDMARKS:
        d[lm] = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    return d


class FakeWristRetargeter:
    def retarget(self, wrist_transform: np.ndarray) -> np.ndarray:
        return wrist_transform + 2.0


class FakeFromModel:
    def get_landmarks(self, pcloud: np.ndarray) -> dict[HandLandmark, np.ndarray]:
        return _landmark_dict()

    def compute_keyvectors(self, data: np.ndarray, **kwargs) -> dict[str, np.ndarray]:
        # One vector per cfg src_key. The test config below uses three keys.
        return {
            "WRIST_to_INDEX_TIP": np.array([0.10, 0.0, 0.0], dtype=np.float32),
            "THUMB_TIP_to_INDEX_TIP": np.array([0.005, 0.0, 0.0], dtype=np.float32),
            "INDEX_TIP_to_MIDDLE_TIP": np.array([0.05, 0.0, 0.0], dtype=np.float32),
        }


class FakeToModel:
    """Two-DoF fake hand."""

    def __init__(self) -> None:
        self.num_actuated_dofs = 2

    def get_neutral_ctrl_pose(self) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)

    def get_landmarks(self, *args, **kwargs) -> dict[HandLandmark, np.ndarray]:
        return _landmark_dict()

    def get_actuated_joint_names(self) -> list[str]:
        return ["a", "b"]

    def get_actuated_joint_limits(self) -> dict[str, tuple[float, float]]:
        return {"a": (-1.0, 1.0), "b": (-1.0, 1.0)}


def _cfg() -> list[dict[str, str]]:
    """One entry per group: wrist, thumb, finger — exercises all branches."""
    return [
        {
            "name": "wrist_to_index_tip",
            "src_key": "WRIST_to_INDEX_TIP",
            "tgt_key": "WRIST_to_INDEX_TIP",
        },
        {
            "name": "thumb_tip_to_index_tip",
            "src_key": "THUMB_TIP_to_INDEX_TIP",
            "tgt_key": "THUMB_TIP_to_INDEX_TIP",
        },
        {
            "name": "index_to_middle",
            "src_key": "INDEX_TIP_to_MIDDLE_TIP",
            "tgt_key": "INDEX_TIP_to_MIDDLE_TIP",
        },
    ]


def _dexpilot_params() -> dict[str, float]:
    return {
        "project_distance": 0.01,
        "escape_distance": 0.015,
        "scale_factor": 1.0,
        "gamma": 0.05,
        "weight_default": 1.0,
        "weight_vector_set1": 2.0,
        "weight_vector_set2": 0.5,
        "eta1": 0.001,
        "eta2": 0.0,
        "weight_loss_velocity": 0.1,
        "beta": 0.02,
    }


def _sampling_params() -> dict[str, object]:
    return {
        "num_samples": 16,
        "num_samples_elite": 4,
        "sigma": 0.1,
        "lambda": 0.1,
        "learning_rate": 1.0,
        "update_cycle": 2,
        "rng_seed": 0,
        "device": "cpu",
        "jax_compilation_cache_enable": False,
    }


def _make_retargeter(
    monkeypatch, *, qpos_canned=None, **kwargs
) -> SamplingBasedRetargeter:
    """Construct a SamplingBasedRetargeter with the JIT step + warmup stubbed.

    Avoids triggering any real JAX compilation (which would be slow and
    would require FakeToModel.compute_keyvectors_jax to exist).
    """
    monkeypatch.setattr(
        bor_mod, "WristRetargeter", lambda *_a, **_k: FakeWristRetargeter()
    )

    def fake_align(points, source_landmarks, target_landmarks, precomputed_scale=None):
        return points.copy(), SciRotation.from_matrix(np.eye(3)), 1.0

    monkeypatch.setattr(sbr_mod, "align_pcloud_kabsch_umeyama", fake_align)

    # Skip warmup so __init__ doesn't trigger a real JIT compile.
    monkeypatch.setattr(
        SamplingBasedRetargeter, "_warmup_jax_compilation", lambda self: None
    )

    wrist_mapping = {
        "tgt_key": "wrist",
        "root_key": "base",
        "transform": np.eye(4, dtype=np.float32),
    }
    alignment = [
        HandLandmark.THUMB_BASE,
        HandLandmark.INDEX_BASE,
        HandLandmark.PINKY_BASE,
    ]
    r = SamplingBasedRetargeter(
        from_model=FakeFromModel(),
        to_model=FakeToModel(),
        wrist_mapping=wrist_mapping,
        alignment_landmarks=alignment,
        keyvectors_cfg=_cfg(),
        dexpilot_params=_dexpilot_params(),
        sampling_params=_sampling_params(),
        **kwargs,
    )

    # Stub the JIT step to a deterministic canned result so retarget() is
    # pure Python under test.
    canned = (
        np.array([0.1, -0.2], dtype=np.float32) if qpos_canned is None else qpos_canned
    )

    def fake_step(rng_key, qpos_current, targets, weights):
        rng_next = rng_key
        qpos_next = jnp.asarray(canned, dtype=jnp.float32)
        losses = jnp.zeros((1,), dtype=jnp.float32)
        return rng_next, qpos_next, losses

    monkeypatch.setattr(r, "_monte_carlo_step_fn", fake_step)
    return r


def test_normalize_keyvectors_cfg_assigns_groups(monkeypatch) -> None:
    """tgt_key prefix maps to the right group label."""
    r = _make_retargeter(monkeypatch)
    groups = {kv["name"]: kv["group"] for kv in r.keyvectors_cfg}
    assert groups == {
        "wrist_to_index_tip": "wrist",
        "thumb_tip_to_index_tip": "thumb",
        "index_to_middle": "finger",
    }


def test_grasp_state_initialized_only_for_thumb_and_finger(monkeypatch) -> None:
    """Wrist entries don't participate in the hysteresis dict."""
    r = _make_retargeter(monkeypatch)
    assert set(r._dexpilot_grasp_states) == {
        "thumb_tip_to_index_tip",
        "index_to_middle",
    }
    assert all(v is False for v in r._dexpilot_grasp_states.values())


def test_init_bounds_clips_to_actuated_limits(monkeypatch) -> None:
    r = _make_retargeter(monkeypatch)
    np.testing.assert_array_equal(np.asarray(r._lower_bounds), [-1.0, -1.0])
    np.testing.assert_array_equal(np.asarray(r._upper_bounds), [1.0, 1.0])


def test_compute_dexpilot_loss_terms_inactive_uses_weight_default(monkeypatch) -> None:
    """All-inactive frame -> every weight equals weight_default (now respects config)."""
    r = _make_retargeter(monkeypatch)
    # Distances above project_distance, so no thumb engages.
    src_keyvectors = {
        "WRIST_to_INDEX_TIP": np.array([0.10, 0.0, 0.0], dtype=np.float32),
        "THUMB_TIP_to_INDEX_TIP": np.array([0.10, 0.0, 0.0], dtype=np.float32),
        "INDEX_TIP_to_MIDDLE_TIP": np.array([0.10, 0.0, 0.0], dtype=np.float32),
    }
    targets, weights = r._compute_dexpilot_loss_terms(src_keyvectors)
    np.testing.assert_allclose(np.asarray(weights), [1.0, 1.0, 1.0], atol=1e-6)
    # Targets are src_vec * scale_factor when inactive.
    np.testing.assert_allclose(
        np.asarray(targets),
        np.tile([0.10, 0.0, 0.0], (3, 1)),
        atol=1e-6,
    )


def test_compute_dexpilot_loss_terms_active_thumb_uses_set1_weight(monkeypatch) -> None:
    """Engaged thumb pulls weight from weight_vector_set1 and target from eta1."""
    r = _make_retargeter(monkeypatch)
    # Distance < project_distance for the thumb -> engages on this frame.
    src_keyvectors = {
        "WRIST_to_INDEX_TIP": np.array([0.10, 0.0, 0.0], dtype=np.float32),
        "THUMB_TIP_to_INDEX_TIP": np.array([0.005, 0.0, 0.0], dtype=np.float32),
        "INDEX_TIP_to_MIDDLE_TIP": np.array([0.10, 0.0, 0.0], dtype=np.float32),
    }
    _, weights = r._compute_dexpilot_loss_terms(src_keyvectors)
    # Order matches _cfg(): [wrist, thumb, finger].
    assert weights[1] == pytest.approx(2.0)  # weight_vector_set1
    assert r._dexpilot_grasp_states["thumb_tip_to_index_tip"] is True


def test_retarget_nominal_shape_and_qpos_prev(monkeypatch) -> None:
    """retarget() returns (1, dof) and updates _qpos_prev to the canned solution."""
    canned = np.array([0.1, -0.2], dtype=np.float32)
    r = _make_retargeter(monkeypatch, qpos_canned=canned)

    pcloud = np.zeros((1, 21, 3), dtype=np.float32)
    out, wrist = r.retarget(pcloud)

    assert out.shape == (1, 2)
    np.testing.assert_allclose(np.asarray(out), canned[np.newaxis], atol=1e-6)
    np.testing.assert_allclose(np.asarray(r._qpos_prev), canned, atol=1e-6)
    assert wrist is None


def test_retarget_clips_to_bounds(monkeypatch) -> None:
    """Out-of-bounds canned qpos gets clipped to [lower, upper] before being returned."""
    out_of_bounds = np.array([5.0, -5.0], dtype=np.float32)
    r = _make_retargeter(monkeypatch, qpos_canned=out_of_bounds)

    pcloud = np.zeros((1, 21, 3), dtype=np.float32)
    out, _ = r.retarget(pcloud)

    np.testing.assert_allclose(np.asarray(out), [[1.0, -1.0]], atol=1e-6)


def test_wrist_transform_path(monkeypatch) -> None:
    r = _make_retargeter(monkeypatch)
    pcloud = np.zeros((1, 21, 3), dtype=np.float32)
    wrist_in = np.ones((1, 4, 4), dtype=np.float32)
    _, wrist_out = r.retarget(pcloud, wrist_transform=wrist_in.copy())
    assert wrist_out is not None
    np.testing.assert_allclose(np.asarray(wrist_out), wrist_in + 2.0, atol=1e-6)


def test_debug_visualization_payload(monkeypatch) -> None:
    r = _make_retargeter(monkeypatch, debug_mode=True, debug_every_n_frames=1)
    pcloud = np.zeros((1, 21, 3), dtype=np.float32)
    r.retarget(pcloud)
    dbg = r.debug_visualization_data
    assert dbg is not None
    assert set(dbg) >= {
        "tgt_landmarks",
        "src_raw_points",
        "src_aligned_points",
        "src_keyvectors",
        "tgt_keyvectors",
        "grasp_active",
    }
    # One flag per cfg entry.
    assert dbg["grasp_active"].shape == (3,)


def test_debug_skipped_when_interval_mismatches(monkeypatch) -> None:
    r = _make_retargeter(monkeypatch, debug_mode=True, debug_every_n_frames=2)
    pcloud = np.zeros((1, 21, 3), dtype=np.float32)
    r.retarget(pcloud)
    r.retarget(pcloud)
    assert r._retarget_step_idx == 2


def test_pcloud_shape_validation(monkeypatch) -> None:
    """4-D pcloud (or anything non-(N,3) / non-(1,N,3)) raises ValueError."""
    r = _make_retargeter(monkeypatch)
    bad = np.zeros((2, 4, 21, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="pcloud must have shape"):
        r.retarget(bad)
