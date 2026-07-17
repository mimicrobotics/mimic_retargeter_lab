"""Unit tests for KeyvectorRetargeter with fakes and monkeypatched optimizer."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from mimic_retargeter_lab.retargeting.online import base_online_retargeter as bor_mod
from mimic_retargeter_lab.retargeting.online import keyvector_retargeter as kvr_mod
from mimic_retargeter_lab.retargeting.online.keyvector_retargeter import KeyvectorRetargeter
from mimic_retargeter_lab.types import HandLandmark


class FakeWristRetargeter:
    def retarget(self, wrist_transform: np.ndarray) -> np.ndarray:
        return wrist_transform + 2.0


class FakeOptimizeResult:
    """Canned result returned by a monkeypatched _optimize_controls."""

    def __init__(
        self,
        x_opt: np.ndarray,
        *,
        fail_optimize: bool = False,
        last_on_fail: np.ndarray | None = None,
    ) -> None:
        self._x_opt = np.asarray(x_opt, dtype=np.float32).reshape(-1)
        self._fail_optimize = fail_optimize
        self._last_on_fail = (
            np.asarray(last_on_fail, dtype=np.float32).reshape(-1)
            if last_on_fail is not None
            else self._x_opt.copy()
        )


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


class FakeFromModel:
    """Fake from_model that returns a fixed keyvector."""

    def get_landmarks(self, pcloud: np.ndarray) -> dict[HandLandmark, np.ndarray]:
        return _landmark_dict()

    def compute_keyvectors(self, data: np.ndarray, **kwargs) -> dict[str, np.ndarray]:
        if data.ndim == 1:
            return {"kv": np.zeros(3, dtype=np.float32)}
        b = data.shape[0]
        return {"kv": np.zeros((b, 3), dtype=np.float32)}


class FakeToModel:
    """Fake to_model that returns a fixed keyvector."""

    def __init__(self) -> None:
        self.num_actuated_dofs = 2

    def get_neutral_ctrl_pose(self) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)

    def get_landmarks(self, *args, **kwargs) -> dict[HandLandmark, np.ndarray]:
        return _landmark_dict()

    def get_fingertip_landmarks(self) -> list[HandLandmark]:
        return list(_FINGERTIP_LANDMARKS)

    def get_actuated_joint_limits(self) -> dict[str, tuple[float, float]]:
        return {"a": (-1.0, 1.0), "b": (-1.0, 1.0)}

    def get_actuated_joint_names(self) -> list[str]:
        return ["a", "b"]

    def compute_keyvectors(self, data: np.ndarray, **kwargs) -> dict[str, np.ndarray]:
        data = np.asarray(data)
        if data.ndim == 1:
            d = data.shape[0]
            if d >= 3:
                return {"kv": data[:3]}
            return {"kv": np.concatenate([data, np.zeros(3 - d, dtype=data.dtype)])}
        b, d = data.shape[0], data.shape[1]
        if d >= 3:
            kv = data[:, :3]
        else:
            pad = np.zeros((b, 3 - d), dtype=data.dtype)
            kv = np.concatenate([data, pad], axis=-1)
        return {"kv": kv}


def _make_retargeter(
    monkeypatch, *, fake_opt: FakeOptimizeResult, **kwargs
) -> KeyvectorRetargeter:
    """Make a KeyvectorRetargeter with fakes and monkeypatched optimizer."""
    monkeypatch.setattr(
        bor_mod, "WristRetargeter", lambda *_a, **_k: FakeWristRetargeter()
    )

    def fake_align(points, source_landmarks, target_landmarks, precomputed_scale=None):
        return points.copy(), SciRotation.from_matrix(np.eye(3)), 1.0

    monkeypatch.setattr(kvr_mod, "align_pcloud_kabsch_umeyama", fake_align)

    cfg = [
        {
            "name": "kv",
            "src_key": "kv",
            "tgt_key": "kv",
            "loss_coef": 1.0,
            "scaling_coef": 1.0,
        }
    ]
    wrist_mapping: dict[str, str | np.ndarray] = {
        "tgt_key": "wrist",
        "root_key": "base",
        "transform": np.eye(4, dtype=np.float32),
    }
    alignment = [
        HandLandmark.THUMB_BASE,
        HandLandmark.INDEX_BASE,
        HandLandmark.PINKY_BASE,
    ]
    keyvector_retargeter = KeyvectorRetargeter(
        from_model=FakeFromModel(),
        to_model=FakeToModel(),
        keyvectors_cfg=cfg,
        regularized_joints={},
        wrist_mapping=wrist_mapping,
        alignment_landmarks=alignment,
        iterations=2,
        **kwargs,
    )

    def _fake_optimize_controls(qpos_init_guess, src_keyvectors_jax):
        if fake_opt._fail_optimize:
            raise RuntimeError("optimizer failed")
        return jnp.asarray(fake_opt._x_opt, dtype=jnp.float32)

    monkeypatch.setattr(
        keyvector_retargeter, "_optimize_controls", _fake_optimize_controls
    )
    return keyvector_retargeter


def test_retarget_nominal_shape_and_qpos_prev(monkeypatch) -> None:
    """Test that the retargeter returns the correct shape and qpos_prev."""
    x_opt = np.array([0.1, -0.2], dtype=np.float32)
    r = _make_retargeter(monkeypatch, fake_opt=FakeOptimizeResult(x_opt))

    pcloud = np.zeros((1, 21, 3), dtype=np.float32)
    out, wrist = r.retarget(pcloud)

    assert out.shape == (1, 2)
    np.testing.assert_allclose(np.asarray(out), [[0.1, -0.2]], atol=1e-6)
    np.testing.assert_allclose(np.asarray(r._qpos_prev), [0.1, -0.2], atol=1e-6)
    assert wrist is None


def test_retarget_accepts_numpy_input(monkeypatch) -> None:
    """Test that the retargeter accepts numpy input."""
    x_opt = np.array([0.0, 0.0], dtype=np.float32)
    r = _make_retargeter(monkeypatch, fake_opt=FakeOptimizeResult(x_opt))
    pcloud = np.zeros((1, 21, 3), dtype=np.float32)
    out, _ = r.retarget(pcloud)
    assert np.asarray(out).shape == (1, 2)


def test_wrist_transform_path(monkeypatch) -> None:
    """Test that the retargeter accepts a wrist transform."""
    r = _make_retargeter(monkeypatch, fake_opt=FakeOptimizeResult(np.zeros(2)))
    pcloud = np.zeros((1, 21, 3), dtype=np.float32)
    wrist_in = np.ones((1, 4, 4), dtype=np.float32)
    _, wrist_out = r.retarget(pcloud, wrist_transform=wrist_in.copy())
    assert wrist_out is not None
    np.testing.assert_allclose(np.asarray(wrist_out), wrist_in + 2.0, atol=1e-6)


def test_debug_visualization_payload(monkeypatch) -> None:
    """Test that the retargeter returns debug visualization data."""
    r = _make_retargeter(
        monkeypatch,
        fake_opt=FakeOptimizeResult(np.zeros(2)),
        debug_mode=True,
        debug_every_n_frames=1,
    )
    pcloud = np.zeros((1, 21, 3), dtype=np.float32)
    r.retarget(pcloud)
    dbg = r.debug_visualization_data
    assert dbg is not None
    assert "tgt_landmarks" in dbg
    assert "src_keyvectors" in dbg
    # Keyvector debug payload now stores (N, 2, 3) segments rather than the
    # legacy (6, 3) fingertip landmarks. The test cfg uses tgt_key="kv"
    # (no "_to_" separator), so _keyvector_segments_from_config produces
    # the empty-segments placeholder of shape (0, 2, 3).
    assert dbg["tgt_keyvectors"].shape == (0, 2, 3)


def test_debug_skipped_when_interval_mismatches(monkeypatch) -> None:
    """Test that the retargeter skips debug visualization when the interval mismatches."""
    r = _make_retargeter(
        monkeypatch,
        fake_opt=FakeOptimizeResult(np.zeros(2)),
        debug_mode=True,
        debug_every_n_frames=2,
    )
    pcloud = np.zeros((1, 21, 3), dtype=np.float32)
    r.retarget(pcloud)
    assert r.debug_visualization_data is not None
    r.retarget(pcloud)
    assert r._retarget_step_idx == 2


def test_optimizer_suboptimal_result_propagates(monkeypatch) -> None:
    """Test that a suboptimal optimizer result still flows through the pipeline."""
    suboptimal = np.array([0.3, -0.4], dtype=np.float32)
    r = _make_retargeter(
        monkeypatch,
        fake_opt=FakeOptimizeResult(suboptimal),
    )
    pcloud = np.zeros((1, 21, 3), dtype=np.float32)
    out, _ = r.retarget(pcloud)
    expected = suboptimal[np.newaxis]
    np.testing.assert_allclose(np.asarray(out), expected, atol=1e-6)
