"""Unit tests for jax_utils device-placement helpers."""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mimic_retargeter_lab.utils.jax_utils import (
    device_put_attrs,
    rebuild_mjx_fk_on_device,
    resolve_jax_device,
)


class _FakeDevice:
    """Minimal jax.Device stand-in for guard-logic tests."""

    def __init__(self, platform: str, name: str = "fake") -> None:
        self.platform = platform
        self._name = name

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _FakeDevice)
            and self.platform == other.platform
            and self._name == other._name
        )

    def __hash__(self) -> int:
        return hash((self.platform, self._name))

    def __repr__(self) -> str:
        return f"_FakeDevice({self.platform}:{self._name})"


class _FakeModel:
    """Records calls to create_mjx_kinematic_model and tracks the pinned device."""

    def __init__(self, initial_device=None) -> None:
        self.calls: list = []
        if initial_device is not None:
            self._mjx_fk_device = initial_device

    def create_mjx_kinematic_model(self, device=None) -> None:
        self.calls.append(device)
        self._mjx_fk_device = device


def test_resolve_jax_device_cpu_returns_cpu() -> None:
    assert resolve_jax_device("cpu").platform == "cpu"


def test_resolve_jax_device_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_jax_device("tpu_v5")


def test_resolve_jax_device_gpu_falls_back_when_unavailable(
    monkeypatch, caplog
) -> None:
    """GPU request with no CUDA backend falls back to CPU + logs a warning.

    Simulates the no-CUDA-jaxlib environment by monkeypatching
    ``jax.devices`` to raise ``RuntimeError`` for the GPU backends — that
    way this path is exercised even on hosts that do have CUDA installed.
    """
    real_devices = jax.devices

    def fake_devices(backend=None):
        if backend in ("cuda", "gpu"):
            raise RuntimeError(f"No {backend} backend (simulated for test)")
        return real_devices(backend) if backend is not None else real_devices()

    monkeypatch.setattr(jax, "devices", fake_devices)

    logger = logging.getLogger("test_jax_utils.fallback")
    logger.propagate = True
    with caplog.at_level(logging.WARNING):
        dev = resolve_jax_device("cuda", logger=logger)
    assert dev.platform == "cpu"
    assert any("GPU requested" in rec.message for rec in caplog.records)


def test_device_put_attrs_moves_arrays_and_skips_none() -> None:
    cpu = jax.devices("cpu")[0]

    class Holder:
        a = jnp.zeros(3, dtype=jnp.float32)
        b = jnp.ones(2, dtype=jnp.float32)
        c = None

    h = Holder()
    device_put_attrs(h, ("a", "b", "c"), cpu)
    np.testing.assert_array_equal(np.asarray(h.a), np.zeros(3))
    np.testing.assert_array_equal(np.asarray(h.b), np.ones(2))
    assert h.c is None


def test_device_put_attrs_typo_raises_attribute_error() -> None:
    cpu = jax.devices("cpu")[0]

    class Holder:
        a = jnp.zeros(3, dtype=jnp.float32)

    with pytest.raises(AttributeError):
        device_put_attrs(Holder(), ("a", "doesnt_exist"), cpu)


def test_rebuild_mjx_fk_on_device_cpu_is_noop() -> None:
    model = _FakeModel()
    rebuild_mjx_fk_on_device(model, _FakeDevice("cpu"))
    assert model.calls == []
    assert not hasattr(model, "_mjx_fk_device")


def test_rebuild_mjx_fk_on_device_first_gpu_call_builds() -> None:
    model = _FakeModel()
    gpu = _FakeDevice("cuda", "0")
    rebuild_mjx_fk_on_device(model, gpu)
    assert model.calls == [gpu]
    assert model._mjx_fk_device == gpu


def test_rebuild_mjx_fk_on_device_same_gpu_is_noop() -> None:
    """Second consumer requesting the same device shares the existing build."""
    gpu = _FakeDevice("cuda", "0")
    model = _FakeModel(initial_device=gpu)
    rebuild_mjx_fk_on_device(model, gpu)
    assert model.calls == []


def test_rebuild_mjx_fk_on_device_conflicting_gpu_raises() -> None:
    """Two non-CPU consumers on a shared model -> loud RuntimeError."""
    model = _FakeModel(initial_device=_FakeDevice("cuda", "0"))
    with pytest.raises(RuntimeError, match="already pinned"):
        rebuild_mjx_fk_on_device(model, _FakeDevice("cuda", "1"))
    assert model.calls == []


def test_rebuild_mjx_fk_on_device_cpu_then_gpu_proceeds() -> None:
    """CPU-first then GPU is allowed (guard only fires on non-CPU current)."""
    model = _FakeModel(initial_device=_FakeDevice("cpu"))
    gpu = _FakeDevice("cuda", "0")
    rebuild_mjx_fk_on_device(model, gpu)
    assert model.calls == [gpu]
    assert model._mjx_fk_device == gpu
