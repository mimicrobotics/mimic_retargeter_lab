#!/usr/bin/env python3
"""Diagnostic: verify each retargeter's persistent JAX state lives on the
device requested in its YAML config.

Spot-checks the four-layer GPU correctness story (see ``docs/jax_gpu_setup.md``):
- Resolves the ``device`` knob from the retargeter's YAML.
- Instantiates the retargeter with a real ``to_model`` and a stub ``from_model``
  (the constructor never calls source-hand methods).
- Reads the device of each persistent JAX attribute the JIT'd loss closes
  over and asserts it matches ``self._device``.
- Runs one MJX FK call through the loss path and asserts the output device
  matches too — this catches Layer 2 silently regressing (closure captured a
  CPU model after a missed rebuild).

Usage
-----
    # Test all four retargeters with the device each YAML specifies (CPU
    # for keyvector/dexpilot/ako, CUDA for sampling_based by default):
    uv run python scripts/check_retargeter_devices.py

    # Override every retargeter to the same device:
    uv run python scripts/check_retargeter_devices.py --device cuda

    # Test a single retargeter:
    uv run python scripts/check_retargeter_devices.py --retargeter ako --device cuda

    # Different hand:
    uv run python scripts/check_retargeter_devices.py --hand shadow_hand
"""

# Import dexworld first — its package init pins JAX_PLATFORMS
# and silences MJX's misleading "Using JAX default device" log.
import dexworld  # noqa: F401

import argparse
from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf

from dexworld.hand_models import create_robot_hand
from dexworld.retargeting.online import create_retargeter
from dexworld.types import Chirality, Retargeter, RobotHandType


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config" / "retargeter"
ASSETS_DIR = REPO_ROOT / "assets" / "mjcf"

ALL_RETARGETERS = ("keyvector", "dexpilot", "ako", "sampling_based")

# Where the `device` knob lives inside each retargeter's `config:` block.
DEVICE_PATH: dict[str, tuple[str, ...]] = {
    "keyvector": ("device",),
    "dexpilot": ("solver_params", "device"),
    "ako": ("solver_params", "device"),
    "sampling_based": ("sampling_params", "device"),
}

# Persistent JAX attrs each retargeter pins on `self._device`. Mirrors the
# attr lists passed to ``device_put_attrs`` in each retargeter's __init__.
INSPECT_ATTRS: dict[str, tuple[str, ...]] = {
    "keyvector": ("_lower_bounds", "_upper_bounds", "_qpos_prev"),
    "dexpilot": (
        "_lower_bounds",
        "_upper_bounds",
        "_qpos_prev",
        "_etas",
        "_active_weights",
        "_inactive_weights",
        "_scaling_coefs",
    ),
    "ako": (
        "_lower_bounds",
        "_upper_bounds",
        "_qpos_prev",
        "_reg_ref",
        "_reg_weights",
    ),
    "sampling_based": (
        "_lower_bounds",
        "_upper_bounds",
        "_q_neutral",
        "_qpos_prev",
        "_jax_rng_key",
    ),
}


class _StubFromModel:
    """Stand-in for the human hand model. Retargeter constructors only store
    the reference; methods are called from ``retarget()`` which we don't run."""

    def get_landmarks(self, *_a, **_k):
        raise RuntimeError("from_model.get_landmarks not callable in diagnostic")

    def compute_keyvectors(self, *_a, **_k):
        raise RuntimeError("from_model.compute_keyvectors not callable in diagnostic")


def _set_in(d: dict, keys: tuple[str, ...], value: str) -> None:
    """Set ``d[keys[0]][keys[1]]...[keys[-1]] = value``, creating dicts."""
    target = d
    for k in keys[:-1]:
        target = target.setdefault(k, {})
    target[keys[-1]] = value


def _check_attr(retargeter, attr_name: str, expected: jax.Device) -> bool:
    """Print one attr's device and return whether it matches ``expected``."""
    if not hasattr(retargeter, attr_name):
        print(f"    [SKIP] {attr_name}: not present")
        return True
    val = getattr(retargeter, attr_name)
    if val is None:
        print(f"    [SKIP] {attr_name}: None")
        return True
    actual = val.device
    ok = actual == expected
    marker = "OK " if ok else "BAD"
    print(f"    [{marker}] {attr_name}: {actual}")
    return ok


def _check_mjx_fk(to_model, expected: jax.Device) -> bool:
    """Run a 1-row FK with the input on ``expected`` (mirroring how the
    retargeter's JIT'd loss invokes it from inside ``jax.default_device``).

    Catches the case where ``rebuild_mjx_fk_on_device`` was skipped/broken
    and the closure-captured MJX arrays are on the wrong device — JAX would
    fall back to a host transfer or recompile, and ``kv[k].device`` would
    not match ``expected``.
    """
    qpos = jax.device_put(
        np.zeros((1, to_model.num_actuated_dofs), dtype=np.float32),
        expected,
    )
    with jax.default_device(expected):
        kv = to_model.compute_keyvectors_jax(qpos, joint_space="ctrl")
    first_key = next(iter(kv))
    actual = kv[first_key].device
    ok = actual == expected
    marker = "OK " if ok else "BAD"
    print(f"    [{marker}] MJX FK output device (key={first_key!r}): {actual}")
    return ok


def _check_one(rt_name: str, hand: str, device_override: str | None) -> bool:
    cfg_path = CONFIG_DIR / rt_name / f"human_hand_to_{hand}.yaml"
    if not cfg_path.exists():
        print(f"\n=== {rt_name} ===")
        print(f"    [SKIP] config not found: {cfg_path}")
        return True

    cfg_full = OmegaConf.load(cfg_path)
    cfg = OmegaConf.to_container(cfg_full, resolve=True)["config"]

    if device_override:
        _set_in(cfg, DEVICE_PATH[rt_name], device_override)

    # Fresh hand model per retargeter — ``rebuild_mjx_fk_on_device`` mutates
    # the to_model's FK kernel, and we want each test to start from CPU.
    hand_path = ASSETS_DIR / hand
    to_model = create_robot_hand(RobotHandType(hand), hand_path, Chirality.RIGHT)
    from_model = _StubFromModel()

    print(f"\n=== {rt_name} ===")
    try:
        retargeter = create_retargeter(
            Retargeter(rt_name),
            from_model=from_model,
            to_model=to_model,
            **cfg,
        )
    except Exception as exc:
        print(f"    [BAD] FAILED to instantiate: {type(exc).__name__}: {exc}")
        return False

    expected = retargeter._device
    print(f"    configured device: {expected}")

    all_ok = True
    for attr in INSPECT_ATTRS[rt_name]:
        all_ok = _check_attr(retargeter, attr, expected) and all_ok
    all_ok = _check_mjx_fk(to_model, expected) and all_ok
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hand",
        default="orca_v2_hand",
        help="Robot hand name (default: orca_v2_hand). Must match the "
        "RobotHandType enum value and the YAML filename suffix.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=("cpu", "cuda", "gpu"),
        help="Override every retargeter's device knob to this value. "
        "Default: leave each YAML's setting alone.",
    )
    parser.add_argument(
        "--retargeter",
        default=None,
        choices=ALL_RETARGETERS,
        help="Test only one retargeter type. Default: all four.",
    )
    args = parser.parse_args()

    print(f"Hand: {args.hand}")
    print(f"Device override: {args.device or '(per-YAML)'}")
    print(f"Available JAX backends: {jax.default_backend()}")
    try:
        cuda_devs = jax.devices("cuda")
    except RuntimeError:
        cuda_devs = []
    print(f"CUDA devices: {cuda_devs or '(none)'}")

    targets = (args.retargeter,) if args.retargeter else ALL_RETARGETERS
    overall_ok = True
    for rt_name in targets:
        overall_ok = _check_one(rt_name, args.hand, args.device) and overall_ok

    print()
    if overall_ok:
        print("All checks passed.")
        return 0
    print("FAILED — at least one retargeter's state is on the wrong device.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
