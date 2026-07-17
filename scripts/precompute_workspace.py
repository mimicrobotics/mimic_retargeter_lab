#!/usr/bin/env python3
"""Precompute the robot's full reachable workspace (KP_R) per hand.

Uniformly samples joint angles within actuator limits, runs batched FK,
and caches fingertip positions as a .npz file per hand.

Usage:
    python scripts/precompute_workspace.py [--num-samples 100000] [--hands shadow_hand mimic_p050_hand ...]
"""

# Import mimic_retargeter_lab first — its package init pins ``JAX_PLATFORMS``
# and silences MJX's misleading "Using JAX default device" log. Must come
# before ``import jax`` because JAX caches platform priority at import time.
import mimic_retargeter_lab  # noqa: F401

import argparse
from pathlib import Path

import jax
import numpy as np

# JAX compilation cache
cache_dir = Path(__file__).parent.parent / ".jax_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(cache_dir))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

from mimic_retargeter_lab.hand_models import create_robot_hand
from mimic_retargeter_lab.types import Chirality, RobotHandType

ASSETS_DIR = Path(__file__).parent.parent / "assets" / "mjcf"
OUTPUT_DIR = Path(__file__).parent.parent / "assets" / "workspace_cache"

ALL_HANDS = [
    "shadow_hand",
    "mimic_p050_hand",
    "wonik_allegro_hand",
    "shadow_dexee_hand",
    "leap_hand",
    "wuji_hand",
    "orca_v2_hand",
]


def precompute_workspace(hand_name: str, num_samples: int, batch_size: int = 4096):
    """Sample joint limits uniformly and run FK to get fingertip positions."""
    print(f"\n{'=' * 60}")
    print(f"  Precomputing workspace for {hand_name}")
    print(f"  Samples: {num_samples:,}")
    print(f"{'=' * 60}")

    hand_path = ASSETS_DIR / hand_name
    hand_type = RobotHandType(hand_name)
    robot = create_robot_hand(hand_type, hand_path, Chirality.RIGHT)

    # Get actuated joint limits
    ctrl_limits = robot.get_actuated_joint_limits()
    actuated_names = robot.get_actuated_joint_names()

    mins = np.array([ctrl_limits[name][0] for name in actuated_names], dtype=np.float32)
    maxs = np.array([ctrl_limits[name][1] for name in actuated_names], dtype=np.float32)

    print(f"  Actuated DOFs: {len(actuated_names)}")
    print(f"  Joint ranges: min={mins.min():.3f}, max={maxs.max():.3f}")

    # Sample uniformly within limits
    rng = np.random.default_rng(42)
    ctrl_samples = rng.uniform(
        mins, maxs, size=(num_samples, len(actuated_names))
    ).astype(np.float32)

    # Run batched FK
    all_positions = {}
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch = ctrl_samples[start:end]
        positions = robot.mjx_fk_body_positions(batch, joint_space="ctrl")

        for name, pos in positions.items():
            pos_np = np.asarray(pos)
            if name not in all_positions:
                all_positions[name] = []
            all_positions[name].append(pos_np)

        if (start // batch_size) % 10 == 0:
            print(f"  FK progress: {end:,}/{num_samples:,}")

    # Concatenate batches
    workspace_pts = {
        name: np.concatenate(chunks, axis=0) for name, chunks in all_positions.items()
    }

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{hand_name}.npz"
    np.savez_compressed(out_path, **workspace_pts)

    print(f"  Saved {len(workspace_pts)} body positions to {out_path}")
    for name, pts in sorted(workspace_pts.items()):
        print(f"    {name}: {pts.shape}")

    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100_000,
        help="Number of random joint configurations to sample (default: 100,000)",
    )
    parser.add_argument(
        "--hands",
        nargs="+",
        default=ALL_HANDS,
        help=f"Hand names to process (default: {ALL_HANDS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="FK batch size (default: 8192)",
    )
    args = parser.parse_args()

    for hand_name in args.hands:
        precompute_workspace(hand_name, args.num_samples, args.batch_size)

    print(f"\nDone. Workspace caches saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
