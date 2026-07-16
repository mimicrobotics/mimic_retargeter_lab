"""Render a slow 360° WebP orbit animation of each robot hand at its neutral pose.

Default: 8-second seamless loop, ~45°/s, with an injected overhead spotlight.
Pass a smaller ``--sweep`` to switch to a sinusoidal swing instead.

Usage:
    python scripts/render_hand_orbit.py --hand shadow_hand
    python scripts/render_hand_orbit.py --all

Headless servers: set MUJOCO_GL=osmesa (or egl) before invocation.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent

HAND_SCENES: dict[str, Path] = {
    "mimic_p050_hand": REPO_ROOT / "assets/mjcf/mimic_p050_hand/scene_right.xml",
    "shadow_hand": REPO_ROOT / "assets/mjcf/shadow_hand/scene_right.xml",
    "shadow_dexee_hand": REPO_ROOT / "assets/mjcf/shadow_dexee_hand/scene.xml",
    "wonik_allegro_hand": REPO_ROOT / "assets/mjcf/wonik_allegro_hand/scene_right.xml",
    "leap_hand": REPO_ROOT / "assets/mjcf/leap_hand/scene_right.xml",
    "wuji_hand": REPO_ROOT / "assets/mjcf/wuji_hand/scene_right.xml",
    "orca_v2_hand": REPO_ROOT / "assets/mjcf/orca_v2_hand/scene_right.xml",
}


def build_model(scene_path: Path, add_spotlight: bool) -> mujoco.MjModel:
    """Compile the scene, optionally injecting an overhead spotlight."""
    cwd = os.getcwd()
    try:
        os.chdir(scene_path.parent)
        spec = mujoco.MjSpec.from_file(str(scene_path))
        if add_spotlight:
            spot = spec.worldbody.add_light()
            spot.name = "render_spot"
            spot.mode = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
            spot.pos = [0.0, 0.0, 1.2]
            spot.dir = [0.0, 0.0, -1.0]
            spot.diffuse = [1.0, 1.0, 1.0]
            spot.specular = [0.4, 0.4, 0.4]
            spot.cutoff = 45.0
            spot.exponent = 10.0
            spot.castshadow = True
        model = spec.compile()
    finally:
        os.chdir(cwd)
    # Brighten the camera-tracking headlight so the side of the hand facing
    # the camera is always well-lit (acts as a fill light).
    model.vis.headlight.ambient[:] = [0.35, 0.35, 0.35]
    model.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
    model.vis.headlight.specular[:] = [0.3, 0.3, 0.3]
    return model


def reset_to_neutral(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "open hand")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_kinematics(model, data)


def compute_hand_bounds(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[np.ndarray, float]:
    """World-frame AABB over hand geoms only (excludes worldbody floor)."""
    mask = model.geom_bodyid != 0
    if not mask.any():
        mask = np.ones_like(mask, dtype=bool)
    pos = data.geom_xpos[mask]
    r = model.geom_rbound[mask][:, None]
    mins = (pos - r).min(axis=0)
    maxs = (pos + r).max(axis=0)
    center = (mins + maxs) / 2
    diag = float(np.linalg.norm(maxs - mins))
    return center, diag


def azimuth_at(i: int, n_frames: int, start_az: float, sweep: float) -> float:
    """Azimuth for frame i. Sinusoidal swing for sweep < 360, continuous orbit otherwise."""
    if sweep >= 360.0:
        return start_az + 360.0 * (i / n_frames)
    return start_az + 0.5 * sweep * np.sin(2.0 * np.pi * i / n_frames)


def encode_webp_with_spinner(
    out_path: Path, frames: np.ndarray, hand: str, **kwargs
) -> None:
    """Run the (slow) WebP encode in a thread; show an elapsed-time spinner."""
    err: dict[str, BaseException] = {}

    def _encode():
        try:
            iio.imwrite(out_path, frames, **kwargs)
        except BaseException as e:
            err["e"] = e

    t = threading.Thread(target=_encode, daemon=True)
    t.start()
    bar = tqdm(
        desc=f"{hand} encoding",
        bar_format="{desc} [{elapsed}]",
        leave=False,
    )
    while t.is_alive():
        bar.refresh()
        time.sleep(0.25)
    t.join()
    bar.close()
    if "e" in err:
        raise err["e"]


def render_orbit(
    hand: str,
    out_dir: Path,
    size: int,
    n_frames: int,
    fps: int,
    elevation: float,
    distance_scale: float,
    start_azimuth: float,
    sweep: float,
    spotlight: bool,
    quality: int,
    tqdm_position: int = 0,
) -> Path:
    scene_path = HAND_SCENES[hand]
    model = build_model(scene_path, add_spotlight=spotlight)
    data = mujoco.MjData(model)
    reset_to_neutral(model, data)

    center, diag = compute_hand_bounds(model, data)
    fovy_rad = np.deg2rad(model.vis.global_.fovy)
    fit_distance = diag / (2.0 * np.tan(fovy_rad / 2.0))

    camera = mujoco.MjvCamera()
    camera.lookat[:] = center
    camera.distance = fit_distance * distance_scale
    camera.elevation = elevation

    renderer = mujoco.Renderer(model, height=size, width=size)
    frames: list[np.ndarray] = []
    for i in tqdm(
        range(n_frames),
        desc=f"{hand} render",
        unit="frame",
        leave=False,
        position=tqdm_position,
    ):
        camera.azimuth = azimuth_at(i, n_frames, start_azimuth, sweep)
        renderer.update_scene(data, camera)
        frames.append(renderer.render().copy())
    renderer.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{hand}.webp"
    encode_webp_with_spinner(
        out_path,
        np.stack(frames),
        hand,
        extension=".webp",
        duration=int(1000 / fps),
        loop=0,
        lossless=False,
        quality=quality,
        method=6,
        minimize_size=True,
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--hand", choices=sorted(HAND_SCENES.keys()))
    group.add_argument("--all", action="store_true")
    parser.add_argument("--size", type=int, default=480)
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--elevation", type=float, default=-10.0)
    parser.add_argument(
        "--start-azimuth",
        type=float,
        default=90.0,
        help="Center azimuth of the rotation, in degrees.",
    )
    parser.add_argument(
        "--sweep",
        type=float,
        default=360.0,
        help="Total angular range. 360 = full continuous orbit (default); "
        "smaller values = sinusoidal swing around start-azimuth.",
    )
    parser.add_argument(
        "--distance-scale",
        type=float,
        default=0.8,
        help="Camera distance as a multiple of the minimum 'fit' distance "
        "(1.0 just fits the AABB; <1 zooms in, may clip).",
    )
    parser.add_argument(
        "--no-spotlight",
        dest="spotlight",
        action="store_false",
        help="Disable the injected overhead spotlight (keep only the scene's lights).",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=80,
        help="WebP quality 0-100. Lower = smaller file. 60-70 looks fine for these.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "media/hands/",
    )
    args = parser.parse_args()

    hands = sorted(HAND_SCENES.keys()) if args.all else [args.hand]
    is_multi = len(hands) > 1
    hands_iter = (
        tqdm(hands, desc="hands", unit="hand", position=0) if is_multi else hands
    )
    for hand in hands_iter:
        out = render_orbit(
            hand=hand,
            out_dir=args.out_dir,
            size=args.size,
            n_frames=args.frames,
            fps=args.fps,
            elevation=args.elevation,
            distance_scale=args.distance_scale,
            start_azimuth=args.start_azimuth,
            sweep=args.sweep,
            spotlight=args.spotlight,
            quality=args.quality,
            tqdm_position=1 if is_multi else 0,
        )
        size_kb = out.stat().st_size / 1024
        try:
            display_path = out.relative_to(REPO_ROOT)
        except ValueError:
            display_path = out
        tqdm.write(f"[render] -> {display_path} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
