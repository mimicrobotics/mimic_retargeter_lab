"""Play back a MANUS recording (TxNx7 NPZ) as a real-time 3D animation.

Loads an NPZ produced by ``record_manus_zmq.py`` and renders the skeleton with
matplotlib — node markers, ID labels, and bones from ``ManusHandModel``.
Playback is paced by the recorded ``timestamps`` array (or wall-clock if it is
missing), scaled by ``--speed``.

Usage:
    python scripts/playback_manus_npz.py dataset/manus/manus.npz
    python scripts/playback_manus_npz.py dataset/manus/manus.npz --speed 0.5 --loop
    python scripts/playback_manus_npz.py dataset/manus/manus.npz --start 100 --end 400
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dexworld.hand_models.manus_hand import ManusHandModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("npz", type=Path, help="Path to the recorded NPZ file.")
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (1.0 = real time).",
    )
    p.add_argument("--loop", action="store_true", help="Loop playback forever.")
    p.add_argument(
        "--start", type=int, default=0, help="First frame index (inclusive)."
    )
    p.add_argument(
        "--end", type=int, default=None, help="Last frame index (exclusive)."
    )
    p.add_argument("--no-labels", action="store_true", help="Hide node ID labels.")
    return p.parse_args()


def load_recording(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    npz = np.load(path, allow_pickle=False)
    if "data" not in npz.files:
        raise KeyError(f"{path} is missing required 'data' array.")
    data = npz["data"]  # (T, N, 7)
    if data.ndim != 3 or data.shape[2] != 7:
        raise ValueError(f"Expected data shape (T, N, 7), got {data.shape}.")

    timestamps = (
        npz["timestamps"] if "timestamps" in npz.files else np.arange(len(data)) / 60.0
    )
    chirality = str(npz["chirality"]) if "chirality" in npz.files else "?"
    return data, timestamps, chirality


def main() -> None:
    args = parse_args()
    data, timestamps, chirality = load_recording(args.npz)

    T, N, _ = data.shape
    end = args.end if args.end is not None else T
    if not (0 <= args.start < end <= T):
        raise ValueError(f"Invalid range [{args.start}, {end}) for T={T}.")

    positions = data[args.start : end, :, 0:3]
    ts = timestamps[args.start : end]
    ts = ts - ts[0]
    n_frames = positions.shape[0]

    print(f"Loaded {args.npz}: T={T}, N={N}, chirality={chirality}")
    print(
        f"Playing frames [{args.start}, {end}) — {n_frames} frames, {ts[-1]:.2f}s @ speed {args.speed}x"
    )

    # Bone topology (only edges referencing nodes that exist in this recording).
    links = [(a, b) for (a, b) in ManusHandModel.HAND_LINKS_25 if a < N and b < N]

    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Fixed limits over the full clip so the viewport doesn't jitter.
    all_pts = positions.reshape(-1, 3)
    center = all_pts.mean(axis=0)
    span = max(np.ptp(all_pts, axis=0)) * 0.6 or 0.1
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)

    scatter = ax.scatter(
        positions[0, :, 0],
        positions[0, :, 1],
        positions[0, :, 2],
        s=50,
        c="tab:blue",
        depthshade=True,
    )
    bone_lines = [ax.plot([], [], [], c="tab:gray", lw=1.5)[0] for _ in links]
    labels = []
    if not args.no_labels:
        labels = [
            ax.text(
                positions[0, i, 0],
                positions[0, i, 1],
                positions[0, i, 2],
                str(i),
                fontsize=8,
                color="red",
                fontweight="bold",
            )
            for i in range(N)
        ]
    title = ax.set_title("")

    try:
        while plt.fignum_exists(fig.number):
            wall_start = time.time()
            for f in range(n_frames):
                if not plt.fignum_exists(fig.number):
                    break
                pts = positions[f]
                scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
                for line, (a, b) in zip(bone_lines, links):
                    line.set_data([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]])
                    line.set_3d_properties([pts[a, 2], pts[b, 2]])
                for i, lbl in enumerate(labels):
                    lbl.set_position_3d(pts[i])
                title.set_text(
                    f"MANUS playback — {chirality}  frame {args.start + f}/{T - 1}  t={ts[f]:.2f}s"
                )

                target = ts[f] / max(args.speed, 1e-6)
                wait = target - (time.time() - wall_start)
                if wait > 0:
                    plt.pause(wait)
                else:
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()

            if not args.loop:
                break

    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.close("all")


if __name__ == "__main__":
    main()
