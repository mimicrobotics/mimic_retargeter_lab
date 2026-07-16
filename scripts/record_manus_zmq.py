"""Record MANUS skeleton frames streamed over ZMQ to an NPZ file.

Subscribes to the ManusZmqBridge PUB socket and buffers every frame.  On exit
(Ctrl+C, duration elapsed, or max_frames reached) writes the buffered data to
an NPZ file with the following arrays:

    data         : (T, N, 7)  float32   — [px, py, pz, qw, qx, qy, qz] per keypoint
    timestamps   : (T,)       float64   — wall-clock seconds (time.time())
    keypoint_ids : (N,)       int32     — MANUS keypoint IDs (0..N-1)

Quaternions are stored in MANUS-native (w, x, y, z) order — the same order the
SDK publishes.  N is 25 by default (the MANUS Pro skeleton).

Output naming convention: when ``-o/--output`` is not given, the filename is
constructed as ``manus_<chirality>_subject-<SUBJECT>_run-<RUN>.npz`` (with
chirality lowercased and run zero-padded to 3 digits) inside ``--output-dir``.

Prerequisites:
    ManusZmqBridge.out is running and publishing to tcp://HOST:PORT.

Usage:
    # Constructed name -> dataset/manus/manus_right_subject-RJM_run-010.npz
    python scripts/record_manus_zmq.py --subject RJM --run 10
    python scripts/record_manus_zmq.py --chirality LEFT --subject RJM --run 11 --duration 10

    # Explicit output path (overrides constructed name)
    python scripts/record_manus_zmq.py -o recordings/right.npz
"""

import argparse
import datetime as dt
import json
import logging
import time
from pathlib import Path

import numpy as np
import tqdm
import zmq

HOST_DEFAULT = "127.0.0.1"
PORT_DEFAULT = 8000
NUM_KEYPOINTS_DEFAULT = 25
OUTPUT_DIR_DEFAULT = Path("dataset/manus")
MANUS_VERSION_DEFAULT = "3.1.1"

logger = logging.getLogger("record_manus_zmq")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Explicit output NPZ path. If omitted, constructed from "
        "--subject and --run inside --output-dir.",
    )
    p.add_argument(
        "--subject",
        default=None,
        help="Subject name (e.g., RJM). Required if --output is not given.",
    )
    p.add_argument(
        "--run",
        type=int,
        default=None,
        help="Run number (zero-padded to 3 digits). Required if --output is not given.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help=f"Directory for constructed filenames (default: {OUTPUT_DIR_DEFAULT}).",
    )
    p.add_argument(
        "--chirality",
        choices=["RIGHT", "LEFT"],
        default="RIGHT",
        type=lambda s: s.upper(),
        help="Which hand topic to subscribe to.",
    )
    p.add_argument("--host", default=HOST_DEFAULT)
    p.add_argument("--port", type=int, default=PORT_DEFAULT)
    p.add_argument(
        "--num-keypoints",
        type=int,
        default=NUM_KEYPOINTS_DEFAULT,
        help="Expected number of skeleton keypoints per frame.",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds (default: run until Ctrl+C).",
    )
    p.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=None,
        help="Stop after collecting this many frames (default: unbounded).",
    )
    p.add_argument(
        "--manus-version",
        default=MANUS_VERSION_DEFAULT,
        help=f"MANUS SDK version (recorded in metadata sidecar; "
        f"default: {MANUS_VERSION_DEFAULT}).",
    )
    p.add_argument(
        "--notes", default=None, help="Free-text note included in the metadata sidecar."
    )

    args = p.parse_args()

    # If -o points to a directory (existing dir, or trailing separator), redirect
    # it to --output-dir so the filename is still constructed from --subject/--run.
    # Lets `python record_manus_zmq.py -o dataset/manus/ --subject RJM --run 10`
    # do the obvious thing instead of producing a literal "dataset/manus.npz".
    if args.output is not None:
        out_str = str(args.output)
        if out_str.endswith("/") or args.output.is_dir():
            args.output_dir = args.output
            args.output = None

    if args.output is None:
        if args.subject is None or args.run is None:
            p.error(
                "Either --output (full file path) or both --subject and --run "
                "must be provided."
            )
        filename = f"manus_{args.chirality.lower()}_subject-{args.subject}_run-{args.run:03d}.npz"
        args.output = args.output_dir / filename

    return args


def write_metadata_sidecar(
    output_path: Path,
    *,
    manus_version: str,
    subject: str | None,
    run: int | None,
    chirality: str,
    data_shape: tuple,
    duration_sec: float,
    rate_hz: float,
    start_time: float,
    end_time: float,
    glove_ids: list,
    notes: str | None,
) -> Path:
    """Write a JSON metadata sidecar next to the NPZ with the same stem."""
    sidecar = output_path.with_suffix(".json")
    metadata = {
        "manus_version": manus_version,
        "subject": subject,
        "run": run,
        "chirality": chirality,
        "data_shape": [int(d) for d in data_shape],
        "data_shape_axes": ["frames", "keypoints", "channels"],
        "channel_layout": "px, py, pz, qw, qx, qy, qz",
        "duration_sec": round(float(duration_sec), 3),
        "avg_rate_hz": round(float(rate_hz), 2),
        "start_time": dt.datetime.fromtimestamp(start_time).isoformat(
            timespec="seconds"
        ),
        "end_time": dt.datetime.fromtimestamp(end_time).isoformat(timespec="seconds"),
        "quaternion_order": "wxyz",
        "glove_ids": [int(g) for g in glove_ids],
        "notes": notes,
    }
    with open(sidecar, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    return sidecar


def parse_frame(msg: str) -> dict | None:
    space_idx = msg.find(" ")
    if space_idx == -1:
        return None
    try:
        return json.loads(msg[space_idx + 1 :])
    except json.JSONDecodeError:
        return None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    # No CONFLATE: keep every frame the publisher sends.
    sock.setsockopt_string(zmq.SUBSCRIBE, args.chirality)
    endpoint = f"tcp://{args.host}:{args.port}"
    sock.connect(endpoint)
    logger.info(f"Connected to {endpoint}, subscribed to '{args.chirality}'.")
    logger.info("Recording... press Ctrl+C to stop.")

    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    glove_ids: set = set()

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    pbar = tqdm.tqdm(
        total=args.num_samples,
        unit="frame",
        desc=f"Recording {args.chirality}",
        dynamic_ncols=True,
    )

    start = time.time()
    try:
        while True:
            if args.duration is not None and (time.time() - start) >= args.duration:
                break
            if args.num_samples is not None and len(frames) >= args.num_samples:
                break

            # Poll so we can periodically check stop conditions even if no data flows.
            events = dict(poller.poll(timeout=200))
            if sock not in events:
                continue

            msg = sock.recv_string()
            ts = time.time()
            frame = parse_frame(msg)
            if frame is None:
                pbar.write("Skipped malformed frame.")
                continue

            # Bridge wire format publishes them as "nodes"; we use "keypoints" downstream.
            keypoints = frame.get("nodes", [])
            if len(keypoints) < args.num_keypoints:
                pbar.write(
                    f"Expected {args.num_keypoints} keypoints, got {len(keypoints)}; skipping."
                )
                continue

            arr = np.zeros((args.num_keypoints, 7), dtype=np.float32)
            for kp in keypoints:
                kp_id = kp["id"]
                if 0 <= kp_id < args.num_keypoints:
                    arr[kp_id, 0:3] = kp["pos"]
                    arr[kp_id, 3:7] = kp["quat"]  # (w, x, y, z) — MANUS native order

            frames.append(arr)
            timestamps.append(ts)

            gid = frame.get("glove_id")
            if gid is not None:
                glove_ids.add(gid)

            elapsed = ts - start
            rate = len(frames) / max(elapsed, 1e-6)
            if args.duration is not None:
                pbar.set_postfix_str(
                    f"{rate:.1f} Hz, {elapsed:.1f}/{args.duration:.0f}s"
                )
            else:
                pbar.set_postfix_str(f"{rate:.1f} Hz")
            pbar.update(1)

    except KeyboardInterrupt:
        pbar.write("Interrupted.")
    finally:
        pbar.close()
        sock.close()
        ctx.term()

    if not frames:
        logger.error("No frames captured — nothing to save.")
        return

    data = np.stack(frames, axis=0)  # (T, N, 7)
    ts_arr = np.asarray(timestamps, dtype=np.float64)
    keypoint_ids = np.arange(args.num_keypoints, dtype=np.int32)

    np.savez(
        args.output,
        data=data,
        timestamps=ts_arr,
        keypoint_ids=keypoint_ids,
        chirality=np.array(args.chirality),
        quaternion_order=np.array("wxyz"),
        glove_ids=np.array(sorted(glove_ids)),
    )

    duration = ts_arr[-1] - ts_arr[0] if len(ts_arr) > 1 else 0.0
    rate = (len(ts_arr) - 1) / duration if duration > 0 else float("nan")
    logger.info(
        f"Saved {data.shape} to {args.output} "
        f"({len(ts_arr)} frames, {duration:.2f}s, {rate:.1f} Hz)"
    )

    sidecar_path = write_metadata_sidecar(
        args.output,
        manus_version=args.manus_version,
        subject=args.subject,
        run=args.run,
        chirality=args.chirality,
        data_shape=data.shape,
        duration_sec=duration,
        rate_hz=rate,
        start_time=float(ts_arr[0]),
        end_time=float(ts_arr[-1]),
        glove_ids=sorted(glove_ids),
        notes=args.notes,
    )
    logger.info(f"Wrote metadata sidecar to {sidecar_path}")


if __name__ == "__main__":
    main()
