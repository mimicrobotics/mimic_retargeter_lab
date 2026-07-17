#!/usr/bin/env python3
"""Profile an online retargeter's ``retarget()`` with cProfile (default: 100 calls).

Run from repo root (use the project venv if ``uv run`` cannot reach PyPI):

  .venv/bin/python scripts/profile_retargeter.py --retargeter dexpilot --hand mimic_p050_hand --n 100

JAX-backed retargeters trigger a long **first-time JAX/MJX compile**; you may
see no stats until that finishes — messages below show progress.

  uv run python scripts/profile_retargeter.py --retargeter keyvector --hand mimic_p050_hand --n 50
  uv run python scripts/profile_retargeter.py --retargeter dexpilot  --out dexpilot.prof

By default the **cProfile stats table** is written to a ``.txt`` file in the current working
directory (``<retargeter>_cprofile_<hand>_n<n>.txt``). Use ``--report PATH`` to set the
file, ``--stdout-stats`` to also print the table to stdout, or ``--no-report`` for stdout only.
"""

from __future__ import annotations

import argparse
import cProfile
import logging
import pstats
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from mimic_retargeter_lab.hand_models import create_robot_hand
from mimic_retargeter_lab.hand_models.mano_keypoint_hand import ManoKeypointHandModel
from mimic_retargeter_lab.retargeting.online import create_retargeter
from mimic_retargeter_lab.types import Chirality, HandLandmark, Retargeter, RobotHandType


def _coerce_retargeter_kwargs(cfg: dict) -> dict:
    out = dict(cfg)
    if "alignment_landmarks" in out:
        out["alignment_landmarks"] = [
            HandLandmark(x) if isinstance(x, str) else x
            for x in out["alignment_landmarks"]
        ]
    return out


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retargeter",
        type=str,
        default="keyvector",
        choices=[r.value for r in Retargeter],
        help="Which online retargeter to profile (matches config/retargeter_cfg/<type>/).",
    )
    parser.add_argument(
        "--hand",
        type=str,
        default="shadow_hand",
        choices=[h.value for h in RobotHandType],
        help="Robot hand (must match assets/mjcf/<name> and the retargeter yaml).",
    )
    parser.add_argument(
        "--n", type=int, default=100, help="Number of retarget() calls."
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="cumulative",
        choices=("cumulative", "tottime", "calls"),
        help="pstats sort key (default: cumulative time).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write binary .prof for snakeviz / py-spy.",
    )
    parser.add_argument("--top", type=int, default=60, help="Lines of stats to print.")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write the cProfile stats table to this .txt file. "
        "If omitted, uses keyvector_cprofile_<hand>_n<n>.txt in the current working directory.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write a .txt report; print the stats table to stdout only.",
    )
    parser.add_argument(
        "--stdout-stats",
        action="store_true",
        help="Print the stats table to stdout as well as writing --report.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of untimed warm-up calls (excluded from cProfile and wall stats). "
        "JAX backends need >=1 to amortize JIT compile; PyTorch benefits from 1-2 to "
        "warm up the optimizer cache.",
    )
    args = parser.parse_args()
    if args.no_report and args.report is not None:
        parser.error("Use either --no-report or --report PATH, not both.")

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    # Silence the chosen retargeter's debug logs during profiling.
    logging.getLogger(
        f"mimic_retargeter_lab.retargeting.online.{args.retargeter}_retargeter"
    ).setLevel(logging.WARNING)

    retargeter_type = Retargeter(args.retargeter)
    log(
        f"profile_retargeter: retargeter={retargeter_type.value!r} "
        f"hand={args.hand!r} n={args.n} (repo {repo})"
    )

    hand_type = RobotHandType(args.hand)
    retargeter_yaml = (
        repo
        / "config"
        / "retargeter"
        / retargeter_type.value
        / f"human_hand_to_{args.hand}.yaml"
    )
    if not retargeter_yaml.is_file():
        raise SystemExit(f"Missing retargeter config: {retargeter_yaml}")

    log(f"Loading retargeter YAML: {retargeter_yaml.name}")
    raw = OmegaConf.load(str(retargeter_yaml))
    retargeter_cfg = OmegaConf.to_container(raw.config, resolve=True)
    assert isinstance(retargeter_cfg, dict)
    retargeter_kwargs = _coerce_retargeter_kwargs(retargeter_cfg)

    log("Building ManoKeypointHandModel…")
    human = ManoKeypointHandModel(
        chirality=Chirality.RIGHT,
    )
    log(f"Building robot {hand_type.value} (MJCF under assets/mjcf/)…")
    robot = create_robot_hand(
        hand_type, repo / "assets" / "mjcf" / hand_type.value, Chirality.RIGHT
    )

    log(
        f"Building {retargeter_type.value} retargeter "
        "(MJX init / JAX compile can take 30–120s here)…"
    )
    retargeter = create_retargeter(
        retargeter_type,
        from_model=human,
        to_model=robot,
        **retargeter_kwargs,
    )
    log("Retargeter ready.")

    rng = np.random.default_rng(0)
    base = rng.normal(size=(1, 21, 3)).astype(np.float32) * 0.02
    # Slightly different pose each call so the optimizer does real work.
    total_calls = args.warmup + args.n
    pclouds = [
        (base + rng.normal(size=base.shape).astype(np.float32) * 0.005)
        for _ in range(total_calls)
    ]
    warmup_pclouds = pclouds[: args.warmup]
    timed_pclouds = pclouds[args.warmup :]

    # JAX backends are async: dispatch returns immediately and the work happens
    # later. For a fair wall-clock measurement we have to force a sync after
    # each retarget(). PyTorch returns synchronously, so this is a no-op.
    try:
        import jax  # type: ignore

        _has_jax = True
    except ImportError:
        _has_jax = False

    def _sync(result) -> None:
        if not _has_jax or result is None:
            return
        try:
            qpos = result[0] if isinstance(result, tuple) else result
            jax.block_until_ready(qpos)
        except Exception:
            pass

    progress_every = max(1, args.n // 10)

    if args.warmup > 0:
        log(
            f"Warm-up: running {args.warmup} untimed call(s) "
            "(JAX backends will JIT-compile during this phase)."
        )
        for pc in warmup_pclouds:
            _sync(retargeter.retarget(pc))
        log("Warm-up complete.")

    def run_loop() -> None:
        for i, pc in enumerate(timed_pclouds, start=1):
            retargeter.retarget(pc)
            if i % progress_every == 0 or i == args.n:
                log(f"  retarget {i}/{args.n} …")

    log(f"Wall-clock pass: timing {args.n} retarget() call(s) with sync.")
    per_call_ms: list[float] = []
    for pc in timed_pclouds:
        t0 = time.perf_counter()
        result = retargeter.retarget(pc)
        _sync(result)
        t1 = time.perf_counter()
        per_call_ms.append((t1 - t0) * 1000.0)
    per_call_ms_sorted = sorted(per_call_ms)

    def _percentile(p: float) -> float:
        idx = max(0, int(round(p * len(per_call_ms_sorted))) - 1)
        return per_call_ms_sorted[idx]

    p95 = _percentile(0.95)
    p99 = _percentile(0.99)
    stdev = statistics.stdev(per_call_ms) if len(per_call_ms) > 1 else 0.0
    # Warn if n is too small to estimate p99 reliably (~10x the denominator).
    p99_note = "" if args.n >= 1000 else "  (note: n<1000 — p99 estimate is noisy)\n"
    wall_clock_summary = (
        f"Wall-clock per-call latency over {args.n} calls "
        f"(warmup={args.warmup}, sync=block_until_ready):\n"
        f"  mean   = {statistics.mean(per_call_ms):.3f} ms\n"
        f"  median = {statistics.median(per_call_ms):.3f} ms\n"
        f"  stdev  = {stdev:.3f} ms\n"
        f"  p95    = {p95:.3f} ms\n"
        f"  p99    = {p99:.3f} ms\n"
        f"{p99_note}"
        f"  min    = {min(per_call_ms):.3f} ms\n"
        f"  max    = {max(per_call_ms):.3f} ms\n"
    )
    log(wall_clock_summary.rstrip())

    log(
        f"cProfile: timing {args.n} retarget() call(s). "
        "First call may JIT-compile JAX again — be patient."
    )
    profiler = cProfile.Profile()
    profiler.enable()
    run_loop()
    profiler.disable()
    log("Done profiling; writing stats.")

    if args.out is not None:
        profiler.dump_stats(str(args.out))
        log(f"Wrote binary profile: {args.out}")

    if args.no_report:
        report_path: Path | None = None
    elif args.report is not None:
        report_path = args.report
    else:
        report_path = (
            Path.cwd() / f"{retargeter_type.value}_cprofile_{args.hand}_n{args.n}.txt"
        )

    header = (
        f"cProfile: {args.n} calls to {type(retargeter).__name__}.retarget() "
        f"(retargeter={retargeter_type.value}, hand={args.hand}, sort={args.sort})\n\n"
    )

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            f.write(header)
            f.write(wall_clock_summary)
            f.write("\n")
            s_file = pstats.Stats(profiler, stream=f)
            s_file.strip_dirs()
            s_file.sort_stats(args.sort)
            s_file.print_stats(args.top)
        log(f"Wrote stats report: {report_path.resolve()}")

    if args.no_report or args.stdout_stats:
        print(header, end="", file=sys.stdout, flush=True)
        print(wall_clock_summary, file=sys.stdout, flush=True)
        s_out = pstats.Stats(profiler, stream=sys.stdout)
        s_out.strip_dirs()
        s_out.sort_stats(args.sort)
        s_out.print_stats(args.top)


if __name__ == "__main__":
    main()
