#!/usr/bin/env python3
"""Inspect a metrics-stats_*.pkl file produced by ``compute_hand_retargeter_pair_metrics.py``.

Walks every metric's nested structure and prints the canonical 12-stat
block (defined in ``mimic_retargeter_lab.metrics._stats.STAT_KEYS``) at each
stat-bearing location, then runs a coverage check that fails loudly if
any expected location is missing keys.

Usage
-----
    python scripts/inspect_metrics_pkl.py <pkl_path>
    python scripts/inspect_metrics_pkl.py <pkl_path> --metric Latency
    python scripts/inspect_metrics_pkl.py <pkl_path> --episode <ep_id>
    python scripts/inspect_metrics_pkl.py <pkl_path> --keys-only
    python scripts/inspect_metrics_pkl.py <pkl_path> --no-coverage
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from mimic_retargeter_lab.metrics._stats import STAT_KEYS


# Each metric maps to a function ``(ep_data) -> [(label, stat_block), ...]``
# enumerating every nested location that should carry a 12-stat block.
StatLocator = Callable[[Dict[str, Any]], List[Tuple[str, Dict[str, Any]]]]


def _collision_locs(ep: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    return [
        ("per_frame_max_depth_stats", ep["per_frame_max_depth_stats"]),
        ("penetration_depth_stats", ep["penetration_depth_stats"]),
    ]


def _flatness_locs(ep: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    out = []
    for finger, ff in ep.items():
        for emb in ("human", "robot"):
            if emb in ff:
                out.append((f"{finger}.{emb}", ff[emb]))
    return out


def _keyvector_locs(ep: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    out = []
    for kv_name, sub_dict in (ep.get("vector_metrics") or {}).items():
        for sub_name, blk in sub_dict.items():
            out.append((f"vector_metrics.{kv_name}.{sub_name}", blk))
    return out


def _latency_locs(ep: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    return [("(per-episode)", ep)]


def _motion_pres_locs(ep: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    return [
        (f"{finger}.pos_alignment", ff["pos_alignment"])
        for finger, ff in ep.items()
        if "pos_alignment" in ff
    ]


def _workspace_locs(ep: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    return [
        (f"utilization.{finger}.distance_stats", blk["distance_stats"])
        for finger, blk in (ep.get("utilization") or {}).items()
        if "distance_stats" in blk
    ]


METRIC_LOCATORS: Dict[str, StatLocator] = {
    "Collision": _collision_locs,
    "Flatness": _flatness_locs,
    "Keyvector Matching": _keyvector_locs,
    "Latency": _latency_locs,
    "Motion Preservation": _motion_pres_locs,
    "Workspace": _workspace_locs,
}


# Per-metric scalar / non-stat-block fields worth surfacing alongside the
# 12-stat blocks. These are pre-aggregated single values per episode (or
# small dicts) — they live in the PKL but don't belong to any
# `summarize_array` location, so the locator-based view skips them.
ScalarLister = Callable[[Dict[str, Any]], List[Tuple[str, Any]]]


def _collision_scalars(ep: Dict[str, Any]) -> List[Tuple[str, Any]]:
    out = [
        ("collision_rate_pct", ep.get("collision_rate_pct")),
        ("frames_with_collision", ep.get("frames_with_collision")),
        ("num_frames", ep.get("num_frames")),
        ("num_unique_colliding_pairs", ep.get("num_unique_colliding_pairs")),
        ("max_penetration_depth_mm", ep.get("max_penetration_depth_mm")),
        ("avg_penetration_depth_mm", ep.get("avg_penetration_depth_mm")),
        ("penetration_tolerance_mm", ep.get("penetration_tolerance_mm")),
    ]
    pairs = ep.get("collision_pairs") or {}
    out.append(("collision_pairs (count)", len(pairs)))
    return out


def _latency_scalars(ep: Dict[str, Any]) -> List[Tuple[str, Any]]:
    return [
        ("device", ep.get("device")),
        ("num_warmup", ep.get("num_warmup")),
        ("num_timed", ep.get("num_timed")),
    ]


def _workspace_scalars(ep: Dict[str, Any]) -> List[Tuple[str, Any]]:
    out = []
    for finger, blk in (ep.get("utilization") or {}).items():
        out.append(
            (
                f"utilization.{finger}",
                f"util={blk.get('utilization', float('nan')):.4f}  "
                f"hits={blk.get('hits')}/{blk.get('num_samples')}  "
                f"radius={blk.get('radius')}",
            )
        )
    return out


METRIC_SCALARS: Dict[str, ScalarLister] = {
    "Collision": _collision_scalars,
    "Latency": _latency_scalars,
    "Workspace": _workspace_scalars,
}


def fmt_stats(d: Dict[str, Any]) -> str:
    """Render the 12-stat block as a single human-readable line."""
    parts = []
    for k in STAT_KEYS:
        v = d.get(k)
        if v is None:
            parts.append(f"{k}=<missing>")
        elif k == "n":
            parts.append(f"{k}={int(v)}")
        else:
            parts.append(f"{k}={v:.4g}")
    return "  ".join(parts)


def missing_stat_keys(d: Dict[str, Any]) -> List[str]:
    return [k for k in STAT_KEYS if k not in d]


def list_top_level(data: Dict[str, Any]) -> None:
    print(f"Top-level metrics: {list(data.keys())}")
    for name, ep_dict in data.items():
        print(f"  {name}: {len(ep_dict)} episode(s)")
    print()


def print_metric(
    metric_name: str,
    ep_dict: Dict[str, Any],
    episode_filter: str | None,
    keys_only: bool,
) -> None:
    locator = METRIC_LOCATORS.get(metric_name)
    if locator is None:
        print(f"[{metric_name}]  (no locator registered — skipping detailed view)")
        return

    print(f"\n{'═' * 80}")
    print(f"  {metric_name}")
    print(f"{'═' * 80}")

    scalar_lister = METRIC_SCALARS.get(metric_name)
    for ep_id, ep_data in ep_dict.items():
        if episode_filter and episode_filter not in ep_id:
            continue
        print(f"\n  episode = {ep_id}")
        try:
            locs = locator(ep_data)
        except Exception as e:
            print(f"    ERROR walking episode: {e}")
            continue
        if not locs:
            print("    (no stat-block locations found)")
        for label, blk in locs:
            miss = missing_stat_keys(blk)
            tag = "OK" if not miss else f"MISSING {miss}"
            print(f"    [{tag}]  {label}")
            if keys_only:
                print(f"        keys: {sorted(blk.keys())}")
            else:
                print(f"        {fmt_stats(blk)}")

        if scalar_lister and not keys_only:
            try:
                scalars = scalar_lister(ep_data)
            except Exception as e:
                print(f"    ERROR listing scalars: {e}")
                scalars = []
            if scalars:
                print("    [scalars]")
                for label, value in scalars:
                    if isinstance(value, float):
                        rendered = f"{value:.4g}"
                    else:
                        rendered = repr(value) if isinstance(value, str) else str(value)
                    print(f"        {label:<35}  {rendered}")


def coverage_check(data: Dict[str, Any]) -> int:
    """Walk every metric × every episode × every location; report missing keys.

    Returns the number of locations that fail the check.
    """
    print(f"\n{'═' * 80}")
    print("  COVERAGE CHECK (every episode × every nested location)")
    print(f"{'═' * 80}\n")

    bad = 0
    for metric_name, ep_dict in data.items():
        locator = METRIC_LOCATORS.get(metric_name)
        if locator is None:
            print(f"  [{metric_name}]  (no locator — skipped)")
            continue
        metric_bad = 0
        metric_total = 0
        for ep_id, ep_data in ep_dict.items():
            try:
                locs = locator(ep_data)
            except Exception as e:
                print(f"  [{metric_name}][{ep_id}]  ERROR: {e}")
                metric_bad += 1
                continue
            for label, blk in locs:
                metric_total += 1
                miss = missing_stat_keys(blk)
                if miss:
                    print(f"  [{metric_name}][{ep_id}].{label}  MISSING {miss}")
                    metric_bad += 1
        bad += metric_bad
        status = "OK" if metric_bad == 0 else f"{metric_bad} location(s) bad"
        print(f"  {metric_name:<25} {metric_total:>4} location(s)   {status}")

    print()
    if bad == 0:
        print(
            f"  Result: ALL OK — every block carries the full {len(STAT_KEYS)}-stat schema"
        )
    else:
        print(f"  Result: {bad} location(s) missing keys")
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pkl_path", type=Path, help="Path to metrics-stats_*.pkl")
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help="Show only this metric (repeatable). Default: show all.",
    )
    parser.add_argument(
        "--episode",
        default=None,
        help="Show only episodes whose id contains this substring.",
    )
    parser.add_argument(
        "--keys-only",
        action="store_true",
        help="Print the keys present at each stat-block location instead of values.",
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Skip the coverage check at the end.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Just list metrics + episode counts and exit.",
    )
    args = parser.parse_args()

    if not args.pkl_path.exists():
        print(f"error: {args.pkl_path} does not exist", file=sys.stderr)
        return 2

    with args.pkl_path.open("rb") as f:
        data = pickle.load(f)

    print(f"PKL: {args.pkl_path}  ({args.pkl_path.stat().st_size / 1e6:.2f} MB)")
    list_top_level(data)
    if args.list:
        return 0

    metrics_to_show = args.metric or list(data.keys())
    for metric_name in metrics_to_show:
        if metric_name not in data:
            print(f"warning: {metric_name!r} not in pickle — skipping")
            continue
        print_metric(metric_name, data[metric_name], args.episode, args.keys_only)

    if not args.no_coverage:
        bad = coverage_check(data)
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
