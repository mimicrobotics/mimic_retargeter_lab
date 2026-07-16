#!/usr/bin/env python3
"""Summarize retargeting metrics from PKL report files.

Reads all metrics-stats_*.pkl files from the reports directory and prints
comparison tables grouped by retargeter for:
  - Keyvector Matching (cosine similarity, scale ratio, angle error, length error)
  - Pinch Grasps (cosine similarity, scale ratio, angle error, length error)
  - Motion Preservation (directional alignment)
  - Flatness (mean squared acceleration)
  - Workspace (utilization)
  - Collision (rate, penetration depth, unique colliding pairs)
  - Latency (per-call retargeter latency: device, mean, median, stddev, p99)

Usage:
    python scripts/summarize_metrics.py [--reports-dir ./reports]
"""

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from dexworld.types.metrics import Metric, metric_spec


KNOWN_DATASETS = ("wilor", "manus")
KNOWN_RETARGETERS = (
    # Match longer (multi-token) suffixes first so e.g. "sampling_based"
    # is preferred over a naive rsplit that would yield "based".
    "sampling_based",
    "joint_angle",
    "dexpilot",
    "keyvector",
    "ako",
)


def load_all_reports(reports_dir: Path) -> dict:
    """Load all PKL files and group by dataset -> retargeter -> hand.

    Filename format: ``metrics-stats_<dataset>_<hand>_<retargeter>.pkl``.
    Legacy two-segment filenames (``metrics-stats_<hand>_<retargeter>.pkl``)
    are interpreted as wilor for backwards compatibility.
    """
    datasets: dict[str, dict[str, dict[str, dict]]] = {}
    for f in sorted(reports_dir.glob("metrics-stats_*.pkl")):
        with open(f, "rb") as fh:
            data = pickle.load(fh)
        label = f.stem.replace("metrics-stats_", "")
        ret = next((r for r in KNOWN_RETARGETERS if label.endswith(f"_{r}")), None)
        if ret is None:
            # Unknown retargeter; fall back to splitting at the last underscore.
            parts = label.rsplit("_", 1)
            if len(parts) != 2:
                continue
            rest, ret = parts
        else:
            rest = label[: -(len(ret) + 1)]
        ds = next((d for d in KNOWN_DATASETS if rest.startswith(f"{d}_")), "wilor")
        hand = rest[len(ds) + 1 :] if rest.startswith(f"{ds}_") else rest
        datasets.setdefault(ds, {}).setdefault(ret, {})[hand] = data
    return datasets


def _get_pos(lm_transforms, lm):
    T = lm_transforms[lm]
    if T.ndim == 3:
        return T[:, :3, 3]
    return T[:3, 3][None, :]


# ── Table helpers ──────────────────────────────────────────────────────────


def print_header(title: str, width: int = 100):
    print(f"\n╔{'═' * width}╗")
    print(f"║  {title:^{width - 4}}  ║")
    print(f"╚{'═' * width}╝\n")


def print_retargeter_header(name: str, width: int = 100):
    print(f"  ┌{'─' * (width - 4)}┐")
    print(f"  │ {name.upper() + ' RETARGETER':^{width - 6}} │")
    print(f"  └{'─' * (width - 4)}┘")


# ── Metric extraction ─────────────────────────────────────────────────────


FINGER_ORDER_4 = [
    "thumb_to_index_tip",
    "thumb_to_middle_tip",
    "thumb_to_ring_tip",
    "thumb_to_little_tip",
]
FINGER_LABELS_4 = ["idx", "mid", "ring", "lit"]

FINGER_ORDER_5 = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]
FINGER_LABELS_5 = ["thm", "idx", "mid", "ring", "lit"]


# Map a summarize-table column label to its KV registry entry. The Pinch
# Grasps pkl reuses these same display strings as literal column keys
# (different schema, same human-facing names) — keep this mapping local
# since the Pinch side has no ``MetricSpec`` yet to drive it.
_KV_COLUMN_TO_METRIC = {
    "Cosine Similarity": Metric.COSINE_SIMILARITY,
    "Angle Error [deg]": Metric.ANGLE_ERROR_DEG,
    "Length Error [mm]": Metric.LENGTH_ERROR_MM,
    "Scale Ratio [robot/human]": Metric.SCALE_RATIO,
}


def kv_table_label(key: str) -> str:
    """Column label for a Keyvector Matching sub-table — defers to the
    registry's :attr:`MetricSpec.summary_label` so a rename in
    :data:`dexworld.types.metrics.METRICS` propagates here automatically.
    """
    m = _KV_COLUMN_TO_METRIC.get(key)
    return metric_spec(m).summary_label if m is not None else key


def extract_reference_pose_metric(data, metric_name, key):
    """Extract a per-keyvector scalar from Keyvector Matching or Pinch Grasps.

    Dispatches on the metric:
      - "Keyvector Matching" — time-series schema:
          ``vector_metrics[name][<short_key>]``. The (pkl_subkey, stat)
          pair comes from :func:`dexworld.types.metrics.metric_spec`'s
          ``kv_detail`` so it stays in sync with the dashboard.
          Median is computed from the raw per-frame array; mean is read
          from the precomputed summary field.
      - "Pinch Grasps" (or any other static-pose metric) — legacy schema:
          ``reference_pose_metrics.error_metrics[name][<column_label>]``.
          Each pose is a single scalar; ``stat`` doesn't apply. The
          column-label string is the data contract with that benchmark
          and is passed through verbatim.
    """
    if metric_name not in data:
        return {}
    metric = data[metric_name]

    if metric_name == "Keyvector Matching":
        m = _KV_COLUMN_TO_METRIC.get(key)
        if m is None:
            return {}
        kv = metric_spec(m).kv_detail
        assert kv is not None, "kv metric must carry kv_detail"
        ts_key, stat = kv.pkl_subkey, kv.pkl_stat
        values = {}
        for ep_id, ep_data in metric.items():
            vm = ep_data.get("vector_metrics", {}) or {}
            for dn, dd in vm.items():
                entry = dd.get(ts_key, {}) or {}
                if stat == "median":
                    raw = entry.get("raw")
                    values[dn] = (
                        float(np.nanmedian(raw))
                        if raw is not None and len(raw) > 0
                        else float("nan")
                    )
                else:
                    values[dn] = entry.get("mean", float("nan"))
        return values

    values = {}
    for ep_id, ep_data in metric.items():
        em = ep_data.get("reference_pose_metrics", {}).get("error_metrics", {})
        for dn, dd in em.items():
            values[dn] = dd.get(key, float("nan"))
    return values


def extract_motion_preservation(data):
    """Extract mean alignment per finger."""
    if "Motion Preservation" not in data:
        return {}
    metric = data["Motion Preservation"]
    means = {}
    for ep_id, ep_data in metric.items():
        for fn, fd in ep_data.items():
            pos = fd.get("pos_alignment", {})
            means[fn] = pos.get("mean", float("nan"))
    return means


def extract_flatness(data):
    """Extract mean squared acceleration per finger for human and robot."""
    if "Flatness" not in data:
        return {}, {}
    metric = data["Flatness"]
    human_means, robot_means = {}, {}
    for ep_id, ep_data in metric.items():
        for fn, fd in ep_data.items():
            h = fd.get("human", {})
            r = fd.get("robot", {})
            human_means[fn] = h.get("mean", float("nan"))
            robot_means[fn] = r.get("mean", float("nan"))
    return human_means, robot_means


def extract_collision(data):
    """Extract per-episode collision aggregates as parallel lists.

    Returns a dict of stat -> list[float], one entry per episode that had
    that stat populated. The caller decides how to aggregate across episodes
    (mean for rates/depths, max for the worst-case penetration, sum for
    counts).
    """
    if "Collision" not in data:
        return {}
    metric = data["Collision"]
    keys = (
        "collision_rate_pct",
        "max_penetration_depth_mm",
        "avg_penetration_depth_mm",
        "num_unique_colliding_pairs",
        "frames_with_collision",
        "num_frames",
    )
    out = {k: [] for k in keys}
    for ep_data in metric.values():
        for k in keys:
            v = ep_data.get(k)
            if v is None:
                continue
            out[k].append(v)
    return out


def extract_latency(data):
    """Aggregate latency across all episodes for a single (hand, retargeter).

    Concatenates per-episode ``latencies_ms`` and recomputes summary stats so
    multi-episode runs aggregate correctly. Returns ``{}`` if Latency is
    absent or no timed frames exist.
    """
    if "Latency" not in data:
        return {}
    metric = data["Latency"]
    all_lats: list[float] = []
    devices = set()
    for ep_data in metric.values():
        all_lats.extend(ep_data.get("latencies_ms", []))
        devices.add(ep_data.get("device", "unknown"))
    if not all_lats:
        return {}
    sorted_l = sorted(all_lats)
    n = len(all_lats)
    p99_idx = max(0, int(round(0.99 * n)) - 1)
    return {
        "device": next(iter(devices)) if len(devices) == 1 else "mixed",
        "mean_ms": float(np.mean(all_lats)),
        "median_ms": float(np.median(all_lats)),
        "stdev_ms": float(np.std(all_lats, ddof=1)) if n > 1 else 0.0,
        "p99_ms": float(sorted_l[p99_idx]),
    }


def extract_workspace(data):
    """Extract chamfer distances and utilization per finger."""
    if "Workspace" not in data:
        return {}, {}
    ws = data["Workspace"]
    chamfer = {}
    utilization = {}
    for ep_id, ep_data in ws.items():
        cd = ep_data.get("chamfer_distances", {})
        chamfer.update(cd)
        util = ep_data.get("utilization", {})
        for finger, util_data in util.items():
            utilization[finger] = util_data.get("utilization", float("nan"))
    return chamfer, utilization


# ── Print tables ───────────────────────────────────────────────────────────


def fmt(v, width=8, decimals=3):
    if isinstance(v, float) and np.isnan(v):
        return f"{'—':>{width}}"
    return f"{v:>{width}.{decimals}f}"


def fmt_pct(v, width=7):
    if isinstance(v, float) and (np.isnan(v) or v == 0):
        return f"{'—':>{width}}"
    return f"{v:>{width}.0%}"


def _print_ref_pose_subtable(
    retargeters, metric_name, all_hands, key, label, fmt_fn=None
):
    """Print one sub-table for a reference pose metric key."""
    if fmt_fn is None:
        fmt_fn = fmt
    for ret_name, hands in retargeters.items():
        print(f"  {label:<30} ", end="")
        for fl in FINGER_LABELS_4:
            print(f"{fl:>10}", end="")
        print(f" │ {'AVG':>10}")
        print(f"  {'-' * 30} " + f"{'-' * 10} " * 4 + f"┼ {'-' * 10}")

        ret_all = []
        for hand in all_hands:
            if hand not in hands:
                continue
            vals_dict = extract_reference_pose_metric(hands[hand], metric_name, key)
            vals = [vals_dict.get(d, float("nan")) for d in FINGER_ORDER_4]
            valid = [v for v in vals if not np.isnan(v)]
            avg = np.mean(valid) if valid else float("nan")
            ret_all.extend(valid)
            print(f"  {hand:<30} ", end="")
            for v in vals:
                print(fmt_fn(v, width=10), end="")
            print(f" │ {fmt_fn(avg, width=10)}")

        overall = np.mean(ret_all) if ret_all else float("nan")
        print(f"  {'':<30} " + f"{'':>10} " * 4 + f"│ {'':>10}")
        print(
            f"  {'OVERALL':>30} " + f"{'':>10} " * 4 + f"│ {fmt_fn(overall, width=10)}"
        )
        print()


def print_ref_pose_table(retargeters, metric_name, all_hands):
    """Print Keyvector Matching or Pinch Grasps tables."""
    is_kv = metric_name == "Keyvector Matching"
    keys = (
        "Cosine Similarity",
        "Scale Ratio [robot/human]",
        "Angle Error [deg]",
        "Length Error [mm]",
    )
    for ret_name, hands in retargeters.items():
        print_retargeter_header(ret_name)
        for key in keys:
            label = kv_table_label(key) if is_kv else key
            _print_ref_pose_subtable(
                {ret_name: hands},
                metric_name,
                all_hands,
                key=key,
                label=label,
            )


def print_motion_preservation_table(retargeters, all_hands):
    for ret_name, hands in retargeters.items():
        print_retargeter_header(ret_name)
        print(f"  {'Hand':<30} ", end="")
        for fl in FINGER_LABELS_5:
            print(f"{fl:>8}", end="")
        print(f" │ {'AVG':>8}")
        print(f"  {'-' * 30} " + f"{'-' * 8} " * 5 + f"┼ {'-' * 8}")

        ret_all_means = []
        for hand in all_hands:
            if hand not in hands:
                continue
            means = extract_motion_preservation(hands[hand])
            vals = [means.get(d, float("nan")) for d in FINGER_ORDER_5]
            valid = [v for v in vals if not np.isnan(v)]
            avg = np.mean(valid) if valid else float("nan")
            ret_all_means.extend(valid)
            print(f"  {hand:<30} ", end="")
            for v in vals:
                print(fmt(v), end="")
            print(f" │ {fmt(avg)}")

        overall_mean = np.mean(ret_all_means) if ret_all_means else float("nan")
        print(f"  {'':<30} " + f"{'':>8} " * 5 + f"│ {'':>8}")
        print(f"  {'OVERALL':>30} " + f"{'':>8} " * 5 + f"│ {fmt(overall_mean)}")
        print()


def print_flatness_table(retargeters, all_hands):
    for ret_name, hands in retargeters.items():
        print_retargeter_header(ret_name)
        print(f"  {'Robot (mean ||accel||²)':<30} ", end="")
        for fl in FINGER_LABELS_5:
            print(f"{fl:>10}", end="")
        print(f" │ {'AVG':>10}")
        print(f"  {'-' * 30} " + f"{'-' * 10} " * 5 + f"┼ {'-' * 10}")

        ret_all = []
        for hand in all_hands:
            if hand not in hands:
                continue
            _, robot_means = extract_flatness(hands[hand])
            vals = [robot_means.get(d, float("nan")) for d in FINGER_ORDER_5]
            valid = [v for v in vals if not np.isnan(v)]
            avg = np.mean(valid) if valid else float("nan")
            ret_all.extend(valid)
            print(f"  {hand:<30} ", end="")
            for v in vals:
                print(f"{v:>10.2e}" if not np.isnan(v) else f"{'—':>10}", end="")
            print(f" │ {avg:>10.2e}" if not np.isnan(avg) else f" │ {'—':>10}")

        overall = np.mean(ret_all) if ret_all else float("nan")
        print(f"  {'':<30} " + f"{'':>10} " * 5 + f"│ {'':>10}")
        print(f"  {'OVERALL':>30} " + f"{'':>10} " * 5 + f"│ {overall:>10.2e}")

        # Human reference
        print()
        print(f"  {'Human (mean ||accel||²)':<30} ", end="")
        for fl in FINGER_LABELS_5:
            print(f"{fl:>10}", end="")
        print(f" │ {'AVG':>10}")
        print(f"  {'-' * 30} " + f"{'-' * 10} " * 5 + f"┼ {'-' * 10}")

        for hand in all_hands:
            if hand not in hands:
                continue
            human_means, _ = extract_flatness(hands[hand])
            vals = [human_means.get(d, float("nan")) for d in FINGER_ORDER_5]
            valid = [v for v in vals if not np.isnan(v)]
            avg = np.mean(valid) if valid else float("nan")
            print(f"  {hand:<30} ", end="")
            for v in vals:
                print(f"{v:>10.2e}" if not np.isnan(v) else f"{'—':>10}", end="")
            print(f" │ {avg:>10.2e}" if not np.isnan(avg) else f" │ {'—':>10}")

        print()


def print_workspace_table(retargeters, all_hands):
    for ret_name, hands in retargeters.items():
        print_retargeter_header(ret_name)

        # Workspace Utilization (KP_R coverage)
        has_utilization = False
        for hand in all_hands:
            if hand in hands:
                _, util = extract_workspace(hands[hand])
                if util:
                    has_utilization = True
                    break

        if has_utilization:
            print(f"  {'Utilization (% of KP_R)':<30} ", end="")
            for fl in FINGER_LABELS_5:
                print(f"{fl:>10}", end="")
            print(f" │ {'AVG':>10}")
            print(f"  {'-' * 30} " + f"{'-' * 10} " * 5 + f"┼ {'-' * 10}")

            ret_all_util = []
            for hand in all_hands:
                if hand not in hands:
                    continue
                _, util = extract_workspace(hands[hand])
                vals = [util.get(d, float("nan")) for d in FINGER_ORDER_5]
                valid = [v for v in vals if not np.isnan(v)]
                avg = np.mean(valid) if valid else float("nan")
                ret_all_util.extend(valid)
                print(f"  {hand:<30} ", end="")
                for v in vals:
                    print(fmt_pct(v, width=10), end="")
                print(f" │ {fmt_pct(avg, width=10)}")

            overall_util = np.mean(ret_all_util) if ret_all_util else float("nan")
            print(f"  {'':<30} " + f"{'':>10} " * 5 + f"│ {'':>10}")
            print(
                f"  {'OVERALL':>30} "
                + f"{'':>10} " * 5
                + f"│ {fmt_pct(overall_util, width=10)}"
            )
            print()

        print()


COLLISION_COLS = (
    # (label, dict_key, aggregator_for_OVERALL_row, decimals)
    ("Coll Rate %", "collision_rate_pct", "mean", 2),
    ("Max Pen mm", "max_penetration_depth_mm", "max", 2),
    ("Avg Pen mm", "avg_penetration_depth_mm", "mean", 2),
    ("Uniq Pairs", "num_unique_colliding_pairs", "mean", 1),
    ("Coll Frames", "frames_with_collision", "sum", 0),
    ("Total Frames", "num_frames", "sum", 0),
)


def _collision_per_hand(stats, key, aggregator):
    vals = stats.get(key, [])
    if not vals:
        return float("nan")
    if aggregator == "max":
        return float(np.max(vals))
    if aggregator == "sum":
        return float(np.sum(vals))
    return float(np.mean(vals))


def print_collision_table(retargeters, all_hands):
    for ret_name, hands in retargeters.items():
        print_retargeter_header(ret_name)
        print(f"  {'Hand':<30} ", end="")
        for label, _, _, _ in COLLISION_COLS:
            print(f"{label:>14}", end="")
        print()
        sep = " ".join("-" * 13 for _ in COLLISION_COLS)
        print(f"  {'-' * 30} {sep}")

        per_hand_vals = {label: [] for label, *_ in COLLISION_COLS}
        for hand in all_hands:
            if hand not in hands:
                continue
            stats = extract_collision(hands[hand])
            if not stats:
                continue
            row_vals = [
                _collision_per_hand(stats, key, agg)
                for _, key, agg, _ in COLLISION_COLS
            ]
            if all(np.isnan(v) for v in row_vals):
                continue
            print(f"  {hand:<30} ", end="")
            for v, (_, _, _, decimals) in zip(row_vals, COLLISION_COLS):
                if np.isnan(v):
                    print(f"{'—':>14}", end="")
                else:
                    print(f"{v:>14.{decimals}f}", end="")
            print()
            for (label, _, _, _), v in zip(COLLISION_COLS, row_vals):
                if not np.isnan(v):
                    per_hand_vals[label].append(v)

        # OVERALL: aggregate per-hand values per column using the same rule.
        print(f"  {'':<30} " + " ".join(" " * 13 for _ in COLLISION_COLS))
        print(f"  {'OVERALL':>30} ", end="")
        for label, _, agg, decimals in COLLISION_COLS:
            vals = per_hand_vals[label]
            if not vals:
                print(f"{'—':>14}", end="")
                continue
            if agg == "max":
                v = max(vals)
            elif agg == "sum":
                v = sum(vals)
            else:
                v = float(np.mean(vals))
            print(f"{v:>14.{decimals}f}", end="")
        print()
        print()


LATENCY_NUM_COLS = (
    ("mean (ms)", "mean_ms", 3),
    ("median (ms)", "median_ms", 3),
    ("stddev (ms)", "stdev_ms", 3),
    ("p99 (ms)", "p99_ms", 3),
)


def print_latency_table(retargeters, all_hands):
    """Per-retargeter latency table; one row per hand with device + numeric stats."""
    for ret_name, hands in retargeters.items():
        print_retargeter_header(ret_name)
        print(f"  {'Hand':<30} {'Device':>10}", end="")
        for label, _, _ in LATENCY_NUM_COLS:
            print(f"{label:>13}", end="")
        print()
        print(f"  {'-' * 30} {'-' * 10}" + f" {'-' * 12}" * len(LATENCY_NUM_COLS))

        any_row = False
        for hand in all_hands:
            if hand not in hands:
                continue
            stats = extract_latency(hands[hand])
            if not stats:
                continue
            any_row = True
            print(f"  {hand:<30} {stats['device']:>10}", end="")
            for _, key, decimals in LATENCY_NUM_COLS:
                print(fmt(stats[key], width=13, decimals=decimals), end="")
            print()
        if not any_row:
            print(f"  {'(no Latency stats)':<30}")
        print()


def print_overall_comparison(retargeters, all_hands):
    """Print a compact cross-retargeter comparison."""
    ret_names = list(retargeters.keys())

    # Column width per retargeter — wide enough that long names like
    # "sampling_based" (14 chars) still get visible padding from neighbors.
    col_w = max(18, max((len(rn) for rn in ret_names), default=0) + 4)

    print(f"  {'Metric':<35}", end="")
    for rn in ret_names:
        print(f"{rn:>{col_w}}", end="")
    print()
    print(f"  {'-' * 35}" + f" {'-' * (col_w - 1)}" * len(ret_names))

    # Keyvector Matching — stat per column matches `_KV_TIME_SERIES_KEY_MAP`.
    for key, label in [
        ("Cosine Similarity", "Keyvector Matching (cos sim, median)"),
        ("Scale Ratio [robot/human]", "Keyvector Matching (scale ratio, median)"),
        ("Angle Error [deg]", "Keyvector Matching (angle err deg, mean)"),
        ("Length Error [mm]", "Keyvector Matching (len err mm, mean)"),
    ]:
        row = []
        for rn in ret_names:
            vals = []
            for hand, data in retargeters[rn].items():
                m = extract_reference_pose_metric(data, "Keyvector Matching", key)
                vals.extend(v for v in m.values() if not np.isnan(v))
            row.append(np.mean(vals) if vals else float("nan"))
        print(f"  {label:<35}", end="")
        for v in row:
            print(f"{fmt(v, width=col_w)}", end="")
        print()

    # Pinch
    for key, label in [
        ("Cosine Similarity", "Pinch Grasps (cos sim)"),
        ("Scale Ratio [robot/human]", "Pinch Grasps (scale ratio)"),
        ("Angle Error [deg]", "Pinch Grasps (angle err deg)"),
        ("Length Error [mm]", "Pinch Grasps (len err mm)"),
    ]:
        row = []
        for rn in ret_names:
            vals = []
            for hand, data in retargeters[rn].items():
                m = extract_reference_pose_metric(data, "Pinch Grasps", key)
                vals.extend(v for v in m.values() if not np.isnan(v))
            row.append(np.mean(vals) if vals else float("nan"))
        print(f"  {label:<35}", end="")
        for v in row:
            print(f"{fmt(v, width=col_w)}", end="")
        print()

    # Motion Preservation
    row_mean = []
    for rn in ret_names:
        means_all = []
        for hand, data in retargeters[rn].items():
            means = extract_motion_preservation(data)
            means_all.extend(v for v in means.values() if not np.isnan(v))
        row_mean.append(np.mean(means_all) if means_all else float("nan"))
    print(f"  {'Motion Pres. (alignment)':<35}", end="")
    for v in row_mean:
        print(f"{fmt(v, width=col_w)}", end="")
    print()

    # Flatness
    row_flat = []
    for rn in ret_names:
        vals = []
        for hand, data in retargeters[rn].items():
            _, robot_means = extract_flatness(data)
            vals.extend(v for v in robot_means.values() if not np.isnan(v))
        row_flat.append(np.mean(vals) if vals else float("nan"))
    if any(not np.isnan(v) for v in row_flat):
        print(f"  {'Flatness (robot ||accel||²)':<35}", end="")
        for v in row_flat:
            print(
                f"{v:>{col_w}.2e}" if not np.isnan(v) else f"{'—':>{col_w}}",
                end="",
            )
        print()

    # Workspace (utilization only — chamfer no longer reported)
    row_util = []
    for rn in ret_names:
        utils = []
        for hand, data in retargeters[rn].items():
            _, utilization = extract_workspace(data)
            utils.extend(v for v in utilization.values() if not np.isnan(v))
        row_util.append(np.mean(utils) if utils else float("nan"))
    if any(not np.isnan(v) for v in row_util):
        print(f"  {'Workspace (utilization)':<35}", end="")
        for v in row_util:
            print(f"{fmt_pct(v, width=col_w)}", end="")
        print()

    # Collision: aggregate as mean-of-hand-means across all episodes per retargeter
    coll_rows = [
        ("Collision (rate %)", "collision_rate_pct", "mean", 2),
        ("Collision (max pen mm)", "max_penetration_depth_mm", "max", 2),
        ("Collision (avg pen mm)", "avg_penetration_depth_mm", "mean", 2),
        ("Collision (uniq pairs)", "num_unique_colliding_pairs", "mean", 1),
    ]
    for label, key, agg, decimals in coll_rows:
        row = []
        any_data = False
        for rn in ret_names:
            per_hand = []
            for hand, data in retargeters[rn].items():
                stats = extract_collision(data)
                v = _collision_per_hand(stats, key, agg)
                if not np.isnan(v):
                    per_hand.append(v)
            if not per_hand:
                row.append(float("nan"))
                continue
            any_data = True
            if agg == "max":
                row.append(max(per_hand))
            elif agg == "sum":
                row.append(sum(per_hand))
            else:
                row.append(float(np.mean(per_hand)))
        if any_data:
            print(f"  {label:<35}", end="")
            for v in row:
                if np.isnan(v):
                    print(f"{'—':>{col_w}}", end="")
                else:
                    print(f"{v:>{col_w}.{decimals}f}", end="")
            print()

    # Latency: device per retargeter (string), plus numeric stats.
    row_device = []
    row_mean = []
    row_median = []
    row_stdev = []
    row_p99 = []
    for rn in ret_names:
        means, medians, stdevs, p99s, devs = [], [], [], [], set()
        for hand, data in retargeters[rn].items():
            s = extract_latency(data)
            if not s:
                continue
            means.append(s["mean_ms"])
            medians.append(s["median_ms"])
            stdevs.append(s["stdev_ms"])
            p99s.append(s["p99_ms"])
            devs.add(s["device"])
        row_mean.append(float(np.mean(means)) if means else float("nan"))
        row_median.append(float(np.mean(medians)) if medians else float("nan"))
        row_stdev.append(float(np.mean(stdevs)) if stdevs else float("nan"))
        row_p99.append(float(np.mean(p99s)) if p99s else float("nan"))
        if not devs:
            row_device.append("—")
        elif len(devs) == 1:
            row_device.append(next(iter(devs)))
        else:
            row_device.append("mixed")

    if any(not np.isnan(v) for v in row_mean):
        print(f"  {'Latency device':<35}", end="")
        for d in row_device:
            print(f"{d:>{col_w}}", end="")
        print()
        print(f"  {'Latency mean (ms)':<35}", end="")
        for v in row_mean:
            print(f"{fmt(v, width=col_w)}", end="")
        print()
        print(f"  {'Latency median (ms)':<35}", end="")
        for v in row_median:
            print(f"{fmt(v, width=col_w)}", end="")
        print()
        print(f"  {'Latency stddev (ms)':<35}", end="")
        for v in row_stdev:
            print(f"{fmt(v, width=col_w)}", end="")
        print()
        print(f"  {'Latency p99 (ms)':<35}", end="")
        for v in row_p99:
            print(f"{fmt(v, width=col_w)}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory containing metrics-stats_*.pkl files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Save output to this file (e.g. reports/summary.txt). "
        "If omitted, auto-generates a timestamped file in reports-dir.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Only print to stdout, do not save to file.",
    )
    args = parser.parse_args()

    datasets = load_all_reports(args.reports_dir)
    if not datasets:
        print(f"No metrics-stats_*.pkl files found in {args.reports_dir}")
        return

    # Determine output file path
    if args.no_save:
        out_path = None
    elif args.output:
        out_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = args.reports_dir / f"metrics_summary_{timestamp}.txt"

    # Tee stdout to file if saving
    original_stdout = sys.stdout
    file_handle = None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        file_handle = open(out_path, "w")
        sys.stdout = _TeeWriter(original_stdout, file_handle)

    try:
        _run_summary(datasets)
        if out_path:
            # Print to original stdout only (not tee'd) so it doesn't appear in the file
            original_stdout.write(f"\nSummary saved to {out_path}\n")
    finally:
        sys.stdout = original_stdout
        if file_handle:
            file_handle.close()


class _TeeWriter:
    """Write to two streams simultaneously."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _run_summary(datasets):
    # Collect all hands across all datasets in consistent order
    all_hands_set = set()
    for retargeters in datasets.values():
        for hands in retargeters.values():
            all_hands_set.update(hands.keys())
    all_hands = sorted(all_hands_set)

    print(
        f"\nRetargetBench Metrics Summary — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"Found {len(datasets)} datasets: {', '.join(datasets.keys())}")
    print(f"Found {len(all_hands)} hands: {', '.join(all_hands)}")

    for ds_name, retargeters in datasets.items():
        ret_names = list(retargeters.keys())
        print_header(f"DATASET: {ds_name.upper()}", width=100)
        print(f"  {len(ret_names)} retargeter(s): {', '.join(ret_names)}")

        # ── Keyvector Matching ──
        print_header("KEYVECTOR MATCHING")
        print_ref_pose_table(retargeters, "Keyvector Matching", all_hands)

        # ── Pinch Grasps (wilor only — MANO-format static dataset) ──
        if ds_name == "wilor":
            print_header("PINCH GRASPS")
            print_ref_pose_table(retargeters, "Pinch Grasps", all_hands)

        # ── Motion Preservation ──
        print_header("MOTION PRESERVATION")
        print_motion_preservation_table(retargeters, all_hands)

        # ── Flatness ──
        print_header("FLATNESS")
        print_flatness_table(retargeters, all_hands)

        # ── Workspace ──
        print_header("WORKSPACE")
        print_workspace_table(retargeters, all_hands)

        # ── Collision ──
        print_header("COLLISION")
        print_collision_table(retargeters, all_hands)

        # ── Latency ──
        print_header("LATENCY")
        print_latency_table(retargeters, all_hands)

        # ── Overall Comparison ──
        print_header("OVERALL COMPARISON")
        print_overall_comparison(retargeters, all_hands)
        print()


if __name__ == "__main__":
    main()
