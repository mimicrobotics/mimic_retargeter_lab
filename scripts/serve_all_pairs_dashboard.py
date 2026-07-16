#!/usr/bin/env python3
"""Serve the side-by-side dashboard comparing *every* (hand, retargeter) pair.

Auto-discovers every ``metrics-stats_<dataset>_<hand>_<retargeter>.pkl``
matching ``--dataset`` in ``--reports-dir`` and groups them by hand. The
**Summary** tab aggregates across all discovered hands; the **Hand** tab
provides a per-hand drilldown with a dropdown that switches between any
discovered hand at runtime.

To deep-dive a *single* (hand, retargeter) pair instead, use
``scripts/serve_single_pair_dashboard.py``.

Usage
-----
    # Default: all hands for the dataset
    python scripts/serve_all_pairs_dashboard.py --dataset manus

    # Limit the comparison to specific retargeters
    python scripts/serve_all_pairs_dashboard.py --dataset manus \
        --retargeter dexpilot --retargeter keyvector --retargeter hybrid

    # Only aggregate over a subset of hands
    python scripts/serve_all_pairs_dashboard.py --dataset manus \
        --include-hand mimic_p050_hand --include-hand shadow_hand

    # Show every Summary row (the default hides Keyvector angle / length
    # error because they're noisy and redundant for the side-by-side view)
    python scripts/serve_all_pairs_dashboard.py --dataset manus --show-all-metrics

    # List which Summary rows are available (and which are hidden by
    # default), then exit
    python scripts/serve_all_pairs_dashboard.py --list-metrics

Run ``scripts/run_all_metrics.sh`` first to populate the pkl files.
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import dash_bootstrap_components as dbc
from dash import Dash
from flask import abort, send_from_directory

from dexworld.dashboard.comparison_dashboard import ComparisonDashboard


REPO_ROOT = Path(__file__).resolve().parent.parent
MEDIA_DIR = REPO_ROOT / "media"


# Match longer (multi-token) retargeter names first so e.g. "sampling_based"
# is preferred over a naive rsplit that would yield "based".
_KNOWN_RETARGETERS = (
    "sampling_based",
    "joint_angle",
    "dexpilot",
    "keyvector",
    "hybrid",
    "geort",
    "ako",
)


def _split_hand_retargeter(rest: str) -> tuple[str, str] | None:
    """Split ``<hand>_<retargeter>`` into ``(hand, retargeter)``.

    Tries the known retargeter list first (longest-suffix match), then
    falls back to the last underscore-segment.
    """
    for ret in _KNOWN_RETARGETERS:
        if rest.endswith(f"_{ret}"):
            return rest[: -(len(ret) + 1)], ret
    if "_" not in rest:
        return None
    hand, ret = rest.rsplit("_", 1)
    return hand, ret


def _load_per_hand_per_retargeter(
    reports_dir: Path,
    dataset: str,
    retargeters_filter: List[str] | None,
    hands_filter: List[str] | None,
) -> Dict[str, Dict[str, dict]]:
    """Load every matching pkl into ``{hand: {retargeter: metrics_stats}}``.

    Filename convention: ``metrics-stats_<dataset>_<hand>_<retargeter>.pkl``.
    """
    prefix = f"metrics-stats_{dataset}_"
    out: Dict[str, Dict[str, dict]] = {}
    for path in sorted(reports_dir.glob(f"{prefix}*.pkl")):
        rest = path.stem[len(prefix) :]
        split = _split_hand_retargeter(rest)
        if split is None:
            continue
        hand, ret = split
        if retargeters_filter and ret not in retargeters_filter:
            continue
        if hands_filter and hand not in hands_filter:
            continue
        with path.open("rb") as f:
            out.setdefault(hand, {})[ret] = pickle.load(f)
    return out


def _print_metric_keys() -> None:
    """Print every Summary stat_key with its group, label, and whether
    it's hidden by default. Pass ``--show-all-metrics`` to override the
    defaults."""
    rows = ComparisonDashboard.available_metric_keys()
    default_hidden = ComparisonDashboard._DEFAULT_HIDDEN_METRICS
    print(
        f"Summary metrics ({len(rows)} rows total, {len(default_hidden)} hidden by default):\n"
    )
    print(f"  {'group':<13} {'stat_key':<22} {'default':<8} label")
    print(f"  {'-' * 13} {'-' * 22} {'-' * 8} {'-' * 40}")
    for group, stat_key, label in rows:
        marker = "hidden" if stat_key in default_hidden else "shown"
        print(f"  {group:<13} {stat_key:<22} {marker:<8} {label}")
    print()
    print("Pass --show-all-metrics to reveal the default-hidden rows.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--list-metrics",
        action="store_true",
        help="Print the list of stat_keys available for --hide-metric and exit.",
    )
    p.add_argument("--dataset", help="Dataset name, e.g. manus")
    p.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory with metrics-stats_*.pkl files (default: ./reports)",
    )
    p.add_argument(
        "--retargeter",
        action="append",
        default=None,
        help="Limit to this retargeter (repeatable). Default: every retargeter found.",
    )
    p.add_argument(
        "--include-hand",
        action="append",
        default=None,
        help="Include this hand in the Summary aggregate (repeatable). "
        "Default: every hand found in the reports dir for the dataset.",
    )
    p.add_argument(
        "--show-all-metrics",
        action="store_true",
        help="Show every Summary row, overriding the default-hidden set "
        f"({sorted(ComparisonDashboard._DEFAULT_HIDDEN_METRICS)}). Detail "
        "tabs are unaffected. Run with --list-metrics to see the full list.",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    if args.list_metrics:
        _print_metric_keys()
        return 0

    # --dataset is required for the actual serving path; only made
    # optional above so --list-metrics works without it.
    if args.dataset is None:
        p.error("the following arguments are required: --dataset")

    # ``hidden_metrics=None`` ⇒ dashboard applies its default-hidden set
    # (currently kv_angle_err / kv_len_err — noisy in side-by-side view).
    # ``--show-all-metrics`` ⇒ pass an empty set, which overrides the
    # default and forces every row to render.
    hidden_metrics_arg: set[str] | None = set() if args.show_all_metrics else None

    hands_filter = list(args.include_hand) if args.include_hand else None

    per_hand_per_retargeter = _load_per_hand_per_retargeter(
        args.reports_dir, args.dataset, args.retargeter, hands_filter
    )
    if not per_hand_per_retargeter:
        print(
            f"error: no metrics-stats_{args.dataset}_*.pkl files found in "
            f"{args.reports_dir}. Run scripts/run_all_metrics.sh first.",
            file=sys.stderr,
        )
        return 2

    print(
        f"Loaded {len(per_hand_per_retargeter)} hand(s) × variable retargeters for "
        f"dataset={args.dataset}:"
    )
    for hand, ret_dict in sorted(per_hand_per_retargeter.items()):
        print(f"  {hand}: {sorted(ret_dict.keys())}")
    if args.show_all_metrics:
        print(
            "Showing all Summary rows (--show-all-metrics): default-hidden "
            f"{sorted(ComparisonDashboard._DEFAULT_HIDDEN_METRICS)} are visible."
        )
    else:
        print(
            "Hiding Summary rows by default: "
            f"{sorted(ComparisonDashboard._DEFAULT_HIDDEN_METRICS)}. "
            "Pass --show-all-metrics to display them."
        )

    app = Dash(
        "RetargetBench Comparison",
        external_stylesheets=[dbc.themes.BOOTSTRAP],
    )

    @app.server.route("/media/<path:filename>")
    def _serve_media(filename: str):
        target = (MEDIA_DIR / filename).resolve()
        if not target.is_file() or MEDIA_DIR not in target.parents:
            abort(404)
        return send_from_directory(MEDIA_DIR, filename)

    # primary_hand is left unset — the dashboard defaults to the first
    # hand (alphabetically). The Hand-tab dropdown lets the user switch
    # to any hand at runtime, so pinning a specific one via CLI is no
    # longer needed.
    app.layout = ComparisonDashboard(
        per_hand_per_retargeter=per_hand_per_retargeter,
        dataset=args.dataset,
        hidden_metrics=hidden_metrics_arg,
    ).build(app)

    print(f"Serving on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=False,
        dev_tools_hot_reload=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
