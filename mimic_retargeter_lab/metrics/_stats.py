"""Shared summary-statistics helper for per-frame metric arrays.

Every metric that produces a per-frame (or per-sample) array — Collision,
Flatness, Keyvector Matching, Latency, Motion Preservation, Workspace —
emits the same 12-stat block via :func:`summarize_array` so downstream
consumers (``scripts/summarize_metrics.py``) can read any of them through
a uniform schema without re-running ``compute_hand_retargeter_pair_metrics``.
"""

from typing import Dict

import numpy as np


STAT_KEYS = (
    "n",
    "mean",
    "median",
    "std",
    "min",
    "max",
    "p1",
    "p5",
    "p25",
    "p75",
    "p95",
    "p99",
)


def _empty_summary() -> Dict[str, float]:
    nan = float("nan")
    out = {k: nan for k in STAT_KEYS}
    out["n"] = 0
    return out


def summarize_array(arr) -> Dict[str, float]:
    """Return a 12-stat summary over ``arr``.

    Non-finite values (NaN/inf) are dropped before computing stats. An empty
    or all-non-finite input yields ``n=0`` with every other stat set to NaN.
    Sample standard deviation (``ddof=1``) is used to match
    ``statistics.stdev`` semantics already in place for Latency.
    """
    arr = np.asarray(arr, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return _empty_summary()
    return {
        "n": n,
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if n > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }
