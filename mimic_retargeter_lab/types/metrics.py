"""Single source of truth for every metric the dashboards / summary
table surface.

Three orthogonal enums describe *what* a metric is; one frozen
:class:`MetricSpec` per leaf metric carries everything else (display
labels, format spec, ranking direction, headline reducer, hidden flag,
keyvector-only detail-page metadata). Every consumer (the comparison
dashboard, the per-(hand, retargeter) detail pages, the
``summarize_metrics.py`` CLI table) reads from :data:`METRICS` so that
renaming a label or adding a metric is a single-tuple-entry edit.
"""

import enum
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np


class MetricFamily(enum.Enum):
    """Display grouping in the comparison dashboard summary grid."""

    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    EFFICIENCY = "efficiency"
    SAFETY = "safety"


class BenchmarkMetric(enum.Enum):
    """Pkl-level data source — one per top-level key written by
    ``compute_hand_retargeter_pair_metrics.py``. Each :class:`MetricSpec` is fed by the
    extractor for exactly one ``BenchmarkMetric``.
    """

    KEYVECTOR_MATCHING = "keyvector_matching"
    PINCH_GRASPS = "pinch_grasps"
    WORKSPACE = "workspace"
    COLLISION = "collision"
    MOTION_PRESERVATION = "motion_preservation"
    FLATNESS = "flatness"
    LATENCY = "latency"


class Metric(enum.Enum):
    """Leaf-level scalars surfaced in the dashboard / summary table.

    ``Metric.FLATNESS`` and ``BenchmarkMetric.FLATNESS`` (and likewise
    ``MOTION_PRESERVATION``) share their string value because the
    benchmark produces a single headline scalar with the same name.
    The two enums encode different concepts — don't compare across
    them by ``.value``.
    """

    # Keyvector Matching — ordered so the kv detail card groups its
    # ``research`` rows (cos sim, scale ratio) before its ``engineering``
    # rows (angle, length error), keeping the section divider clean.
    COSINE_SIMILARITY = "cosine_similarity"
    SCALE_RATIO = "scale_ratio"
    ANGLE_ERROR_DEG = "angle_error_deg"
    LENGTH_ERROR_MM = "length_error_mm"
    # Motion Preservation
    MOTION_PRESERVATION = "motion_preservation"
    # Flatness
    FLATNESS = "flatness"
    # Workspace
    WORKSPACE_UTILIZATION = "workspace_utilization"
    # Latency
    LATENCY_AVG = "latency_avg"
    LATENCY_P99 = "latency_p99"
    # Collision
    COLLISION_RATE = "collision_rate"
    COLLISION_MAX_PENETRATION = "collision_max_penetration"
    COLLISION_UNIQUE_PAIRS = "collision_unique_pairs"


class Direction(str, enum.Enum):
    """Drives the podium ranking + the ↑/↓/→0/→1 glyph next to row labels."""

    UP = "up"  # higher is better
    DOWN = "down"  # lower is better
    ABS_DOWN = "abs_down"  # closer to 0 is better
    NEAR_ONE = "near_one"  # closer to 1 is better


# ── KV-only detail-page metadata ──────────────────────────────────────────


@dataclass(frozen=True)
class KvDetailFields:
    """Per-keyvector card metadata used only by ``KeyvectorMatchingPage``.

    None of these apply to non-kv metrics — every other detail page uses
    a different layout (per-finger box plots, time-series, episode
    tables) so the fields would be dead weight there.
    """

    pkl_subkey: str
    """Key inside ``vector_metrics[<keyvector>][<pkl_subkey>]``."""

    pkl_stat: Literal["mean", "median"]
    """Which precomputed summary stat to read for the dashboard headline.
    Cosine similarity / scale ratio use median (robust against bimodal
    still-frame noise); angle / length errors use mean."""

    card_label: str
    """Row label printed inside the per-keyvector card on the detail page."""

    unit: str
    """Suffix appended to formatted values inside the card (e.g. ``"°"``,
    ``" mm"``, or empty string)."""

    stats_variant: Literal["percentiles", "percentiles_both_tails", "moments"]
    """Which stats summary the detail page renders. ``percentiles`` =
    median + p75/p25/p5/p1 (bounded distributions like cos sim);
    ``percentiles_both_tails`` adds p95/p99 (deviations both ways
    matter, like scale ratio); ``moments`` = mean/std/min/max
    (additive errors, directly actionable when tuning)."""

    section: Literal["research", "engineering"]
    """Which group the row sits in inside the detail card. The page
    draws a divider when the section flips — research = scale-invariant
    axes, engineering = absolute errors."""

    value_fmt: str | None = None
    """Override ``MetricSpec.fmt`` for this metric's per-value text in
    the kv card. Use to add a sign prefix (e.g. ``"+.2f"`` for length
    error, where the sign communicates over- vs. under-shoot). ``None``
    falls back to ``MetricSpec.fmt`` so the card and the dashboard
    summary read the same number."""


# ── The spec ──────────────────────────────────────────────────────────────


_BENCHMARK_EMOJI: dict[BenchmarkMetric, str] = {
    BenchmarkMetric.KEYVECTOR_MATCHING: "✋",
    BenchmarkMetric.MOTION_PRESERVATION: "➡️",
    BenchmarkMetric.FLATNESS: "🫓",
    BenchmarkMetric.WORKSPACE: "🛠",
    BenchmarkMetric.COLLISION: "💥",
    BenchmarkMetric.LATENCY: "⏱️",
    BenchmarkMetric.PINCH_GRASPS: "🤏",
}


@dataclass(frozen=True)
class MetricSpec:
    """One entry per metric. Edit a field here to change a metric's
    appearance everywhere — the dashboards and the summary CLI all
    read from :data:`METRICS`."""

    metric: Metric
    family: MetricFamily
    benchmark: BenchmarkMetric

    summary_label: str
    """Row label in the comparison dashboard summary grid + chart title."""

    yaxis_label: str
    """Y-axis title in the per-metric bar chart (also used by the
    flatness detail page chart)."""

    direction: Direction

    fmt: str
    """Bare format spec (e.g. ``".4f"``, ``".1%"``). Consumers wrap as
    needed — plotly tickformat uses it directly, ``str.format`` wraps
    as ``f"{{:{fmt}}}"``."""

    reducer: Callable[[Any], float]
    """Maps the benchmark's extractor output to a single scalar
    headline. Signature depends on the benchmark family — see
    :data:`METRICS` for examples."""

    hidden_by_default: bool = False
    kv_detail: KvDetailFields | None = None

    @property
    def emoji(self) -> str:
        return _BENCHMARK_EMOJI[self.benchmark]


# ── Reducer helpers ───────────────────────────────────────────────────────


def _avg(d: dict[str, float]) -> float:
    """Mean of a per-keyvector / per-finger dict, NaN-skipping."""
    vals = [v for v in d.values() if not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


# ── The registry — single source of truth ─────────────────────────────────
#
# Ordered so the comparison dashboard summary grid renders rows in the
# intended sequence (each MetricFamily is a contiguous run).


METRICS: tuple[MetricSpec, ...] = (
    # Accuracy ────────────────────────────────────────────────────────────
    MetricSpec(
        metric=Metric.COSINE_SIMILARITY,
        family=MetricFamily.ACCURACY,
        benchmark=BenchmarkMetric.KEYVECTOR_MATCHING,
        summary_label="Cosine Similarity",
        yaxis_label="Cosine Similarity",
        direction=Direction.NEAR_ONE,
        fmt=".4f",
        reducer=lambda kv: _avg(kv["cosine_similarity"]),
        kv_detail=KvDetailFields(
            pkl_subkey="cosine_similarity",
            pkl_stat="median",
            card_label="cosine_similarity",
            unit="",
            stats_variant="percentiles",
            section="research",
        ),
    ),
    MetricSpec(
        metric=Metric.SCALE_RATIO,
        family=MetricFamily.ACCURACY,
        benchmark=BenchmarkMetric.KEYVECTOR_MATCHING,
        summary_label="Scale Ratio (robot/human)",
        yaxis_label="Scale Ratio (robot/human)",
        direction=Direction.NEAR_ONE,
        fmt=".3f",
        reducer=lambda kv: _avg(kv["scale_ratio"]),
        kv_detail=KvDetailFields(
            pkl_subkey="scale_ratio",
            pkl_stat="median",
            card_label="scale_ratio (robot/human)",
            unit="",
            stats_variant="percentiles_both_tails",
            section="research",
        ),
    ),
    MetricSpec(
        metric=Metric.ANGLE_ERROR_DEG,
        family=MetricFamily.ACCURACY,
        benchmark=BenchmarkMetric.KEYVECTOR_MATCHING,
        summary_label="Angle Error [deg] (mean)",
        yaxis_label="Angle Error (deg)",
        direction=Direction.DOWN,
        fmt=".2f",
        reducer=lambda kv: _avg(kv["angle_error_deg"]),
        hidden_by_default=True,
        kv_detail=KvDetailFields(
            pkl_subkey="angle_error_deg",
            pkl_stat="mean",
            card_label="angle_difference",
            unit="°",
            stats_variant="moments",
            section="engineering",
        ),
    ),
    MetricSpec(
        metric=Metric.LENGTH_ERROR_MM,
        family=MetricFamily.ACCURACY,
        benchmark=BenchmarkMetric.KEYVECTOR_MATCHING,
        summary_label="Length Error [mm] (|mean|)",
        yaxis_label="Length Error (mm)",
        direction=Direction.ABS_DOWN,
        fmt=".2f",
        reducer=lambda kv: _avg(kv["length_error_mm"]),
        hidden_by_default=True,
        kv_detail=KvDetailFields(
            pkl_subkey="length_error_mm",
            pkl_stat="mean",
            card_label="length_difference",
            unit=" mm",
            stats_variant="moments",
            section="engineering",
            # Show sign so the reader sees over- vs. under-shoot at a glance.
            value_fmt="+.2f",
        ),
    ),
    # Consistency ─────────────────────────────────────────────────────────
    MetricSpec(
        metric=Metric.MOTION_PRESERVATION,
        family=MetricFamily.CONSISTENCY,
        benchmark=BenchmarkMetric.MOTION_PRESERVATION,
        summary_label="Motion Preservation Alignment",
        yaxis_label="Alignment Score",
        direction=Direction.NEAR_ONE,
        fmt=".3f",
        reducer=_avg,
    ),
    MetricSpec(
        metric=Metric.FLATNESS,
        family=MetricFamily.CONSISTENCY,
        benchmark=BenchmarkMetric.FLATNESS,
        summary_label="Flatness — Robot ‖accel‖²",
        yaxis_label="‖accel‖²",
        direction=Direction.DOWN,
        fmt=".2e",
        # Extractor returns ``(human_means, robot_means)``; the headline
        # is robot only.
        reducer=lambda flat: _avg(flat[1]),
    ),
    # Efficiency ──────────────────────────────────────────────────────────
    MetricSpec(
        metric=Metric.WORKSPACE_UTILIZATION,
        family=MetricFamily.EFFICIENCY,
        benchmark=BenchmarkMetric.WORKSPACE,
        summary_label="Workspace Utilization",
        yaxis_label="Utilization (%)",
        direction=Direction.UP,
        fmt=".1%",
        # Extractor returns ``(util_per_finger, dist_per_finger)``.
        reducer=lambda ws: _avg(ws[0]),
    ),
    MetricSpec(
        metric=Metric.LATENCY_AVG,
        family=MetricFamily.EFFICIENCY,
        benchmark=BenchmarkMetric.LATENCY,
        summary_label="Latency Mean [ms]",
        yaxis_label="Latency (ms)",
        direction=Direction.DOWN,
        fmt=".3f",
        reducer=lambda lat: lat.get("mean_ms", float("nan")),
    ),
    MetricSpec(
        metric=Metric.LATENCY_P99,
        family=MetricFamily.EFFICIENCY,
        benchmark=BenchmarkMetric.LATENCY,
        summary_label="Latency p99 [ms]",
        yaxis_label="Latency p99 (ms)",
        direction=Direction.DOWN,
        fmt=".3f",
        reducer=lambda lat: lat.get("p99_ms", float("nan")),
        # Computed pre-refactor but never surfaced in the summary grid;
        # exposed here behind --show-all-metrics for parity with the
        # other percentile-based latency stats.
        hidden_by_default=True,
    ),
    # Safety ──────────────────────────────────────────────────────────────
    MetricSpec(
        metric=Metric.COLLISION_RATE,
        family=MetricFamily.SAFETY,
        benchmark=BenchmarkMetric.COLLISION,
        summary_label="Collision Rate %",
        yaxis_label="Collision Rate (%)",
        direction=Direction.DOWN,
        fmt=".2f",
        reducer=lambda c: c.get("collision_rate_pct", float("nan")),
    ),
    MetricSpec(
        metric=Metric.COLLISION_MAX_PENETRATION,
        family=MetricFamily.SAFETY,
        benchmark=BenchmarkMetric.COLLISION,
        summary_label="Collision Max Pen [mm]",
        yaxis_label="Max Penetration (mm)",
        direction=Direction.DOWN,
        fmt=".2f",
        reducer=lambda c: c.get("max_penetration_depth_mm", float("nan")),
    ),
    MetricSpec(
        metric=Metric.COLLISION_UNIQUE_PAIRS,
        family=MetricFamily.SAFETY,
        benchmark=BenchmarkMetric.COLLISION,
        summary_label="Collision Unique Pairs",
        yaxis_label="Unique Pairs (count)",
        direction=Direction.DOWN,
        fmt=".1f",
        reducer=lambda c: c.get("num_unique_colliding_pairs", float("nan")),
    ),
)


# ── Lookup helpers ────────────────────────────────────────────────────────


_BY_METRIC: dict[Metric, MetricSpec] = {s.metric: s for s in METRICS}


def metric_spec(m: Metric) -> MetricSpec:
    return _BY_METRIC[m]


def metrics_in_family(f: MetricFamily) -> tuple[MetricSpec, ...]:
    return tuple(s for s in METRICS if s.family == f)


def metrics_for_benchmark(b: BenchmarkMetric) -> tuple[MetricSpec, ...]:
    return tuple(s for s in METRICS if s.benchmark == b)


def kv_metrics() -> tuple[MetricSpec, ...]:
    return metrics_for_benchmark(BenchmarkMetric.KEYVECTOR_MATCHING)


def families_in_order() -> tuple[MetricFamily, ...]:
    """Families in their first-appearance order in :data:`METRICS`."""
    seen: list[MetricFamily] = []
    for s in METRICS:
        if s.family not in seen:
            seen.append(s.family)
    return tuple(seen)
