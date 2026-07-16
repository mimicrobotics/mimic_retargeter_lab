"""Side-by-side retargeter comparison dashboard.

Built from a ``{retargeter_name: metrics_stats}`` dict — one entry per
``metrics-stats_<dataset>_<hand>_<retargeter>.pkl`` produced by
``compute_hand_retargeter_pair_metrics.py``. The structure mirrors the per-(hand, retargeter)
``Dashboard`` (top-level ``dbc.Tabs``) but each tab compares all
retargeters at once instead of showing one in isolation.

Tabs
----
* **Summary** — cross-hand aggregate. Each cell is the mean of that
  retargeter's per-hand value across every loaded hand, styled with
  podium tints to highlight the top three.
* **Hand** — single-hand drilldown. Same grid layout as Summary, but
  the values come from one hand at a time. A ``dcc.Dropdown`` lets the
  user switch between any hand discovered for the dataset; a Dash
  callback re-renders the grid on change.

Numbers shown follow the same conventions as ``summarize_metrics.py``:
median for Keyvector Matching cosine_similarity / scale_ratio, mean for
the other 12-stat fields, and per-episode aggregates for Collision /
Latency / Workspace.
"""

from typing import Any, Dict, Iterable, List, Literal, Sequence

import dash_bootstrap_components as dbc
import numpy as np
import plotly.colors as pc
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output

from dexworld.types.metrics import (
    METRICS,
    BenchmarkMetric,
    Direction,
    MetricSpec,
    families_in_order,
    kv_metrics,
    metrics_in_family,
)


# ── Stable retargeter colour palette ──────────────────────────────────────

# Categorical Set2 + Set1 covers the 7 retargeters (ako, dexpilot, geort,
# hybrid, joint_angle, keyvector, sampling_based) without recycling.
_PALETTE: Sequence[str] = (
    pc.qualitative.Set2 + pc.qualitative.Set1 + pc.qualitative.Pastel
)

# All retargeter chips use the same near-black text on top of their
# pastel backgrounds — uniform reads better than auto-contrast that
# flipped some chips to white.
_CHIP_TEXT_COLOR = "#111827"  # gray-900
_CHIP_SUB_COLOR = "rgba(17,24,39,0.65)"  # gray-900 @ 65% — for the sub-line


def retargeter_color(name: str, all_names: Sequence[str]) -> str:
    """Stable colour assignment so each retargeter looks the same in every
    chart of the dashboard."""
    return _PALETTE[all_names.index(name) % len(_PALETTE)]


# Distinct from the retargeter palette so a hand chip and a retargeter
# chip never collide visually in the same chart.
_HAND_PALETTE: Sequence[str] = pc.qualitative.D3 + pc.qualitative.Dark24


def hand_color(name: str, all_names: Sequence[str]) -> str:
    """Stable colour for each hand across every per-metric bar chart."""
    return _HAND_PALETTE[all_names.index(name) % len(_HAND_PALETTE)]


# ── Per-metric extractors (per-retargeter aggregation) ────────────────────
#
# Each extractor takes one retargeter's pkl content (the full
# ``metrics_stats`` dict from ``compute_hand_retargeter_pair_metrics.py``) and returns a dict
# in the shape needed by the corresponding bar-chart builder. Aggregation
# across episodes is done by averaging the relevant stat — same convention
# as ``summarize_metrics.py``.


def _extract_keyvector(data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Return ``{pkl_subkey: {keyvector_name: value}}`` averaged across
    episodes for one retargeter.

    Driven by :func:`dexworld.types.metrics.kv_metrics`; the (subkey, stat)
    pairs come from each spec's ``kv_detail`` so adding a new kv metric is
    a one-line edit in the registry.
    """
    pairs = [(s.kv_detail.pkl_subkey, s.kv_detail.pkl_stat) for s in kv_metrics()]
    out: Dict[str, Dict[str, float]] = {sk: {} for sk, _ in pairs}
    metric = data.get("Keyvector Matching") or {}
    if not metric:
        return out

    # Collect per-(sub_metric, keyvector) lists across episodes, then mean.
    accum: Dict[str, Dict[str, list]] = {sk: {} for sk, _ in pairs}
    for ep in metric.values():
        for kv_name, sub_dict in (ep.get("vector_metrics") or {}).items():
            for sub_metric, stat in pairs:
                blk = sub_dict.get(sub_metric, {}) or {}
                v = blk.get(stat)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                accum[sub_metric].setdefault(kv_name, []).append(float(v))
    for sm, kv_to_vals in accum.items():
        for kv_name, vals in kv_to_vals.items():
            out[sm][kv_name] = float(np.mean(vals)) if vals else float("nan")
    return out


def _extract_motion_preservation(data: Dict[str, Any]) -> Dict[str, float]:
    """Return ``{finger: mean alignment across episodes}``."""
    metric = data.get("Motion Preservation") or {}
    accum: Dict[str, list] = {}
    for ep in metric.values():
        for finger, fd in ep.items():
            v = (fd.get("pos_alignment") or {}).get("mean")
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            accum.setdefault(finger, []).append(float(v))
    return {fn: float(np.mean(vs)) for fn, vs in accum.items()}


def _extract_flatness(
    data: Dict[str, Any],
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Return ``(human_means, robot_means)`` keyed by finger."""
    metric = data.get("Flatness") or {}
    h_acc: Dict[str, list] = {}
    r_acc: Dict[str, list] = {}
    for ep in metric.values():
        for finger, fd in ep.items():
            for emb, accum in (("human", h_acc), ("robot", r_acc)):
                v = (fd.get(emb) or {}).get("mean")
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                accum.setdefault(finger, []).append(float(v))
    return (
        {fn: float(np.mean(vs)) for fn, vs in h_acc.items()},
        {fn: float(np.mean(vs)) for fn, vs in r_acc.items()},
    )


def _extract_workspace(
    data: Dict[str, Any],
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Return ``(utilization_per_finger, mean_distance_per_finger_m)``."""
    metric = data.get("Workspace") or {}
    util_acc: Dict[str, list] = {}
    dist_acc: Dict[str, list] = {}
    for ep in metric.values():
        for finger, blk in (ep.get("utilization") or {}).items():
            u = blk.get("utilization")
            if u is not None:
                util_acc.setdefault(finger, []).append(float(u))
            d = (blk.get("distance_stats") or {}).get("mean")
            if d is not None and not np.isnan(d):
                dist_acc.setdefault(finger, []).append(float(d))
    return (
        {fn: float(np.mean(vs)) for fn, vs in util_acc.items()},
        {fn: float(np.mean(vs)) for fn, vs in dist_acc.items()},
    )


def _extract_collision(data: Dict[str, Any]) -> Dict[str, float]:
    """Return ``{stat_label: aggregate_across_episodes}``.

    Aggregation matches ``summarize_metrics.py``: mean for rates / depths,
    max for the worst-case max penetration, sum for frame counts.
    """
    metric = data.get("Collision") or {}
    aggregators = {
        "collision_rate_pct": "mean",
        "max_penetration_depth_mm": "max",
        "avg_penetration_depth_mm": "mean",
        "num_unique_colliding_pairs": "mean",
        "frames_with_collision": "sum",
        "num_frames": "sum",
    }
    out: Dict[str, float] = {}
    for key, agg in aggregators.items():
        vals = [ep[key] for ep in metric.values() if key in ep and ep[key] is not None]
        if not vals:
            out[key] = float("nan")
            continue
        if agg == "max":
            out[key] = float(np.max(vals))
        elif agg == "sum":
            out[key] = float(np.sum(vals))
        else:
            out[key] = float(np.mean(vals))
    return out


def _extract_latency(data: Dict[str, Any]) -> Dict[str, float | str]:
    """Return ``{mean_ms, median_ms, stdev_ms, p99_ms, device}`` aggregated
    across episodes (pooled raw samples for the percentiles, just like
    ``summarize_metrics.extract_latency``)."""
    metric = data.get("Latency") or {}
    pooled: list = []
    devices = set()
    for ep in metric.values():
        pooled.extend(ep.get("latencies_ms", []))
        d = ep.get("device")
        if d is not None:
            devices.add(d)
    if not pooled:
        return {
            "mean_ms": float("nan"),
            "median_ms": float("nan"),
            "stdev_ms": float("nan"),
            "p99_ms": float("nan"),
            "device": "—",
        }
    arr = np.asarray(pooled, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "stdev_ms": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "p99_ms": float(np.percentile(arr, 99)),
        "device": "mixed"
        if len(devices) > 1
        else (next(iter(devices)) if devices else "—"),
    }


# ── Per-episode extractors ────────────────────────────────────────────────
#
# Mirror the cross-episode extractors above but return per-episode
# results in the *same shape* the corresponding ``MetricSpec.reducer``
# expects — so a reducer written for the Summary aggregate can be reused
# verbatim on per-episode data. The Hand tab's line+ribbon charts use
# these to show episode-to-episode variation within a single
# (hand, retargeter).


def _extract_keyvector_per_episode(
    data: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return ``{episode_id: {pkl_subkey: {keyvector_name: value}}}``."""
    pairs = [(s.kv_detail.pkl_subkey, s.kv_detail.pkl_stat) for s in kv_metrics()]
    metric = data.get("Keyvector Matching") or {}
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for ep_id, ep in metric.items():
        ep_out: Dict[str, Dict[str, float]] = {sk: {} for sk, _ in pairs}
        for kv_name, sub_dict in (ep.get("vector_metrics") or {}).items():
            for sub_metric, stat in pairs:
                blk = sub_dict.get(sub_metric, {}) or {}
                v = blk.get(stat)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                ep_out[sub_metric][kv_name] = float(v)
        out[ep_id] = ep_out
    return out


def _extract_motion_preservation_per_episode(
    data: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Return ``{episode_id: {finger: pos_alignment.mean}}``."""
    metric = data.get("Motion Preservation") or {}
    out: Dict[str, Dict[str, float]] = {}
    for ep_id, ep in metric.items():
        ep_out: Dict[str, float] = {}
        for finger, fd in ep.items():
            v = (fd.get("pos_alignment") or {}).get("mean")
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            ep_out[finger] = float(v)
        out[ep_id] = ep_out
    return out


def _extract_flatness_per_episode(
    data: Dict[str, Any],
) -> Dict[str, tuple[Dict[str, float], Dict[str, float]]]:
    """Return ``{episode_id: (human_means, robot_means)}`` keyed by finger."""
    metric = data.get("Flatness") or {}
    out: Dict[str, tuple[Dict[str, float], Dict[str, float]]] = {}
    for ep_id, ep in metric.items():
        h_means: Dict[str, float] = {}
        r_means: Dict[str, float] = {}
        for finger, fd in ep.items():
            for emb, target in (("human", h_means), ("robot", r_means)):
                v = (fd.get(emb) or {}).get("mean")
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                target[finger] = float(v)
        out[ep_id] = (h_means, r_means)
    return out


def _extract_workspace_per_episode(
    data: Dict[str, Any],
) -> Dict[str, tuple[Dict[str, float], Dict[str, float]]]:
    """Return ``{episode_id: (utilization_per_finger, distance_per_finger)}``."""
    metric = data.get("Workspace") or {}
    out: Dict[str, tuple[Dict[str, float], Dict[str, float]]] = {}
    for ep_id, ep in metric.items():
        util: Dict[str, float] = {}
        dist: Dict[str, float] = {}
        for finger, blk in (ep.get("utilization") or {}).items():
            u = blk.get("utilization")
            if u is not None and not (isinstance(u, float) and np.isnan(u)):
                util[finger] = float(u)
            d = (blk.get("distance_stats") or {}).get("mean")
            if d is not None and not (isinstance(d, float) and np.isnan(d)):
                dist[finger] = float(d)
        out[ep_id] = (util, dist)
    return out


def _extract_collision_per_episode(
    data: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Return ``{episode_id: {stat_label: value}}`` — collision stats are
    already per-episode in the pkl; this just slices them out."""
    metric = data.get("Collision") or {}
    keys = (
        "collision_rate_pct",
        "max_penetration_depth_mm",
        "avg_penetration_depth_mm",
        "num_unique_colliding_pairs",
        "frames_with_collision",
        "num_frames",
    )
    out: Dict[str, Dict[str, float]] = {}
    for ep_id, ep in metric.items():
        out[ep_id] = {
            k: float(ep[k])
            for k in keys
            if ep.get(k) is not None
            and not (isinstance(ep.get(k), float) and np.isnan(ep[k]))
        }
    return out


def _extract_latency_per_episode(
    data: Dict[str, Any],
) -> Dict[str, Dict[str, float | str]]:
    """Return ``{episode_id: {mean_ms, p99_ms, device}}`` — percentiles are
    recomputed from each episode's raw latencies, mirroring the pooled
    version in :func:`_extract_latency`."""
    metric = data.get("Latency") or {}
    out: Dict[str, Dict[str, float | str]] = {}
    for ep_id, ep in metric.items():
        lats = ep.get("latencies_ms") or []
        device = ep.get("device", "—")
        if not lats:
            out[ep_id] = {
                "mean_ms": float("nan"),
                "p99_ms": float("nan"),
                "device": device,
            }
            continue
        arr = np.asarray(list(lats), dtype=np.float64)
        out[ep_id] = {
            "mean_ms": float(np.mean(arr)),
            "p99_ms": float(np.percentile(arr, 99)),
            "device": device,
        }
    return out


# ── Tab builders ──────────────────────────────────────────────────────────


class ComparisonDashboard:
    """Build the Dash layout for cross-retargeter comparison.

    Two tabs, one shared grid layout:

      * **Summary** tab — aggregate across *all* hands in
        ``per_hand_per_retargeter``. Each headline number per
        (retargeter, metric) is the mean of that retargeter's per-hand
        value, so a retargeter that wins consistently across hands
        outranks one that wins for a single hand only.
      * **Hand** tab — single-hand drilldown. Same grid layout as
        Summary, but the values come from one hand at a time. A
        ``dcc.Dropdown`` lets the user switch between any of
        ``self.hands``; a Dash callback (registered in :meth:`build`)
        re-renders the grid on change. ``primary_hand`` is the initial
        selection.

    The shared grid renderer is :meth:`_build_grid`; both tabs feed it
    a ``{retargeter: {stat_key: ...}}`` dict produced by either
    :meth:`_aggregate_across_hands` or :meth:`_aggregate_for_one_hand`.
    """

    # Stat keys hidden out-of-the-box in the Summary tab. Derived from
    # :data:`dexworld.types.metrics.METRICS` (anything with
    # ``hidden_by_default=True``), so flipping a metric's visibility is
    # an edit on its ``MetricSpec``, not on this class. Pass
    # ``hidden_metrics=set()`` (or ``--show-all-metrics`` on the CLI) to
    # override.
    _DEFAULT_HIDDEN_METRICS: frozenset[str] = frozenset(
        s.metric.value for s in METRICS if s.hidden_by_default
    )

    def __init__(
        self,
        per_hand_per_retargeter: Dict[str, Dict[str, Dict[str, Any]]],
        dataset: str,
        primary_hand: str | None = None,
        hidden_metrics: Iterable[str] | None = None,
    ):
        if not per_hand_per_retargeter:
            raise ValueError(
                "per_hand_per_retargeter is empty — at least one hand must "
                "be present to render the comparison dashboard."
            )
        self.per_hand_per_retargeter = per_hand_per_retargeter
        self.dataset = dataset
        self.hands: List[str] = sorted(per_hand_per_retargeter.keys())
        # ``primary_hand`` is just the initial selection of the Hand tab's
        # dropdown — the user can switch to any hand at runtime. Default
        # to the first hand alphabetically so callers don't have to pick.
        if primary_hand is None:
            primary_hand = self.hands[0]
        elif primary_hand not in per_hand_per_retargeter:
            raise ValueError(
                f"primary_hand={primary_hand!r} not in per_hand_per_retargeter "
                f"(available: {self.hands})"
            )
        self.primary_hand = primary_hand
        # Union of every retargeter that appears in any hand — drives a
        # stable colour map shared across both Summary and detail views.
        all_rets: List[str] = []
        for hand_data in per_hand_per_retargeter.values():
            for r in hand_data.keys():
                if r not in all_rets:
                    all_rets.append(r)
        self.retargeters: List[str] = sorted(all_rets)
        self.colors: Dict[str, str] = {
            r: retargeter_color(r, self.retargeters) for r in self.retargeters
        }
        # Stat keys to skip in the Summary tab (no-op for the detail tabs).
        # ``None`` (the default) means "use ``_DEFAULT_HIDDEN_METRICS``";
        # an explicit (possibly empty) iterable overrides the default —
        # so passing ``set()`` shows every row.
        if hidden_metrics is None:
            hidden = set(self._DEFAULT_HIDDEN_METRICS)
        else:
            hidden = set(hidden_metrics)
        # Validated up-front so a typo fails fast instead of silently
        # rendering every row.
        known = {sk for _, sk, _ in self.available_metric_keys()}
        unknown = hidden - known
        if unknown:
            raise ValueError(
                f"hidden_metrics contains unknown stat_keys: {sorted(unknown)}. "
                f"Known: {sorted(known)}"
            )
        self.hidden_metrics: set[str] = hidden

    @classmethod
    def available_metric_keys(cls) -> List[tuple[str, str, str]]:
        """Every Summary row as ``(group, stat_key, label)``.

        Used by :func:`scripts.serve_all_pairs_dashboard.main` to print a
        discoverable list at startup. ``stat_key`` is the canonical id
        (matches the keys produced by :meth:`_aggregate_across_hands`);
        ``label`` is the human-readable row title shown in the table.
        Rows whose ``stat_key`` is in :attr:`_DEFAULT_HIDDEN_METRICS` are
        hidden by default unless the caller passes
        ``hidden_metrics=set()``.
        """
        return [
            (s.family.value.title(), s.metric.value, s.summary_label) for s in METRICS
        ]

    # ── Public ────────────────────────────────────────────────────────────

    def build(self, app) -> html.Div:
        n_hands = len(self.hands)
        summary_label = (
            f"📋 Summary (all {n_hands} hands)" if n_hands > 1 else "📋 Summary"
        )

        # Register the dropdown→content callback for the Hand tab. The
        # callback now updates both the grid and the per-metric charts
        # section in a single output container so they re-render
        # together when the user switches hands. Closure over ``self``
        # keeps the registration local and avoids the plumbing required
        # to store ``app`` on the instance. The IDs match the elements
        # built inside ``_build_hand_tab``.
        @app.callback(
            [
                Output("cmp-hand-content", "children"),
                Output("cmp-hand-image", "src"),
            ],
            [Input("cmp-hand-dropdown", "value")],
        )
        def _update_hand_content(hand: str):
            if not hand or hand not in self.per_hand_per_retargeter:
                return (
                    html.Div(
                        "Select a hand from the dropdown.",
                        style={"padding": "12px", "color": "#6b7280"},
                    ),
                    "",
                )
            return self._build_hand_content(hand), f"/media/hands/{hand}.webp"

        tabs = [
            dbc.Tab(
                label=summary_label,
                tab_id="cmp-summary",
                children=self._build_summary_tab(),
            ),
            dbc.Tab(
                label="🤖 Hand",
                tab_id="cmp-hand",
                children=self._build_hand_tab(),
            ),
        ]
        title = (
            f"ManuFold Metrics — {self.dataset} / "
            f"{n_hands} hand(s) / {len(self.retargeters)} retargeter(s)"
        )
        sub = (
            f"Summary aggregates across hands: {', '.join(self.hands)}. "
            f"Hand drills into one hand at a time "
            f"(initial selection: {self.primary_hand.replace('_', ' ')})."
        )
        return html.Div(
            [
                html.H2(title),
                html.Div(
                    sub,
                    style={
                        "color": "#666",
                        "marginBottom": "8px",
                        "fontSize": "0.92em",
                    },
                ),
                html.Div(
                    [
                        html.Span(
                            r,
                            style={
                                "backgroundColor": self.colors[r],
                                "color": _CHIP_TEXT_COLOR,
                                "padding": "4px 10px",
                                "marginRight": "8px",
                                "borderRadius": "12px",
                                "fontWeight": "bold",
                                "display": "inline-block",
                            },
                        )
                        for r in self.retargeters
                    ],
                    style={"marginBottom": "12px"},
                ),
                dbc.Tabs(tabs, id="cmp-tabs", className="top-tabs"),
            ]
        )

    # ── Summary tab ───────────────────────────────────────────────────────

    def _per_hand_headline(self, hand_data: Dict[str, Any]) -> Dict[str, float | str]:
        """Reduce one (hand, retargeter) pkl down to a flat headline-stat dict.

        One scalar per metric, keyed by ``MetricSpec.metric.value``.
        Each spec's ``reducer`` is given the matching benchmark family's
        extractor output, so adding a metric is a one-line edit in
        :data:`dexworld.types.metrics.METRICS`. ``np.nan`` indicates the
        metric isn't available for that (hand, retargeter).
        """
        extracts: Dict[BenchmarkMetric, Any] = {
            BenchmarkMetric.KEYVECTOR_MATCHING: _extract_keyvector(hand_data),
            BenchmarkMetric.MOTION_PRESERVATION: _extract_motion_preservation(
                hand_data
            ),
            BenchmarkMetric.FLATNESS: _extract_flatness(hand_data),
            BenchmarkMetric.WORKSPACE: _extract_workspace(hand_data),
            BenchmarkMetric.COLLISION: _extract_collision(hand_data),
            BenchmarkMetric.LATENCY: _extract_latency(hand_data),
        }
        out: Dict[str, float | str] = {
            s.metric.value: s.reducer(extracts[s.benchmark]) for s in METRICS
        }
        # ``lat_device`` is a string identifier (CUDA / CPU / mixed) that
        # doesn't fit MetricSpec's float-headline contract — pass it
        # through alongside so the retargeter chip header can display it.
        out["lat_device"] = extracts[BenchmarkMetric.LATENCY].get("device", "—")
        return out

    def _per_episode_headlines(
        self, hand_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Reduce one (hand, retargeter) pkl down to ``{ep_id: {stat_key: scalar}}``.

        Mirrors :meth:`_per_hand_headline` but applied per-episode instead
        of pooled across episodes — the Hand tab's line+ribbon charts use
        the spread across these episode headlines as the variation
        source (since the cross-hand view is meaningless when there's
        only one hand). Each ``MetricSpec.reducer`` is reused verbatim
        on per-episode family extracts.
        """
        per_ep_extracts: Dict[BenchmarkMetric, Dict[str, Any]] = {
            BenchmarkMetric.KEYVECTOR_MATCHING: _extract_keyvector_per_episode(
                hand_data
            ),
            BenchmarkMetric.MOTION_PRESERVATION: _extract_motion_preservation_per_episode(
                hand_data
            ),
            BenchmarkMetric.FLATNESS: _extract_flatness_per_episode(hand_data),
            BenchmarkMetric.WORKSPACE: _extract_workspace_per_episode(hand_data),
            BenchmarkMetric.COLLISION: _extract_collision_per_episode(hand_data),
            BenchmarkMetric.LATENCY: _extract_latency_per_episode(hand_data),
        }
        # Union of episode IDs across families (some metrics may be
        # missing for an episode but present for others — keep them).
        ep_ids: set[str] = set()
        for fam in per_ep_extracts.values():
            ep_ids.update(fam.keys())

        out: Dict[str, Dict[str, float]] = {}
        for ep_id in sorted(ep_ids):
            ep_headline: Dict[str, float] = {}
            for s in METRICS:
                fam_extract = per_ep_extracts[s.benchmark].get(ep_id)
                if fam_extract is None:
                    ep_headline[s.metric.value] = float("nan")
                    continue
                try:
                    ep_headline[s.metric.value] = float(s.reducer(fam_extract))
                except (KeyError, TypeError, ValueError):
                    ep_headline[s.metric.value] = float("nan")
            out[ep_id] = ep_headline
        return out

    def _aggregate_across_hands(self) -> Dict[str, Dict[str, float | str]]:
        """For each retargeter, return ``{stat_key: mean_across_hands}``.

        For numeric stats: simple mean of per-hand headline values
        (``np.nan`` skipped). For ``lat_device`` (string): the unique
        device if all hands match, else ``"mixed"``.
        """
        out: Dict[str, Dict[str, float | str]] = {}
        for ret in self.retargeters:
            per_hand_headlines: List[Dict[str, float | str]] = []
            for hand in self.hands:
                hd = self.per_hand_per_retargeter[hand].get(ret)
                if hd is None:
                    continue
                per_hand_headlines.append(self._per_hand_headline(hd))
            if not per_hand_headlines:
                continue

            ret_stats: Dict[str, float | str] = {}
            for key in (s.metric.value for s in METRICS):
                vals = [
                    h[key]
                    for h in per_hand_headlines
                    if isinstance(h[key], (int, float)) and not np.isnan(h[key])
                ]
                ret_stats[key] = float(np.mean(vals)) if vals else float("nan")

            devices = {
                h["lat_device"]
                for h in per_hand_headlines
                if isinstance(h["lat_device"], str) and h["lat_device"] != "—"
            }
            if not devices:
                ret_stats["lat_device"] = "—"
            elif len(devices) == 1:
                ret_stats["lat_device"] = next(iter(devices))
            else:
                ret_stats["lat_device"] = "mixed"

            ret_stats["n_hands"] = sum(
                1
                for h in self.hands
                if self.per_hand_per_retargeter[h].get(ret) is not None
            )
            out[ret] = ret_stats
        return out

    def _aggregate_for_one_hand(self, hand: str) -> Dict[str, Dict[str, float | str]]:
        """Single-hand headline dict — one entry per retargeter that has
        data for ``hand``. Mirrors :meth:`_aggregate_across_hands` but
        skips the cross-hand averaging step, so the values are exactly
        what the per-(hand, retargeter) pkl reports.

        ``n_hands`` is hard-coded to 1 (the chip header sub-line shows
        ``1 hand(s) · <device>``); ``lat_device`` comes straight from
        :meth:`_per_hand_headline` so a CUDA / CPU / mixed-pool device
        renders correctly.
        """
        if hand not in self.per_hand_per_retargeter:
            return {}
        out: Dict[str, Dict[str, float | str]] = {}
        for ret in self.retargeters:
            hd = self.per_hand_per_retargeter[hand].get(ret)
            if hd is None:
                continue
            headline = dict(self._per_hand_headline(hd))
            headline["n_hands"] = 1
            out[ret] = headline
        return out

    # Monochrome indigo podium tints — solid Tailwind shades so the
    # gradient is unmistakable on most monitors. The earlier low-alpha
    # values left rank 2 and rank 3 nearly indistinguishable; switching
    # to solid indigo-300 / -200 / -100 gives every step a clear ~25 RGB
    # unit jump, and rank 1 sits visibly darker than the rest. Only ranks
    # 1–3 get a background; rank ≥ 4 stays white.
    _PODIUM_TINT = {
        1: "rgb(165, 180, 252)",  # indigo-300 — clearly the deepest
        2: "rgb(199, 210, 254)",  # indigo-200 — mid step
        3: "rgb(224, 231, 255)",  # indigo-100 — lightest of the three
    }
    # 1st place gets a thin coloured border too, so the gold-medal cell
    # registers even when adjacent rows are also full of indigo.
    _PODIUM_BORDER = {
        1: "rgb(99, 102, 241)",  # indigo-500 — visible 1px frame on 1st
        2: "rgb(165, 180, 252)",  # indigo-300 — softer
        3: "rgb(199, 210, 254)",  # indigo-200 — softest
    }
    _PODIUM_ICON = {1: "🥇", 2: "🥈", 3: "🥉"}
    _PODIUM_FONT_WEIGHT = {1: "700", 2: "600", 3: "500"}

    @staticmethod
    def _direction_glyph(direction: Direction) -> str:
        """Single-character indicator placed inline next to the metric
        name. ``↑``/``↓`` for monotone metrics; ``→0``/``→1`` for the
        edge-cases where neither extreme is good."""
        return {
            Direction.UP: "↑",
            Direction.DOWN: "↓",
            Direction.ABS_DOWN: "→0",
            Direction.NEAR_ONE: "→1",
        }[direction]

    @staticmethod
    def _direction_help(direction: Direction) -> str:
        """Tooltip text for the direction glyph."""
        return {
            Direction.UP: "higher is better",
            Direction.DOWN: "lower is better",
            Direction.ABS_DOWN: "closer to 0 is better",
            Direction.NEAR_ONE: "closer to 1 is better",
        }[direction]

    @staticmethod
    def _rank_score(value: float, direction: Direction) -> float:
        """Convert a raw value into a "lower is better" score so we can rank
        every metric the same way regardless of direction."""
        if direction == Direction.UP:
            return -value
        if direction == Direction.DOWN:
            return value
        if direction == Direction.ABS_DOWN:
            return abs(value)
        if direction == Direction.NEAR_ONE:
            return abs(value - 1.0)
        raise ValueError(direction)

    @classmethod
    def _podium_style(cls, rank: int | None) -> dict:
        """Return the visual treatment for a cell at this rank.

        Implements the **podium rule**:

        * Rank 1 — darkest indigo tint, bold, gold medal.
        * Rank 2–3 — graduated tint and silver/bronze medal.
        * Rank ≥ 4 / unranked — no tint at all (the cell stays white) so
          the user's eye is drawn straight to the podium.
        """
        if rank in cls._PODIUM_TINT:
            return {
                "backgroundColor": cls._PODIUM_TINT[rank],
                "fontWeight": cls._PODIUM_FONT_WEIGHT[rank],
                "icon": cls._PODIUM_ICON[rank],
                "textColor": "#1e1b4b" if rank == 1 else "#312e81",  # indigo-950 / -900
            }
        return {
            "backgroundColor": "transparent",
            "fontWeight": "400",
            "icon": "",
            "textColor": "#374151",  # gray-700
        }

    @staticmethod
    def _format_value(value: Any, fmt: str) -> str:
        if value is None:
            return "—"
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return "—"
        try:
            return f"{value:{fmt}}"
        except (TypeError, ValueError):
            return str(value)

    def _per_hand_breakdown_text(self, ret: str, stat_key: str) -> str:
        """Multi-line tooltip text showing each hand's value for one
        (retargeter, metric) cell — what went into the row's aggregate."""
        lines = []
        for hand in self.hands:
            hd = self.per_hand_per_retargeter[hand].get(ret)
            if hd is None:
                lines.append(f"  {hand}: —")
                continue
            v = self._per_hand_headline(hd).get(stat_key, float("nan"))
            shown = v if isinstance(v, str) else f"{v:.4g}" if not np.isnan(v) else "—"
            lines.append(f"  {hand}: {shown}")
        return "\n".join(lines)

    # ── Cell builders (broken out so the loop in _build_summary_tab stays
    # readable; each helper produces one element of the CSS grid). ──

    def _build_corner_cell(self) -> html.Div:
        return html.Div(
            "Metric",
            style={
                "fontWeight": "bold",
                "padding": "10px 8px",
                "fontSize": "0.95em",
                "color": "#374151",
                "alignSelf": "center",
            },
        )

    def _build_retargeter_header_cell(
        self, retargeter: str, n_hands_for_r: int, device: str
    ) -> html.Div:
        return html.Div(
            [
                html.Div(
                    retargeter,
                    style={
                        "fontWeight": "bold",
                        "fontSize": "1.0em",
                        "color": _CHIP_TEXT_COLOR,
                        "textAlign": "center",
                    },
                ),
                html.Div(
                    f"{n_hands_for_r} hand(s) · {device}",
                    style={
                        "fontSize": "0.75em",
                        "color": _CHIP_SUB_COLOR,
                        "textAlign": "center",
                    },
                ),
            ],
            style={
                "backgroundColor": self.colors[retargeter],
                "padding": "8px 6px",
                "borderRadius": "4px",
            },
        )

    def _build_group_header_cell(self, name: str) -> html.Div:
        """Section divider that spans the full grid width."""
        return html.Div(
            name,
            style={
                "gridColumn": "1 / -1",
                "fontWeight": "700",
                "fontSize": "0.85em",
                "letterSpacing": "0.08em",
                "textTransform": "uppercase",
                "color": "#4338ca",  # indigo-700
                "padding": "14px 8px 4px 4px",
                "borderBottom": "2px solid #e0e7ff",  # indigo-100
                "marginTop": "6px",
            },
        )

    def _build_metric_label_cell(self, spec: MetricSpec) -> html.Div:
        children: List[Any] = []
        if spec.emoji:
            children.append(
                html.Span(
                    spec.emoji,
                    title=f"{spec.benchmark.value} family",
                    style={
                        "marginRight": "8px",
                        "fontSize": "1.05em",
                        "lineHeight": "1.2em",
                    },
                )
            )
        children.append(
            html.Span(
                spec.summary_label,
                style={"fontWeight": "500", "fontSize": "0.95em"},
            )
        )
        children.append(
            html.Span(
                f" {self._direction_glyph(spec.direction)}",
                title=self._direction_help(spec.direction),
                style={
                    "fontWeight": "700",
                    "color": "#4338ca",
                    "marginLeft": "6px",
                    "cursor": "help",
                },
            )
        )
        return html.Div(
            children,
            style={
                "padding": "8px",
                "backgroundColor": "#f9fafb",
                "borderRadius": "4px",
                "alignSelf": "stretch",
                "display": "flex",
                "alignItems": "center",
            },
        )

    def _build_value_cell(
        self,
        retargeter: str,
        value: float,
        rank: int | None,
        total_ranked: int,
        fmt: str,
        stat_key: str,
        tooltip_override: str | None = None,
    ) -> html.Div:
        podium = self._podium_style(rank)
        bg_color = podium["backgroundColor"]

        # Compose hover tooltip. Summary mode (default) shows the
        # per-hand breakdown that fed the aggregate; Hand mode passes
        # ``tooltip_override`` so the tooltip just summarises the single
        # value without a redundant breakdown.
        rank_label = f"{rank} of {total_ranked}" if rank is not None else "unranked"
        if tooltip_override is not None:
            hover_text = tooltip_override
        else:
            hover_text = (
                f"{retargeter}\n"
                f"value: {self._format_value(value, fmt)}\n"
                f"rank: {rank_label}\n"
                f"\nper-hand:\n"
                f"{self._per_hand_breakdown_text(retargeter, stat_key)}"
            )

        # Value row: medal icon (if podium) + the numeric value.
        value_row_children: List[Any] = []
        if podium["icon"]:
            value_row_children.append(
                html.Span(
                    podium["icon"],
                    style={
                        "fontSize": "1.0em",
                        "marginRight": "4px",
                        "lineHeight": "1.2em",
                    },
                )
            )
        value_row_children.append(
            html.Span(
                self._format_value(value, fmt),
                style={
                    "fontSize": "1.30em",
                    "fontWeight": podium["fontWeight"],
                    "fontFamily": "ui-monospace, SF Mono, monospace",
                    "color": podium["textColor"],
                },
            )
        )

        # Podium cells get a coloured border (1st = thicker indigo-500
        # frame, 2nd/3rd = softer); rank-4+ cells stay on the default
        # grey hairline.
        if rank in self._PODIUM_BORDER:
            border_color = self._PODIUM_BORDER[rank]
            border_width = "2px" if rank == 1 else "1px"
        else:
            border_color = "#e5e7eb"
            border_width = "1px"

        cell_style: Dict[str, Any] = {
            "padding": "0 6px 6px 6px",
            "backgroundColor": bg_color,
            "border": f"{border_width} solid {border_color}",
            "borderRadius": "4px",
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "center",
        }

        return html.Div(
            [
                # Retargeter colour stripe — keeps column-tracking even when
                # the cell itself isn't on the podium.
                html.Div(
                    style={
                        "height": "4px",
                        "backgroundColor": self.colors[retargeter],
                        "borderRadius": "3px 3px 0 0",
                        "marginBottom": "5px",
                    }
                ),
                html.Div(
                    value_row_children,
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "gap": "2px",
                    },
                ),
            ],
            title=hover_text,
            style=cell_style,
        )

    def _build_grid(
        self,
        agg: Dict[str, Dict[str, float | str]],
        *,
        single_hand: str | None = None,
    ) -> html.Div:
        """Build the comparison grid from a per-retargeter headline dict.

        Shared by ``_build_summary_tab`` (which feeds in
        :meth:`_aggregate_across_hands`) and ``_build_hand_tab`` (which
        feeds in :meth:`_aggregate_for_one_hand`). The grid layout, the
        podium-tinting, the direction glyphs, and the metric grouping are
        identical — only the data source differs.

        Parameters
        ----------
        agg
            ``{retargeter: {stat_key: value, ..., "n_hands": int,
            "lat_device": str}}``. ``n_hands`` and ``lat_device`` drive
            the retargeter header chip's sub-line.
        single_hand
            When non-None, the grid was built from one hand. Each value
            cell's tooltip is replaced with a short single-hand summary
            (no per-hand breakdown, since there's only one).
        """
        retargeters_present = [r for r in self.retargeters if r in agg]

        # CSS grid: fixed-width label column, equal-width retargeter columns.
        grid_style = {
            "display": "grid",
            "gridTemplateColumns": "260px "
            + " ".join(["1fr"] * len(retargeters_present)),
            "gap": "6px",
            "alignItems": "stretch",
        }

        grid_children: List[html.Div] = []

        # ── Header row (corner + retargeter chips) ──
        grid_children.append(self._build_corner_cell())
        for r in retargeters_present:
            grid_children.append(
                self._build_retargeter_header_cell(
                    r, agg[r]["n_hands"], agg[r]["lat_device"]
                )
            )

        # ── Grouped data rows ──
        for family in families_in_order():
            # Apply the hidden-metric filter per-family; if a family has
            # no surviving rows, skip its sub-header too (no orphan
            # headers).
            visible_specs = [
                s
                for s in metrics_in_family(family)
                if s.metric.value not in self.hidden_metrics
            ]
            if not visible_specs:
                continue
            grid_children.append(self._build_group_header_cell(family.value.title()))

            for spec in visible_specs:
                stat_key = spec.metric.value
                fmt = spec.fmt
                direction = spec.direction
                grid_children.append(self._build_metric_label_cell(spec))

                # Pull this row's values across present retargeters.
                row_values: Dict[str, float] = {}
                for r in retargeters_present:
                    v = agg[r].get(stat_key, float("nan"))
                    row_values[r] = (
                        float(v) if isinstance(v, (int, float)) else float("nan")
                    )

                valid_pairs = [
                    (r, v)
                    for r, v in row_values.items()
                    if not (isinstance(v, float) and np.isnan(v))
                ]
                valid_pairs_sorted = sorted(
                    valid_pairs, key=lambda rv: self._rank_score(rv[1], direction)
                )
                rank_by_ret: Dict[str, int] = {
                    r: i + 1 for i, (r, _) in enumerate(valid_pairs_sorted)
                }
                total_ranked = len(valid_pairs_sorted)

                for r in retargeters_present:
                    rank = rank_by_ret.get(r)
                    tooltip_override: str | None = None
                    if single_hand is not None:
                        rank_str = (
                            f"{rank} of {total_ranked}"
                            if rank is not None
                            else "unranked"
                        )
                        tooltip_override = (
                            f"{r}\n"
                            f"hand: {single_hand}\n"
                            f"value: {self._format_value(row_values[r], fmt)}\n"
                            f"rank: {rank_str}"
                        )
                    grid_children.append(
                        self._build_value_cell(
                            retargeter=r,
                            value=row_values[r],
                            rank=rank,
                            total_ranked=total_ranked,
                            fmt=fmt,
                            stat_key=stat_key,
                            tooltip_override=tooltip_override,
                        )
                    )

        return html.Div(grid_children, style=grid_style)

    # ── Per-metric line + ribbon charts ───────────────────────────────────
    #
    # Companion to the podium grid: one chart per metric, x-axis is the
    # retargeters. A line tracks the per-retargeter mean across hands,
    # a shaded ribbon spans the min-max envelope, and individual hand
    # values sit on top as coloured markers. Lets the reader spot
    # per-hand variance that the cross-hand mean in the grid smooths
    # over.

    def _build_per_metric_charts_section(self) -> html.Div:
        """Render a chart per visible metric, organised by metric family.

        Each chart is a line + min-max ribbon across retargeters with
        per-hand markers on top. Hands use the stable
        :func:`hand_color` palette.
        """
        # Compute every (hand, retargeter) headline once and reuse it
        # across all charts — extracting from pkl per chart would scale
        # multiplicatively with metric count.
        headlines: Dict[str, Dict[str, Dict[str, float | str]]] = {}
        for hand in self.hands:
            headlines[hand] = {}
            for ret in self.retargeters:
                hd = self.per_hand_per_retargeter[hand].get(ret)
                if hd is not None:
                    headlines[hand][ret] = self._per_hand_headline(hd)

        sections: List[Any] = [
            html.H3(
                "Per-metric breakdown across hands",
                style={
                    "marginTop": "24px",
                    "marginBottom": "4px",
                    "fontSize": "1.05em",
                    "color": "#374151",
                },
            ),
            html.Div(
                "Each chart shows one metric across retargeters. The line "
                "tracks the mean across hands; whiskers + shaded ribbon "
                "show the min–max range; coloured markers are individual "
                "hand values. A long whisker (or wide ribbon) means high "
                "per-hand variance.",
                style={
                    "color": "#555",
                    "marginBottom": "12px",
                    "fontSize": "0.9em",
                },
            ),
        ]

        for family in families_in_order():
            visible_specs = [
                s
                for s in metrics_in_family(family)
                if s.metric.value not in self.hidden_metrics
            ]
            if not visible_specs:
                continue
            sections.append(self._build_group_header_cell(family.value.title()))
            sections.append(
                html.Div(
                    [
                        self._build_metric_line_chart(spec, headlines)
                        for spec in visible_specs
                    ],
                    style={
                        "display": "grid",
                        # One chart per row — wide bars let per-hand
                        # comparisons read cleanly even with 7 retargeter
                        # groups across the x-axis.
                        "gridTemplateColumns": "1fr",
                        "gap": "12px",
                        "marginBottom": "12px",
                    },
                )
            )

        return html.Div(sections)

    def _build_per_metric_charts_section_for_hand(self, hand: str) -> html.Div:
        """Hand-tab counterpart of :meth:`_build_per_metric_charts_section`.

        Same line + ribbon layout, but the per-retargeter spread comes
        from episode-to-episode variation within
        ``self.per_hand_per_retargeter[hand][retargeter]`` instead of
        cross-hand variation — there's only one hand selected, so
        episodes are the natural variation axis.
        """
        hand_data = self.per_hand_per_retargeter.get(hand, {})

        # Reshape per-retargeter per-episode headlines into the chart's
        # expected ``{member: {retargeter: {stat_key: scalar}}}`` shape.
        headlines: Dict[str, Dict[str, Dict[str, float | str]]] = {}
        ep_ids: set[str] = set()
        for ret, hd in hand_data.items():
            for ep_id, ep_headline in self._per_episode_headlines(hd).items():
                ep_ids.add(ep_id)
                headlines.setdefault(ep_id, {})[ret] = ep_headline
        members = sorted(ep_ids)

        sections: List[Any] = [
            html.H3(
                f"Per-metric breakdown across episodes — {hand.replace('_', ' ')}",
                style={
                    "marginTop": "24px",
                    "marginBottom": "4px",
                    "fontSize": "1.05em",
                    "color": "#374151",
                },
            ),
            html.Div(
                "Each chart shows one metric across retargeters for this "
                "hand. The line tracks the mean across episodes; whiskers "
                "+ shaded ribbon show the min–max range; semi-transparent "
                "indigo dots are the individual per-episode values from "
                "the dataset. A long whisker (or wide ribbon) means high "
                "per-episode variance — the retargeter behaves "
                "inconsistently across the dataset's demos.",
                style={
                    "color": "#555",
                    "marginBottom": "12px",
                    "fontSize": "0.9em",
                },
            ),
        ]
        if not members:
            sections.append(
                html.Div(
                    "No per-episode data available for this hand.",
                    style={"color": "#9ca3af", "padding": "12px"},
                )
            )
            return html.Div(sections)

        for family in families_in_order():
            visible_specs = [
                s
                for s in metrics_in_family(family)
                if s.metric.value not in self.hidden_metrics
            ]
            if not visible_specs:
                continue
            sections.append(self._build_group_header_cell(family.value.title()))
            sections.append(
                html.Div(
                    [
                        self._build_metric_line_chart(
                            spec,
                            headlines,
                            members=members,
                            member_label="episode",
                            markers="pooled",
                        )
                        for spec in visible_specs
                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr",
                        "gap": "12px",
                        "marginBottom": "12px",
                    },
                )
            )

        return html.Div(sections)

    def _build_metric_line_chart(
        self,
        spec: MetricSpec,
        headlines: Dict[str, Dict[str, Dict[str, float | str]]],
        *,
        members: List[str] | None = None,
        member_label: str = "hand",
        markers: Literal["per_member", "pooled", "off"] = "per_member",
    ) -> html.Div:
        """Line + min-max ribbon chart for one metric.

        ``x = retargeters``. The line tracks the per-retargeter mean
        across the spread members; the shaded ribbon spans the min-max
        envelope; each member's individual value can appear as a
        marker.

        Parameters
        ----------
        spec
            Metric to plot.
        headlines
            ``{member: {retargeter: {stat_key: scalar}}}`` — the raw
            data fed to the chart.
        members
            Identifiers that contribute the per-retargeter spread.
            Defaults to ``self.hands`` (Summary tab); pass episode IDs
            for the Hand tab.
        member_label
            Singular noun used in legend / hover text — ``"hand"`` for
            the Summary tab, ``"episode"`` for the Hand tab.
        markers
            How to draw individual data points:

            * ``"per_member"`` (default) — one scatter trace per member
              with :func:`hand_color` and a legend entry. Best when
              members are few and identifying individuals matters
              (Summary tab, ≤ ~10 hands).
            * ``"pooled"`` — single semi-transparent indigo scatter
              trace with all (retargeter, member) points and one
              legend entry. Best when there are many members and
              per-identity legend rows would be noise (Hand tab, N
              episodes). Hover still surfaces the member id.
            * ``"off"`` — no markers; only the line + ribbon +
              annotations show.
        """
        if members is None:
            members = self.hands

        stat_key = spec.metric.value
        fmt = spec.fmt
        retargeters_present = [
            r
            for r in self.retargeters
            if any(r in headlines.get(m, {}) for m in members)
        ]

        # Collect per-(retargeter, member) scalars; drop NaN/missing.
        per_ret_member_vals: Dict[str, Dict[str, float]] = {
            r: {} for r in retargeters_present
        }
        for m in members:
            for ret in retargeters_present:
                v = headlines.get(m, {}).get(ret, {}).get(stat_key, float("nan"))
                if isinstance(v, (int, float)) and not (
                    isinstance(v, float) and np.isnan(v)
                ):
                    per_ret_member_vals[ret][m] = float(v)

        # Per-retargeter aggregates for the line + ribbon.
        mins: List[float | None] = []
        means: List[float | None] = []
        maxes: List[float | None] = []
        for ret in retargeters_present:
            vs = list(per_ret_member_vals[ret].values())
            if vs:
                mins.append(min(vs))
                means.append(sum(vs) / len(vs))
                maxes.append(max(vs))
            else:
                mins.append(None)
                means.append(None)
                maxes.append(None)

        # Chart-consistent indigo so the eye locks onto the metric
        # shape, not the retargeter colour scheme (which lives on the
        # podium grid above).
        _LINE_COLOR = "rgb(99, 102, 241)"  # indigo-500
        _RIBBON_COLOR = "rgba(165, 180, 252, 0.35)"  # indigo-300 @ 35% alpha

        fig = go.Figure()

        # Ribbon — two traces: invisible upper edge, then lower edge
        # with ``fill="tonexty"`` to shade the band between them.
        fig.add_trace(
            go.Scatter(
                x=retargeters_present,
                y=maxes,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name="max",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=retargeters_present,
                y=mins,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=_RIBBON_COLOR,
                showlegend=False,
                hoverinfo="skip",
                name="min",
            )
        )

        # Mean line on top of the ribbon, with asymmetric error bars
        # from min to max so the per-retargeter spread is pinned to its
        # x position — the ribbon shows the across-retargeter envelope,
        # the whiskers show the at-this-retargeter range.
        mean_text = [
            self._format_value(m, fmt) if m is not None else "—" for m in means
        ]
        range_text = [
            f"range: {self._format_value(mx - mn, fmt)}"
            if mn is not None and mx is not None
            else "no data"
            for mn, mx in zip(mins, maxes)
        ]
        err_plus = [
            (mx - m) if (mx is not None and m is not None) else 0.0
            for mx, m in zip(maxes, means)
        ]
        err_minus = [
            (m - mn) if (mn is not None and m is not None) else 0.0
            for mn, m in zip(mins, means)
        ]
        fig.add_trace(
            go.Scatter(
                x=retargeters_present,
                y=means,
                mode="lines+markers",
                line=dict(color=_LINE_COLOR, width=2),
                marker=dict(color=_LINE_COLOR, size=8),
                name=f"mean across {member_label}s",
                text=mean_text,
                customdata=range_text,
                hovertemplate=(
                    "<b>%{x}</b><br>mean: %{text}<br>%{customdata}<extra></extra>"
                ),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=err_plus,
                    arrayminus=err_minus,
                    color=_LINE_COLOR,
                    thickness=1.5,
                    width=8,
                ),
            )
        )

        # Per-retargeter ``avg: … / range: …`` boxed annotations,
        # anchored in paper coordinates ABOVE the plot area so they
        # don't push the y-axis past the real data extents (which would
        # mislead the reader about variance). ``range`` is the scalar
        # spread ``max - min``; collapses to mean only when min == max.
        for ret, m, mn, mx in zip(retargeters_present, means, mins, maxes):
            if m is None:
                continue
            if mn == mx:
                label = f"avg: {self._format_value(m, fmt)}"
            else:
                label = (
                    f"avg: {self._format_value(m, fmt)}<br>"
                    f"range: {self._format_value(mx - mn, fmt)}"
                )
            fig.add_annotation(
                x=ret,
                xref="x",
                y=1.0,
                yref="paper",
                yanchor="bottom",
                yshift=6,  # gap between plot top edge and box bottom
                text=label,
                showarrow=False,
                font=dict(size=12, color=_LINE_COLOR),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.92)",
                bordercolor=_LINE_COLOR,
                borderwidth=1,
                borderpad=4,
            )

        # Per-member markers on top — keeps individual-member visibility
        # without re-cluttering the chart like the bars did.
        if markers == "per_member":
            for m in members:
                ys: List[float | None] = [
                    per_ret_member_vals[r].get(m) for r in retargeters_present
                ]
                if not any(y is not None for y in ys):
                    continue
                m_display = str(m).replace("_", " ")
                fig.add_trace(
                    go.Scatter(
                        x=retargeters_present,
                        y=ys,
                        mode="markers",
                        name=m_display,
                        marker=dict(
                            color=hand_color(m, members),
                            size=8,
                            line=dict(color="white", width=1),
                        ),
                        hovertemplate=(
                            f"<b>%{{x}}</b><br>{m_display}: %{{y}}<extra></extra>"
                        ),
                    )
                )
        elif markers == "pooled":
            # Single trace, one legend entry, semi-transparent indigo
            # markers stacked at each retargeter x. Hover surfaces the
            # member id even though it's not in the legend.
            xs: List[str] = []
            ys_pool: List[float] = []
            member_ids: List[str] = []
            for m in members:
                for r in retargeters_present:
                    v = per_ret_member_vals[r].get(m)
                    if v is None:
                        continue
                    xs.append(r)
                    ys_pool.append(v)
                    member_ids.append(str(m).replace("_", " "))
            if xs:
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys_pool,
                        mode="markers",
                        name=f"per-{member_label} value",
                        marker=dict(
                            color="rgba(99, 102, 241, 0.55)",  # indigo-500 @ 55%
                            size=6,
                            line=dict(color="white", width=0.5),
                        ),
                        customdata=member_ids,
                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            f"{member_label}: %{{customdata}}<br>"
                            "value: %{y}<extra></extra>"
                        ),
                    )
                )

        title = (
            f"{spec.emoji} <u>{spec.summary_label}</u> "
            f"{self._direction_glyph(spec.direction)}"
        ).strip()
        # Re-use the same numeric format spec as the value labels so
        # y-axis ticks read the same (e.g. ``.1%`` shows percentages,
        # ``.2e`` shows scientific notation for flatness).
        y_axis_title = spec.yaxis_label

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18)),
            # Top margin holds the per-retargeter ``avg / range`` boxed
            # annotations (paper-coordinate, ``yref="paper"``). The
            # y-axis itself stays auto-scaled to the data so the chart
            # doesn't lie about variance — the boxes live above the
            # plot, not inside it.
            margin=dict(l=64, r=12, t=110, b=48),
            height=360,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="center",
                x=0.5,
                font=dict(size=10),
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=False, tickfont=dict(size=11), type="category"),
            yaxis=dict(
                title=dict(text=y_axis_title, font=dict(size=11)),
                gridcolor="#e5e7eb",
                zerolinecolor="#9ca3af",
                tickformat=fmt,
                tickfont=dict(size=10),
            ),
        )
        return html.Div(
            dcc.Graph(
                figure=fig,
                # ``displayModeBar="hover"`` keeps the chart clean when
                # idle but reveals plotly's toolbar (incl. the camera /
                # download-as-PNG button) on mouseover. The unrelated
                # interaction tools (lasso, box-select, autoscale) are
                # removed so only the download + reset zoom remain.
                # ``toImageButtonOptions`` gives each downloaded PNG a
                # readable name derived from the metric and renders at
                # 2× scale for crisp slides.
                config={
                    "displayModeBar": "hover",
                    "displaylogo": False,
                    "modeBarButtonsToRemove": [
                        "lasso2d",
                        "select2d",
                        "autoScale2d",
                        "zoom2d",
                        "pan2d",
                        "zoomIn2d",
                        "zoomOut2d",
                    ],
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": f"{spec.metric.value}_chart",
                        "scale": 2,
                    },
                },
            ),
            style={
                "border": "1px solid #e5e7eb",
                "borderRadius": "4px",
                "padding": "4px",
                "backgroundColor": "white",
            },
        )

    def _build_summary_tab(self) -> html.Div:
        """Cross-hand aggregate grid + explanatory note + legend.

        Calls :meth:`_build_grid` with the cross-hand-averaged headline
        dict; layout/tinting/grouping all live in the shared grid
        builder. This method is responsible only for the wrapping
        material around the grid.
        """
        agg = self._aggregate_across_hands()
        grid = self._build_grid(agg)

        # ── Header note & legend ──
        note_lines: List[str] = [
            f"Aggregated over {len(self.hands)} hand(s): "
            f"{', '.join(h.replace('_', ' ') for h in self.hands)}. "
            f"Each cell = mean across hands of that retargeter's per-hand "
            f"value. Hover any cell for the per-hand breakdown."
        ]
        if self.hidden_metrics:
            # Distinguish "factory defaults still in effect" from "user
            # asked to hide a custom set" so the dashboard's behaviour
            # isn't a black box.
            if self.hidden_metrics == set(self._DEFAULT_HIDDEN_METRICS):
                note_lines.append(
                    "Hidden by default: "
                    + ", ".join(sorted(self.hidden_metrics))
                    + ". Pass --show-all-metrics to display them."
                )
            else:
                note_lines.append(
                    "Hidden Summary rows: "
                    + ", ".join(sorted(self.hidden_metrics))
                    + "."
                )
        n_hands_note = html.Div(
            [html.Div(line) for line in note_lines],
            style={"color": "#555", "marginBottom": "10px", "fontSize": "0.9em"},
        )

        # Explanatory block — collapsed by default so it doesn't push the
        # grid down for repeat users, but always available for newcomers.
        # Spelling out *what the data bar is* was the user-reported gap.
        how_to_read = html.Details(
            [
                html.Summary(
                    "ℹ️  How to read this table",
                    style={
                        "cursor": "pointer",
                        "fontWeight": "600",
                        "color": "#4338ca",
                        "padding": "4px 0",
                    },
                ),
                html.Ul(
                    [
                        html.Li(
                            [
                                html.B("Podium colours:"),
                                " only the top 3 retargeters in each row get tinted "
                                "and a medal — ",
                                html.Span(
                                    " 🥇 1st ",
                                    style={
                                        "backgroundColor": self._PODIUM_TINT[1],
                                        "padding": "1px 6px",
                                        "borderRadius": "3px",
                                        "fontWeight": "700",
                                    },
                                ),
                                " is the deepest indigo and has a thicker border, ",
                                html.Span(
                                    " 🥈 2nd ",
                                    style={
                                        "backgroundColor": self._PODIUM_TINT[2],
                                        "padding": "1px 6px",
                                        "borderRadius": "3px",
                                        "fontWeight": "600",
                                    },
                                ),
                                " is mid, ",
                                html.Span(
                                    " 🥉 3rd ",
                                    style={
                                        "backgroundColor": self._PODIUM_TINT[3],
                                        "padding": "1px 6px",
                                        "borderRadius": "3px",
                                        "fontWeight": "500",
                                    },
                                ),
                                " is the lightest of the three. Rank 4+ stays "
                                "white so the eye locks onto the podium.",
                            ]
                        ),
                        html.Li(
                            [
                                html.B("Direction glyphs"),
                                " next to each metric name: ",
                                html.Code("↑"),
                                " higher is better, ",
                                html.Code("↓"),
                                " lower is better, ",
                                html.Code("→0"),
                                " closer to 0 is better, ",
                                html.Code("→1"),
                                " closer to 1 is better. Hover the glyph to see "
                                "the explanation.",
                            ]
                        ),
                        html.Li(
                            [
                                html.B("Column colour stripe"),
                                " above each value cell uses the retargeter's "
                                "stable colour so you can track one retargeter "
                                "down the table even when its cells aren't on "
                                "the podium.",
                            ]
                        ),
                    ],
                    style={
                        "margin": "8px 0 4px 18px",
                        "fontSize": "0.88em",
                        "lineHeight": "1.55em",
                    },
                ),
            ],
            open=False,
            style={
                "marginBottom": "10px",
                "padding": "6px 10px",
                "border": "1px solid #e5e7eb",
                "borderRadius": "4px",
                "backgroundColor": "#fafafa",
            },
        )

        legend_chip_style: Dict[str, str] = {
            "padding": "4px 10px",
            "marginRight": "6px",
            "borderRadius": "3px",
            "border": "1px solid #e5e7eb",
            "fontFamily": "ui-monospace, SF Mono, monospace",
        }
        legend = html.Div(
            [
                html.Span(
                    "Podium:", style={"fontWeight": "bold", "marginRight": "10px"}
                ),
                html.Span(
                    "🥇 1st",
                    style={
                        **legend_chip_style,
                        "backgroundColor": self._PODIUM_TINT[1],
                        "fontWeight": "700",
                    },
                ),
                html.Span(
                    "🥈 2nd",
                    style={
                        **legend_chip_style,
                        "backgroundColor": self._PODIUM_TINT[2],
                        "fontWeight": "600",
                    },
                ),
                html.Span(
                    "🥉 3rd",
                    style={
                        **legend_chip_style,
                        "backgroundColor": self._PODIUM_TINT[3],
                        "fontWeight": "500",
                    },
                ),
                html.Span(
                    "4th+",
                    style={**legend_chip_style, "color": "#9ca3af"},
                ),
                html.Span("  ·  ", style={"color": "#aaa", "margin": "0 6px"}),
                html.Span(
                    "↑ higher better",
                    style={**legend_chip_style, "color": "#4338ca"},
                ),
                html.Span(
                    "↓ lower better",
                    style={**legend_chip_style, "color": "#4338ca"},
                ),
            ],
            style={
                "marginBottom": "14px",
                "fontSize": "0.85em",
                "display": "flex",
                "flexWrap": "wrap",
                "alignItems": "center",
                "gap": "4px",
            },
        )

        per_metric_charts = self._build_per_metric_charts_section()

        return html.Div([n_hands_note, how_to_read, legend, grid, per_metric_charts])

    # ── Hand tab (per-hand drilldown) ──────────────────────────────────────

    def _build_hand_content(self, hand: str) -> html.Div:
        """Render the dropdown-driven section of the Hand tab — grid +
        per-metric charts — for one hand. Returned wholesale by the
        Dash callback so both views re-render together on hand change.
        """
        grid = self._build_grid(
            self._aggregate_for_one_hand(hand),
            single_hand=hand,
        )
        charts = self._build_per_metric_charts_section_for_hand(hand)
        return html.Div([grid, charts])

    def _build_hand_tab(self) -> html.Div:
        """Per-hand drilldown — same grid layout as Summary, plus a
        per-metric line+ribbon section showing episode-to-episode
        spread for the selected hand.

        The dropdown lists every loaded hand and defaults to
        ``self.primary_hand``. Both grid and chart section live inside
        ``cmp-hand-content``; the Dash callback (registered in
        :meth:`build`) replaces the whole container on dropdown change
        so they stay in sync.
        """
        initial_content = self._build_hand_content(self.primary_hand)
        note = html.Div(
            "Per-hand drilldown — same grid layout as Summary, but the "
            "values come from one hand at a time instead of being averaged "
            "across all hands. The per-metric charts below show the "
            "spread across episodes within this hand. Use the dropdown "
            "to switch hands.",
            style={"color": "#555", "marginBottom": "10px", "fontSize": "0.9em"},
        )
        dropdown_label = html.Label(
            "Hand:",
            htmlFor="cmp-hand-dropdown",
            style={
                "fontWeight": "600",
                "marginRight": "8px",
                "color": "#374151",
                "fontSize": "0.9em",
            },
        )
        dropdown = dcc.Dropdown(
            id="cmp-hand-dropdown",
            options=[{"label": h.replace("_", " "), "value": h} for h in self.hands],
            value=self.primary_hand,
            clearable=False,
            style={"width": "320px"},
        )
        hand_image = html.Img(
            id="cmp-hand-image",
            src=f"/media/hands/{self.primary_hand}.webp",
            style={
                "height": "100px",
                "marginLeft": "16px",
                "borderRadius": "6px",
                "objectFit": "contain",
                "backgroundColor": "#1f2937",
            },
        )
        controls = html.Div(
            [dropdown_label, dropdown, hand_image],
            style={
                "display": "flex",
                "alignItems": "center",
                "marginBottom": "12px",
            },
        )
        return html.Div(
            [
                note,
                controls,
                html.Div(initial_content, id="cmp-hand-content"),
            ]
        )
