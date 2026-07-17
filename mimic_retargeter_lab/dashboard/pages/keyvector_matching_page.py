"""Dashboard page for the time-series Keyvector Matching metric.

Layout (top → bottom):

    EPISODES (vertical radio; first ~4 visible, rest scroll)
        ◉ ep_id    cos_sim (median)=…  scale_ratio (robot/human, median)=…  n=…
        ○ ep_id    …
    ──────────────────────────────────────────────────────────────────────
    Keyvector Matching (Dataset: <selected ep_id>)
    ┌────────────────────────────────┐  ┌────────────────────────────────┐
    │ thumb_to_index_tip             │  │ thumb_to_middle_tip            │
    │   cosine_similarity: median=…  │  │   cosine_similarity: …         │
    │   scale_ratio:       median=…  │  │   scale_ratio:       …         │
    │   angle_difference:  median=…  │  │   angle_difference:  …         │
    │   length_difference: median=…  │  │   length_difference: …         │
    └────────────────────────────────┘  └────────────────────────────────┘
    ┌────────────────────────────────┐  ┌────────────────────────────────┐
    │ thumb_to_ring_tip              │  │ thumb_to_little_tip            │
    │   …                            │  │   …                            │
    └────────────────────────────────┘  └────────────────────────────────┘

The metric ordering inside each card is set by
:func:`mimic_retargeter_lab.types.metrics.kv_metrics` — direction-and-scale-invariant
axes first (cosine_similarity, scale_ratio), then absolute-error axes
(angle_difference, length_difference).
"""

from typing import Any, Dict, List

import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output

from mimic_retargeter_lab.types.metrics import kv_metrics


_PRIMARY_KEY = "cosine_similarity"


class KeyvectorMatchingPage:
    LEFT_COL_PX = 240
    CONTENT_HEIGHT = "calc(100vh - 120px)"

    def __init__(self, app, page_id: str):
        self.app = app
        self.page_id = page_id

    # ── Data accessors ───────────────────────────────────────────────────

    @staticmethod
    def _to_list_safe(v) -> List[Any]:
        if v is None:
            return []
        try:
            return list(v)
        except Exception:
            return [v]

    @staticmethod
    def _vector_names(ep_data: Dict[str, Any]) -> List[str]:
        return list((ep_data.get("vector_metrics") or {}).keys())

    def _vector_metric(
        self, ep_data: Dict[str, Any], vec_name: str, key: str
    ) -> Dict[str, Any]:
        return ((ep_data.get("vector_metrics") or {}).get(vec_name, {}) or {}).get(
            key, {}
        ) or {}

    def _vector_raw(
        self, ep_data: Dict[str, Any], vec_name: str, key: str
    ) -> np.ndarray:
        raw = self._vector_metric(ep_data, vec_name, key).get("raw")
        if raw is None:
            return np.array([], dtype=float)
        arr = np.asarray(list(raw), dtype=float)
        return arr[~np.isnan(arr)]

    def _vector_smoothed(
        self, ep_data: Dict[str, Any], vec_name: str, key: str
    ) -> np.ndarray:
        smoothed = self._vector_metric(ep_data, vec_name, key).get("smoothed")
        if smoothed is None:
            return np.array([], dtype=float)
        arr = np.asarray(list(smoothed), dtype=float)
        return arr

    def _pool_episode_cos_sim(self, ep_data: Dict[str, Any]) -> np.ndarray:
        """Concat ``cosine_similarity.raw`` across all configured vectors."""
        return self._pool_episode_metric(ep_data, _PRIMARY_KEY)

    def _pool_episode_scale_ratio(self, ep_data: Dict[str, Any]) -> np.ndarray:
        """Concat ``scale_ratio.raw`` across all configured vectors."""
        return self._pool_episode_metric(ep_data, "scale_ratio")

    def _pool_episode_metric(self, ep_data: Dict[str, Any], key: str) -> np.ndarray:
        pooled: list = []
        for vec_name in self._vector_names(ep_data):
            pooled.extend(list(self._vector_raw(ep_data, vec_name, key)))
        if not pooled:
            return np.array([], dtype=float)
        arr = np.asarray(pooled, dtype=float)
        return arr[~np.isnan(arr)]

    # ── Stats text ───────────────────────────────────────────────────────

    @staticmethod
    def _percentiles_stats_text(
        arr: np.ndarray, value_fmt: str = "{:.3f}", unit: str = ""
    ) -> str:
        """Five-number summary (median, p75, p25, p5, p1) + n.

        Used for ``cosine_similarity`` — bounded distribution where the low
        tail surfaces failures.
        """
        if arr.size == 0:
            return "(no data)"

        def f(v):
            return f"{value_fmt.format(v)}{unit}"

        return (
            f"median={f(np.median(arr))} | "
            f"p75={f(np.percentile(arr, 75))} | "
            f"p25={f(np.percentile(arr, 25))} | "
            f"p5={f(np.percentile(arr, 5))} | "
            f"p1={f(np.percentile(arr, 1))} | "
            f"n={arr.size}"
        )

    @staticmethod
    def _percentiles_both_tails_stats_text(
        arr: np.ndarray, value_fmt: str = "{:.3f}", unit: str = ""
    ) -> str:
        """Five-number summary + p95/p99 + n.

        Used for ``scale_ratio`` — deviations in either direction matter, so
        both low and high tails are shown.
        """
        if arr.size == 0:
            return "(no data)"

        def f(v):
            return f"{value_fmt.format(v)}{unit}"

        return (
            f"median={f(np.median(arr))} | "
            f"p75={f(np.percentile(arr, 75))} | "
            f"p25={f(np.percentile(arr, 25))} | "
            f"p5={f(np.percentile(arr, 5))} | "
            f"p1={f(np.percentile(arr, 1))} | "
            f"p95={f(np.percentile(arr, 95))} | "
            f"p99={f(np.percentile(arr, 99))} | "
            f"n={arr.size}"
        )

    @staticmethod
    def _moments_stats_text(
        arr: np.ndarray, value_fmt: str = "{:.3f}", unit: str = ""
    ) -> str:
        """``mean | std | min | max | n``.

        Used for the additive-error metrics (``angle_difference``,
        ``length_difference``) where mean encodes systematic bias and std
        encodes consistency — directly actionable when tuning a retargeter.
        Stddev is non-negative so the sign-prefix in ``value_fmt`` is
        stripped for that field only.
        """
        if arr.size == 0:
            return "(no data)"

        std_fmt = value_fmt.replace("+", "")  # std is always >= 0; drop sign prefix

        def signed(v: float) -> str:
            return f"{value_fmt.format(v)}{unit}"

        def unsigned(v: float) -> str:
            return f"{std_fmt.format(v)}{unit}"

        return (
            f"mean={signed(float(np.mean(arr)))} | "
            f"std={unsigned(float(np.std(arr)))} | "
            f"min={signed(float(np.min(arr)))} | "
            f"max={signed(float(np.max(arr)))} | "
            f"n={arr.size}"
        )

    # Backwards-compatible alias still used by the per-episode summary text
    # helpers below — they only need the cosine-similarity-style 5-number
    # summary.
    _summary_stats_text = _percentiles_stats_text

    @classmethod
    def _cos_sim_stats_text(cls, arr: np.ndarray) -> str:
        return cls._percentiles_stats_text(arr, value_fmt="{:.3f}", unit="")

    # ── Stats block (per-metric line) ────────────────────────────────────

    # The set of rows rendered in each card, their data key, display
    # label, format, unit, stats_variant, and section all come from
    # :func:`mimic_retargeter_lab.types.metrics.kv_metrics` — one ``MetricSpec``
    # per row, with kv-specific extras on ``MetricSpec.kv_detail``.
    # Edit a spec there and both this card and the comparison
    # dashboard pick up the change.
    _STATS_FORMATTERS = {
        "percentiles": "_percentiles_stats_text",
        "percentiles_both_tails": "_percentiles_both_tails_stats_text",
        "moments": "_moments_stats_text",
    }

    def _build_vector_card(self, ep_data: Dict[str, Any], vec_name: str) -> html.Div:
        """Card showing one keyvector's stats: header + 4 metric rows.

        Row ordering and per-row metadata (data key, display label,
        format, unit, stats variant, section) come from
        :func:`mimic_retargeter_lab.types.metrics.kv_metrics` — the registry is
        the single source of truth.
        """
        rows = []
        prev_section: str | None = None
        for spec in kv_metrics():
            kv = spec.kv_detail
            assert kv is not None, "kv_metrics() entries must carry kv_detail"
            # Insert a divider when the section flips (research → engineering).
            # Use a Div (not html.Hr) — Bootstrap's reboot.css restyles `<hr>`
            # to a translucent 1-px gray bar that renders almost invisibly
            # against our card background. A bordered Div sidesteps that.
            if prev_section is not None and kv.section != prev_section:
                rows.append(
                    html.Div(
                        style={
                            "margin": "10px 0",
                            "borderTop": "2px dashed #888",
                            "width": "100%",
                        },
                    )
                )
            prev_section = kv.section

            arr = self._vector_raw(ep_data, vec_name, kv.pkl_subkey)
            formatter = getattr(self, self._STATS_FORMATTERS[kv.stats_variant])
            # Wrap the bare format spec for ``str.format``-style usage.
            # ``kv.value_fmt`` overrides ``spec.fmt`` when a sign prefix
            # or other display tweak is needed for this metric only.
            value_fmt = "{:" + (kv.value_fmt or spec.fmt) + "}"
            text = formatter(arr, value_fmt=value_fmt, unit=kv.unit)
            rows.append(
                html.Div(
                    [
                        html.Span(
                            f"{kv.card_label}: ",
                            style={
                                "fontWeight": "600",
                                "color": "#333",
                                "minWidth": "200px",
                                "display": "inline-block",
                            },
                        ),
                        html.Span(
                            text,
                            style={"color": "#333", "fontFamily": "monospace"},
                        ),
                    ],
                    style={"padding": "2px 4px", "fontSize": "14px"},
                )
            )

        return html.Div(
            [
                html.Div(
                    vec_name,
                    style={
                        "fontWeight": "700",
                        "fontSize": "16px",
                        "padding": "0 0 8px 0",
                        "borderBottom": "1px solid #eee",
                        "marginBottom": "8px",
                    },
                ),
                *rows,
            ],
            style={
                "border": "1px solid #ddd",
                "borderRadius": "8px",
                "padding": "12px 16px",
                "boxSizing": "border-box",
                "backgroundColor": "#fafafa",
                "flex": "1 1 calc(50% - 16px)",
                "minWidth": "420px",
            },
        )

    # ── Layout ───────────────────────────────────────────────────────────

    def _build_episode_row_label(
        self,
        ep_id: str,
        cos_sim_arr: np.ndarray,
        scale_arr: np.ndarray,
    ) -> html.Div:
        """Episode-row summary: ``ep_id   cos_sim median | scale_ratio median | n``."""
        cos_med = f"{np.median(cos_sim_arr):.3f}" if cos_sim_arr.size else "—"
        scale_med = f"{np.median(scale_arr):.3f}" if scale_arr.size else "—"
        n = cos_sim_arr.size
        summary = (
            f"cos_sim (median)={cos_med} | "
            f"scale_ratio (robot/human, median)={scale_med} | "
            f"n={n}"
        )
        return html.Div(
            [
                html.Div(
                    ep_id,
                    style={
                        "fontWeight": "600",
                        "fontSize": "14px",
                        "whiteSpace": "nowrap",
                        "marginRight": "24px",
                    },
                ),
                html.Div(
                    summary,
                    style={
                        "fontFamily": "monospace",
                        "fontSize": "14px",
                        "color": "#333",
                        "whiteSpace": "nowrap",
                    },
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "row",
                "alignItems": "center",
                "width": "100%",
                "padding": "6px 0",
            },
        )

    def build_layout(self, metric_stats: Dict[str, Any]) -> html.Div:
        if not metric_stats:
            return html.Div([html.H3("No metric data available")])

        episode_ids = list(metric_stats.keys())
        if not episode_ids:
            return html.Div([html.H3("No episodes available")])

        first_ep = next(iter(metric_stats.values()))
        vector_names = self._vector_names(first_ep)
        if not vector_names:
            return html.Div([html.H3("No vector_metrics in episode data")])

        # Precompute pooled cos_sim and scale_ratio per episode (top section).
        pooled_cos_sim: Dict[str, np.ndarray] = {
            ep_id: self._pool_episode_cos_sim(metric_stats[ep_id])
            for ep_id in episode_ids
        }
        pooled_scale: Dict[str, np.ndarray] = {
            ep_id: self._pool_episode_scale_ratio(metric_stats[ep_id])
            for ep_id in episode_ids
        }

        # ── Top: per-episode radio with cos_sim/scale_ratio summary ────
        episode_options = [
            {
                "label": self._build_episode_row_label(
                    ep_id, pooled_cos_sim[ep_id], pooled_scale[ep_id]
                ),
                "value": ep_id,
            }
            for ep_id in episode_ids
        ]
        episode_radio = dcc.RadioItems(
            options=episode_options,
            value=episode_ids[0],
            id=f"kvm-episode-radio-{self.page_id}",
            labelStyle={
                "display": "flex",
                "alignItems": "center",
                "width": "100%",
                "borderBottom": "1px solid #eee",
                "padding": "4px 0",
            },
            inputStyle={"marginRight": "10px", "flexShrink": "0"},
            style={"width": "100%"},
        )

        section_heading_style = {
            "marginTop": "0",
            "marginBottom": "4px",
            "fontSize": "14px",
            "fontWeight": "600",
            "color": "#555",
            "textTransform": "uppercase",
            "letterSpacing": "0.04em",
        }

        # Cap the episode list at ~4 visible rows; subsequent episodes scroll.
        # One row is ~38px (radio + label padding + 1px bottom border).
        EPISODE_ROW_PX = 38
        EPISODE_VISIBLE_ROWS = 4
        episode_scroll_max = f"{EPISODE_ROW_PX * EPISODE_VISIBLE_ROWS}px"

        episode_section = html.Div(
            [
                html.H6("Episodes", style=section_heading_style),
                html.Div(
                    [episode_radio],
                    style={
                        "maxHeight": episode_scroll_max,
                        "overflowY": "auto",
                        "border": "1px solid #eee",
                        "borderRadius": "4px",
                    },
                ),
            ],
            style={"width": "100%", "marginBottom": "12px"},
        )

        # ── Bottom: title + per-vector card grid ───────────────────────
        bottom_section = html.Div(
            id=f"kvm-bottom-content-{self.page_id}",
            style={"width": "100%"},
        )

        layout = html.Div(
            [
                episode_section,
                html.Hr(style={"margin": "12px 0", "borderTop": "2px solid #ccc"}),
                bottom_section,
            ],
            style={
                "display": "flex",
                "flexDirection": "column",
                "width": "100%",
                "height": self.CONTENT_HEIGHT,
                "boxSizing": "border-box",
                "padding": "8px",
                "overflowY": "auto",
            },
        )

        # ── Callback: update bottom pane on episode change ────────────
        @self.app.callback(
            Output(f"kvm-bottom-content-{self.page_id}", "children"),
            [Input(f"kvm-episode-radio-{self.page_id}", "value")],
        )
        def _update_bottom(ep_id):
            if not ep_id:
                return html.Div("Select an episode", style={"padding": "12px"})
            ep = metric_stats.get(ep_id, {}) or {}
            cards = [self._build_vector_card(ep, v) for v in self._vector_names(ep)]
            return html.Div(
                [
                    html.H4(
                        f"Keyvector Matching (Dataset: {ep_id})",
                        style={"marginTop": "0", "marginBottom": "12px"},
                    ),
                    html.Div(
                        cards,
                        style={
                            "display": "flex",
                            "flexDirection": "row",
                            "flexWrap": "wrap",
                            "gap": "16px",
                            "alignItems": "stretch",
                        },
                    ),
                ],
                style={"display": "flex", "flexDirection": "column", "width": "100%"},
            )

        return layout
