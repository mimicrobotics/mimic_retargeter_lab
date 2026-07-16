from typing import Any, Dict, List

import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output


class MotionPreservationPage:
    """
    Dashboard page for the Motion Preservation metric.

    Layout (top → bottom):

        ┌──────────────────────────────────────────────────────────┐
        │ EPISODES                                                  │
        │ ◉ ep_id  n=…  mean=…  …    [horizontal box plot]          │
        │ ○ ep_id  n=…  mean=…  …    [horizontal box plot]          │
        ├──────────────────────────────────────────────────────────┤
        │ FRAMES (for the selected episode)                         │
        │ ◉ index_tip   ┌──────────────────────────┐                │
        │ ○ middle_tip  │ box plot for (ep, frame) │                │
        │ ○ ring_tip    └──────────────────────────┘                │
        │ ○ pinky_tip   ┌──────────────────────────┐                │
        │ ○ thumb_tip   │ alignment vs timesteps   │                │
        │               └──────────────────────────┘                │
        └──────────────────────────────────────────────────────────┘
    """

    LEFT_COL_PX = 220
    CONTENT_HEIGHT = "calc(100vh - 120px)"

    def __init__(self, app, page_id: str):
        self.app = app
        self.page_id = page_id

    # ── Helpers ───────────────────────────────────────────────────────────

    def _to_list_safe(self, v) -> List[Any]:
        if v is None:
            return []
        try:
            return list(v)
        except Exception:
            return [v]

    def _pool_episode_alignment(self, ep_data: Dict[str, Any]) -> np.ndarray:
        """Concat per-frame `pos_alignment.raw` arrays across all frames."""
        pooled: list = []
        for frame_name in sorted(ep_data.keys()):
            align = ep_data[frame_name].get("pos_alignment", {}) or {}
            raw = align.get("raw")
            if raw is None:
                continue
            pooled.extend(list(raw))
        if not pooled:
            return np.array([], dtype=float)
        arr = np.asarray(pooled, dtype=float)
        return arr[~np.isnan(arr)]

    def _frame_alignment(self, ep_data: Dict[str, Any], frame_name: str) -> np.ndarray:
        align = ep_data.get(frame_name, {}).get("pos_alignment", {}) or {}
        raw = align.get("raw")
        if raw is None:
            return np.array([], dtype=float)
        arr = np.asarray(list(raw), dtype=float)
        return arr[~np.isnan(arr)]

    @staticmethod
    def _stats_text(arr: np.ndarray) -> str:
        if arr.size == 0:
            return "(no data)"
        # Order: central tendency + IQR, then low-tail worst-case percentiles.
        return (
            f"median={np.median(arr):.3f} | "
            f"p75={np.percentile(arr, 75):.3f} | "
            f"p25={np.percentile(arr, 25):.3f} | "
            f"p5={np.percentile(arr, 5):.3f} | "
            f"p1={np.percentile(arr, 1):.3f} | "
            f"n={arr.size}"
        )

    def _build_box_figure(self, arr: np.ndarray, color: str = "#0d6efd") -> go.Figure:
        """Compact horizontal box plot suitable for embedding in a row."""
        fig = go.Figure()
        if arr.size == 0:
            fig.update_layout(margin=dict(l=8, r=8, t=8, b=8), height=60)
            return fig
        fig.add_trace(
            go.Box(
                x=arr,
                orientation="h",
                boxmean=True,
                marker_color=color,
                line_color=color,
                showlegend=False,
            )
        )
        fig.update_layout(
            margin=dict(l=8, r=8, t=8, b=8),
            xaxis=dict(range=[-1.05, 1.05], title="cosine similarity"),
            yaxis=dict(showticklabels=False),
            height=80,
        )
        return fig

    def _build_line_figure(
        self, ep_data: Dict[str, Any], ep_id: str, frame_name: str
    ) -> go.Figure:
        """Per-frame alignment-vs-timesteps line for a single episode."""
        fig = go.Figure()
        align = ep_data.get(frame_name, {}).get("pos_alignment") or {}
        smoothed = self._to_list_safe(align.get("smoothed"))
        mean_val = align.get("mean", float("nan"))
        if smoothed:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(smoothed))),
                    y=smoothed,
                    mode="lines",
                    name=f"{ep_id} (mean={mean_val:.3f})",
                    line=dict(color="#0d6efd"),
                )
            )
        fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="green",
            opacity=0.5,
            annotation_text="perfect",
            annotation_position="top left",
        )
        fig.update_layout(
            title=f"{frame_name} — directional alignment ({ep_id})",
            xaxis_title="timesteps",
            yaxis_title="cosine similarity",
            yaxis_range=[-1.1, 1.1],
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            margin=dict(l=40, r=24, t=36, b=40),
        )
        return fig

    # ── Layout ────────────────────────────────────────────────────────────

    def _build_episode_row_label(self, ep_id: str, arr: np.ndarray) -> html.Div:
        """Component label for a single episode option in the radio.

        Layout: [ep_id, full width] [stats text, separated by spacing].
        """
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
                    self._stats_text(arr),
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

        # Frame names assumed consistent across episodes; pick from first.
        frame_names = sorted(next(iter(metric_stats.values())).keys())
        if not frame_names:
            return html.Div([html.H3("No frame data available")])

        # ── Precompute pooled alignments + per-(ep, frame) figures ─────
        episode_pooled: Dict[str, np.ndarray] = {
            ep_id: self._pool_episode_alignment(metric_stats[ep_id])
            for ep_id in episode_ids
        }
        frame_box_figs: Dict[tuple, go.Figure] = {}
        line_figs: Dict[tuple, go.Figure] = {}
        for ep_id in episode_ids:
            for fn in frame_names:
                arr = self._frame_alignment(metric_stats[ep_id], fn)
                frame_box_figs[(ep_id, fn)] = self._build_box_figure(
                    arr, color="#198754"
                )
                line_figs[(ep_id, fn)] = self._build_line_figure(
                    metric_stats[ep_id], ep_id, fn
                )

        # ── Top: per-episode radio with embedded stats text ───────────
        episode_options = [
            {
                "label": self._build_episode_row_label(ep_id, episode_pooled[ep_id]),
                "value": ep_id,
            }
            for ep_id in episode_ids
        ]
        episode_radio = dcc.RadioItems(
            options=episode_options,
            value=episode_ids[0],
            id=f"mp-episode-radio-{self.page_id}",
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

        # ── Bottom: frame radio + dynamic plots ───────────────────────
        frame_radio = dcc.RadioItems(
            options=[{"label": fn, "value": fn} for fn in frame_names],
            value=frame_names[0],
            id=f"mp-frame-radio-{self.page_id}",
            inputStyle={"marginRight": "8px"},
            style={
                "display": "flex",
                "flexDirection": "column",
                "paddingRight": "6px",
            },
            className="mp-inner-radio",
        )

        left_col_style = {
            "borderRight": "1px solid #ddd",
            "padding": "8px",
            "width": f"{self.LEFT_COL_PX}px",
            "minWidth": f"{self.LEFT_COL_PX}px",
            "maxWidth": f"{self.LEFT_COL_PX}px",
            "boxSizing": "border-box",
            "display": "flex",
            "flexDirection": "column",
        }

        right_col_style = {
            "flex": "1 1 auto",
            "minWidth": "0",
            "display": "flex",
            "flexDirection": "column",
            "boxSizing": "border-box",
            "paddingLeft": "12px",
        }

        bottom_section = html.Div(
            [
                html.H6("Frames", style={**section_heading_style, "marginTop": "12px"}),
                html.Div(
                    [
                        html.Div([frame_radio], style=left_col_style),
                        html.Div(
                            id=f"mp-bottom-content-{self.page_id}",
                            style=right_col_style,
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexDirection": "row",
                        "alignItems": "stretch",
                        "width": "100%",
                    },
                ),
            ],
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

        # ── Callback: update bottom pane on (ep, frame) change ───────
        @self.app.callback(
            Output(f"mp-bottom-content-{self.page_id}", "children"),
            [
                Input(f"mp-episode-radio-{self.page_id}", "value"),
                Input(f"mp-frame-radio-{self.page_id}", "value"),
            ],
        )
        def _update_bottom(ep_id, frame_name):
            if not ep_id or not frame_name:
                return html.Div(
                    "Select an episode and a frame", style={"padding": "12px"}
                )
            box_fig = frame_box_figs.get((ep_id, frame_name), go.Figure())
            line_fig = line_figs.get((ep_id, frame_name), go.Figure())
            arr = self._frame_alignment(metric_stats[ep_id], frame_name)
            return html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "Statistics: ",
                                style={"fontWeight": "600", "fontSize": "20px"},
                            ),
                            html.Span(
                                self._stats_text(arr),
                                style={"color": "#333", "fontSize": "18px"},
                            ),
                        ],
                        style={"padding": "8px 4px"},
                    ),
                    dcc.Graph(
                        figure=box_fig,
                        style={"height": "120px", "width": "100%"},
                        config={"displayModeBar": False, "responsive": True},
                    ),
                    dcc.Graph(
                        figure=line_fig,
                        style={"height": "calc(100vh - 480px)", "width": "100%"},
                        config={"responsive": True},
                    ),
                ],
                style={"display": "flex", "flexDirection": "column", "width": "100%"},
            )

        return layout
