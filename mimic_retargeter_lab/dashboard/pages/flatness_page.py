from typing import Any, Dict, List

import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output

from mimic_retargeter_lab.types.metrics import Metric, metric_spec


# Y-axis title for the per-frame ``‖accel‖²`` plots. Pulled from the
# canonical registry so a label rename in
# :data:`mimic_retargeter_lab.types.metrics.METRICS` propagates here automatically.
_FLATNESS_Y_AXIS_TITLE = f"{metric_spec(Metric.FLATNESS).yaxis_label} (log scale)"


class FlatnessPage:
    """
    Dashboard page for the Flatness metric.

    Layout (top → bottom):

        ┌──────────────────────────────────────────────────────────┐
        │ EPISODES                                                  │
        │ ◉ ep_id    human=<median> | robot=<median>                │
        │ ○ ep_id    human=<median> | robot=<median>                │
        ├──────────────────────────────────────────────────────────┤
        │ FRAMES (for the selected episode)                         │
        │ ◉ thumb_tip   ┌──────────────────────────────────────┐    │
        │ ○ index_tip   │ box plot: human + robot accel²       │    │
        │ ○ middle_tip  └──────────────────────────────────────┘    │
        │ ○ ring_tip    Statistics (Robot / Human)                  │
        │ ○ pinky_tip   ┌──────────────────────────────────────┐    │
        │               │ accel² vs timesteps (log y)          │    │
        │               └──────────────────────────────────────┘    │
        └──────────────────────────────────────────────────────────┘

    Lower is better for flatness, so the worst-case percentiles surfaced
    are p95 and p99 (high-tail spikes in squared acceleration).
    """

    LEFT_COL_PX = 220
    CONTENT_HEIGHT = "calc(100vh - 120px)"

    HUMAN_COLOR = "#1f77b4"
    ROBOT_COLOR = "#d62728"

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

    def _frame_accel(
        self, ep_data: Dict[str, Any], frame_name: str, embodiment: str
    ) -> np.ndarray:
        """Return raw `accel_norm_sq` for one (frame, embodiment)."""
        data = ep_data.get(frame_name, {}).get(embodiment, {}) or {}
        raw = data.get("accel_norm_sq")
        if raw is None:
            return np.array([], dtype=float)
        arr = np.asarray(list(raw), dtype=float)
        return arr[~np.isnan(arr)]

    def _pool_episode_accel(
        self, ep_data: Dict[str, Any], embodiment: str
    ) -> np.ndarray:
        """Concat per-frame `<embodiment>.accel_norm_sq` across all frames."""
        pooled: list = []
        for frame_name in sorted(ep_data.keys()):
            data = ep_data[frame_name].get(embodiment, {}) or {}
            raw = data.get("accel_norm_sq")
            if raw is None:
                continue
            pooled.extend(list(raw))
        if not pooled:
            return np.array([], dtype=float)
        arr = np.asarray(pooled, dtype=float)
        return arr[~np.isnan(arr)]

    @staticmethod
    def _stats_text(arr: np.ndarray) -> str:
        if arr.size == 0:
            return "(no data)"
        # Order: central tendency + IQR, then high-tail worst-case percentiles.
        return (
            f"median={np.median(arr):.3e} | "
            f"p75={np.percentile(arr, 75):.3e} | "
            f"p25={np.percentile(arr, 25):.3e} | "
            f"p95={np.percentile(arr, 95):.3e} | "
            f"p99={np.percentile(arr, 99):.3e} | "
            f"n={arr.size}"
        )

    def _build_box_figure(
        self, robot_arr: np.ndarray, human_arr: np.ndarray
    ) -> go.Figure:
        """Horizontal box plot with two boxes (robot + human) on a log axis."""
        fig = go.Figure()
        if robot_arr.size:
            fig.add_trace(
                go.Box(
                    x=robot_arr,
                    orientation="h",
                    name="robot",
                    boxmean=True,
                    marker_color=self.ROBOT_COLOR,
                    line_color=self.ROBOT_COLOR,
                    showlegend=False,
                )
            )
        if human_arr.size:
            fig.add_trace(
                go.Box(
                    x=human_arr,
                    orientation="h",
                    name="human",
                    boxmean=True,
                    marker_color=self.HUMAN_COLOR,
                    line_color=self.HUMAN_COLOR,
                    showlegend=False,
                )
            )
        fig.update_layout(
            margin=dict(l=8, r=8, t=8, b=32),
            xaxis=dict(
                title=_FLATNESS_Y_AXIS_TITLE,
                type="log",
                exponentformat="power",
                showexponent="all",
            ),
            height=160,
        )
        return fig

    def _build_line_figure(
        self, ep_data: Dict[str, Any], ep_id: str, frame_name: str
    ) -> go.Figure:
        """Per-frame ‖accel‖² vs timesteps line for a single episode."""
        fig = go.Figure()
        frame_data = ep_data.get(frame_name, {}) or {}
        for embodiment, color in (
            ("human", self.HUMAN_COLOR),
            ("robot", self.ROBOT_COLOR),
        ):
            data = frame_data.get(embodiment) or {}
            smoothed = self._to_list_safe(data.get("smoothed"))
            mean_val = data.get("mean", float("nan"))
            if smoothed:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(smoothed))),
                        y=smoothed,
                        mode="lines",
                        name=f"{embodiment} (mean={mean_val:.2e})",
                        line=dict(color=color),
                    )
                )
        fig.update_layout(
            title=f"{frame_name} — flatness ({ep_id})",
            xaxis_title="timesteps",
            yaxis=dict(
                title=_FLATNESS_Y_AXIS_TITLE,
                type="log",
                exponentformat="power",
                showexponent="all",
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            margin=dict(l=40, r=24, t=36, b=40),
        )
        return fig

    # ── Layout ────────────────────────────────────────────────────────────

    def _build_episode_row_label(
        self, ep_id: str, human_arr: np.ndarray, robot_arr: np.ndarray
    ) -> html.Div:
        """Per-episode radio-row label: `ep_id   human=<median> | robot=<median>`."""

        def fmt(arr: np.ndarray) -> str:
            return f"{np.median(arr):.3e}" if arr.size else "—"

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
                    [
                        html.Span(
                            "human=",
                            style={"color": self.HUMAN_COLOR, "fontWeight": "600"},
                        ),
                        html.Span(fmt(human_arr), style={"marginRight": "10px"}),
                        html.Span("|", style={"color": "#999", "marginRight": "10px"}),
                        html.Span(
                            "robot=",
                            style={"color": self.ROBOT_COLOR, "fontWeight": "600"},
                        ),
                        html.Span(fmt(robot_arr)),
                    ],
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

        frame_names = sorted(next(iter(metric_stats.values())).keys())
        if not frame_names:
            return html.Div([html.H3("No frame data available")])

        # ── Precompute pooled accel + per-(ep, frame) figures ──────────
        episode_human_pooled: Dict[str, np.ndarray] = {
            ep_id: self._pool_episode_accel(metric_stats[ep_id], "human")
            for ep_id in episode_ids
        }
        episode_robot_pooled: Dict[str, np.ndarray] = {
            ep_id: self._pool_episode_accel(metric_stats[ep_id], "robot")
            for ep_id in episode_ids
        }
        frame_box_figs: Dict[tuple, go.Figure] = {}
        line_figs: Dict[tuple, go.Figure] = {}
        for ep_id in episode_ids:
            for fn in frame_names:
                robot_arr = self._frame_accel(metric_stats[ep_id], fn, "robot")
                human_arr = self._frame_accel(metric_stats[ep_id], fn, "human")
                frame_box_figs[(ep_id, fn)] = self._build_box_figure(
                    robot_arr, human_arr
                )
                line_figs[(ep_id, fn)] = self._build_line_figure(
                    metric_stats[ep_id], ep_id, fn
                )

        # ── Top: per-episode radio with embedded stats text ───────────
        episode_options = [
            {
                "label": self._build_episode_row_label(
                    ep_id,
                    episode_human_pooled[ep_id],
                    episode_robot_pooled[ep_id],
                ),
                "value": ep_id,
            }
            for ep_id in episode_ids
        ]
        episode_radio = dcc.RadioItems(
            options=episode_options,
            value=episode_ids[0],
            id=f"fl-episode-radio-{self.page_id}",
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
            id=f"fl-frame-radio-{self.page_id}",
            inputStyle={"marginRight": "8px"},
            style={
                "display": "flex",
                "flexDirection": "column",
                "paddingRight": "6px",
            },
            className="fl-inner-radio",
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
                            id=f"fl-bottom-content-{self.page_id}",
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
            Output(f"fl-bottom-content-{self.page_id}", "children"),
            [
                Input(f"fl-episode-radio-{self.page_id}", "value"),
                Input(f"fl-frame-radio-{self.page_id}", "value"),
            ],
        )
        def _update_bottom(ep_id, frame_name):
            if not ep_id or not frame_name:
                return html.Div(
                    "Select an episode and a frame", style={"padding": "12px"}
                )
            box_fig = frame_box_figs.get((ep_id, frame_name), go.Figure())
            line_fig = line_figs.get((ep_id, frame_name), go.Figure())
            robot_arr = self._frame_accel(metric_stats[ep_id], frame_name, "robot")
            human_arr = self._frame_accel(metric_stats[ep_id], frame_name, "human")
            return html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "Statistics (human): ",
                                style={
                                    "fontWeight": "600",
                                    "fontSize": "20px",
                                    "color": self.HUMAN_COLOR,
                                },
                            ),
                            html.Span(
                                self._stats_text(human_arr),
                                style={"color": "#333", "fontSize": "18px"},
                            ),
                        ],
                        style={"padding": "0 4px 8px 4px"},
                    ),
                    html.Div(
                        [
                            html.Span(
                                "Statistics (robot): ",
                                style={
                                    "fontWeight": "600",
                                    "fontSize": "20px",
                                    "color": self.ROBOT_COLOR,
                                },
                            ),
                            html.Span(
                                self._stats_text(robot_arr),
                                style={"color": "#333", "fontSize": "18px"},
                            ),
                        ],
                        style={"padding": "6px 4px"},
                    ),
                    dcc.Graph(
                        figure=box_fig,
                        style={"height": "180px", "width": "100%"},
                        config={"displayModeBar": False, "responsive": True},
                    ),
                    dcc.Graph(
                        figure=line_fig,
                        style={"height": "calc(100vh - 540px)", "width": "100%"},
                        config={"responsive": True},
                    ),
                ],
                style={"display": "flex", "flexDirection": "column", "width": "100%"},
            )

        return layout
