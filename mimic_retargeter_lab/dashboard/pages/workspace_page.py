from typing import Any, Dict

import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output


class WorkspacePage:
    """
    Dashboard page for the Workspace metric.

    Layout (top → bottom):

        ┌──────────────────────────────────────────────────────────┐
        │ EPISODES                                                  │
        │ ◉ ep_id   mean_chamfer=…  thumb=…  index=…  middle=…  …   │
        │ ○ ep_id   mean_chamfer=…  thumb=…  index=…  middle=…  …   │
        ├──────────────────────────────────────────────────────────┤
        │ FRAMES (for the selected episode)                         │
        │ ◉ thumb_tip   Statistics for (ep, frame): chamfer, util   │
        │ ○ index_tip   ┌──────────────────────────────────────┐    │
        │ ○ middle_tip  │  3D scatter — human (gray) vs robot   │    │
        │ ○ ring_tip    │  (blue) workspace point clouds        │    │
        │ ○ pinky_tip   └──────────────────────────────────────┘    │
        └──────────────────────────────────────────────────────────┘
    """

    LEFT_COL_PX = 220
    CONTENT_HEIGHT = "calc(100vh - 120px)"

    HUMAN_COLOR = "#7f7f7f"
    ROBOT_COLOR = "#1f77b4"

    def __init__(self, app, page_id: str):
        self.app = app
        self.page_id = page_id

    # ── Helpers ───────────────────────────────────────────────────────────

    def _build_workspace_figure(
        self,
        human_pts: np.ndarray | None,
        robot_pts: np.ndarray | None,
        frame_name: str,
        ep_id: str,
        util_data: Dict[str, Any] | None = None,
    ) -> go.Figure:
        """3D scatter combining human + robot workspace point clouds."""
        fig = go.Figure()

        for pts, color, marker_size, opacity, name in (
            (human_pts, self.HUMAN_COLOR, 3, 0.6, f"{frame_name} (human)"),
            (robot_pts, self.ROBOT_COLOR, 4, 0.9, f"{frame_name} (robot)"),
        ):
            if pts is None:
                continue
            arr = np.asarray(pts)
            if arr.size == 0:
                continue
            if arr.ndim == 1 and arr.shape[0] == 3:
                arr = arr.reshape(1, 3)
            elif arr.ndim == 2 and arr.shape[1] != 3:
                try:
                    arr = arr.reshape(-1, 3)
                except Exception:
                    continue
            fig.add_trace(
                go.Scatter3d(
                    x=arr[:, 0],
                    y=arr[:, 1],
                    z=arr[:, 2],
                    mode="markers",
                    marker=dict(size=marker_size, color=color, opacity=opacity),
                    name=name,
                )
            )

        fig.update_layout(
            scene=dict(
                xaxis=dict(title="X"),
                yaxis=dict(title="Y"),
                zaxis=dict(title="Z"),
                aspectmode="data",
                camera=dict(projection=dict(type="orthographic")),
            ),
            margin=dict(l=0, r=0, t=8, b=0),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.7,
                xanchor="right",
                x=0.99,
                font=dict(size=16),
                bgcolor="rgba(255,255,255,0.7)",
            ),
        )

        # In-figure annotation with workspace utilization (top-left of plot area).
        if util_data:
            util_pct = float(util_data.get("utilization", 0.0)) * 100
            hits = util_data.get("hits", "?")
            num_samples = util_data.get("num_samples", "?")
            radius = util_data.get("radius", float("nan"))
            fig.add_annotation(
                text=(
                    f"<b>Workspace utilization: {util_pct:.2f}%</b><br>"
                    f"{hits}/{num_samples} samples covered<br>"
                    f"(radius {radius} m)"
                ),
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                showarrow=False,
                align="left",
                font=dict(size=14, color="#222"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#666",
                borderwidth=1,
                borderpad=8,
            )

        return fig

    @staticmethod
    def _episode_utilization_text(utilization: Dict[str, Dict[str, Any]]) -> str:
        """Per-episode utilization summary: `mean=…% | thumb=…% | index=…% | …`."""
        if not utilization:
            return "(no utilization data)"
        per_frame_vals = {
            name: float(d.get("utilization", float("nan"))) * 100
            for name, d in utilization.items()
            if isinstance(d, dict)
        }
        valid = [v for v in per_frame_vals.values() if not np.isnan(v)]
        mean_val = float(np.mean(valid)) if valid else float("nan")
        per_frame = " | ".join(
            f"{name.replace('_tip', '')}={val:.2f}%"
            for name, val in per_frame_vals.items()
        )
        return f"mean={mean_val:.2f}% | {per_frame}"

    @staticmethod
    def _frame_stats_text(util_data: Dict[str, Any] | None) -> str:
        if not util_data:
            return "utilization=(no data)"
        util_pct = float(util_data.get("utilization", float("nan"))) * 100
        hits = util_data.get("hits", "?")
        num_samples = util_data.get("num_samples", "?")
        radius = util_data.get("radius", float("nan"))
        return (
            f"utilization={util_pct:.2f}% "
            f"({hits}/{num_samples} samples) | "
            f"radius={radius} m"
        )

    # ── Layout ────────────────────────────────────────────────────────────

    def _build_episode_row_label(
        self, ep_id: str, utilization: Dict[str, Dict[str, Any]]
    ) -> html.Div:
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
                    self._episode_utilization_text(utilization),
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

        episode_ids = sorted(metric_stats.keys())
        if not episode_ids:
            return html.Div([html.H3("No episodes available")])

        # Frame names assumed consistent across episodes; pick from first.
        first_ep = metric_stats[episode_ids[0]]
        frame_names = list((first_ep.get("workspace_pts") or {}).keys())
        if not frame_names:
            return html.Div([html.H3("No frame data available")])

        # ── Precompute per-(ep, frame) figures ─────────────────────────
        workspace_figs: Dict[tuple, go.Figure] = {}
        for ep_id in episode_ids:
            ep = metric_stats[ep_id]
            ws_pts = ep.get("workspace_pts") or {}
            util = ep.get("utilization") or {}
            for fn in frame_names:
                fd = ws_pts.get(fn) or {}
                workspace_figs[(ep_id, fn)] = self._build_workspace_figure(
                    fd.get("human"),
                    fd.get("robot"),
                    fn,
                    ep_id,
                    util_data=util.get(fn),
                )

        # ── Top: per-episode radio with embedded utilization summary ─
        episode_options = [
            {
                "label": self._build_episode_row_label(
                    ep_id, metric_stats[ep_id].get("utilization", {}) or {}
                ),
                "value": ep_id,
            }
            for ep_id in episode_ids
        ]
        episode_radio = dcc.RadioItems(
            options=episode_options,
            value=episode_ids[0],
            id=f"ws-episode-radio-{self.page_id}",
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
                html.H6("Episodes (utilization)", style=section_heading_style),
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

        # ── Bottom: frame radio + dynamic plot ────────────────────────
        frame_radio = dcc.RadioItems(
            options=[{"label": fn, "value": fn} for fn in frame_names],
            value=frame_names[0],
            id=f"ws-frame-radio-{self.page_id}",
            inputStyle={"marginRight": "8px"},
            style={
                "display": "flex",
                "flexDirection": "column",
                "paddingRight": "6px",
            },
            className="ws-inner-radio",
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
                            id=f"ws-bottom-content-{self.page_id}",
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
            Output(f"ws-bottom-content-{self.page_id}", "children"),
            [
                Input(f"ws-episode-radio-{self.page_id}", "value"),
                Input(f"ws-frame-radio-{self.page_id}", "value"),
            ],
        )
        def _update_bottom(ep_id, frame_name):
            if not ep_id or not frame_name:
                return html.Div(
                    "Select an episode and a frame", style={"padding": "12px"}
                )
            ep = metric_stats.get(ep_id, {}) or {}
            util_data = (ep.get("utilization") or {}).get(frame_name)
            fig = workspace_figs.get((ep_id, frame_name), go.Figure())
            return html.Div(
                [
                    html.H4(
                        f"Workspace — {ep_id} : {frame_name}",
                        style={"marginTop": "0", "marginBottom": "8px"},
                    ),
                    html.Div(
                        [
                            html.Span(
                                "Statistics: ",
                                style={"fontWeight": "600", "fontSize": "20px"},
                            ),
                            html.Span(
                                self._frame_stats_text(util_data),
                                style={"color": "#333", "fontSize": "18px"},
                            ),
                        ],
                        style={"padding": "6px 4px"},
                    ),
                    dcc.Graph(
                        figure=fig,
                        style={"height": "calc(100vh - 360px)", "width": "100%"},
                        config={"responsive": True},
                    ),
                ],
                style={"display": "flex", "flexDirection": "column", "width": "100%"},
            )

        return layout
