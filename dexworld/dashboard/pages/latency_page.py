from typing import Any, Dict

import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots


class LatencyPage:
    """
    Dashboard page for the Latency metric.

    Layout: device badge on top, then a left-sidebar episode selector and a
    right pane that updates per selected episode. The right pane shows a
    title, a Statistics line, and a box-plot + histogram of the episode's
    latency distribution (with Mean / Median / p99 reference lines).
    """

    LEFT_COL_PX = 220
    CONTENT_HEIGHT = "calc(100vh - 120px)"

    def __init__(self, app, page_id: str):
        self.app = app
        self.page_id = page_id

    # ── Helpers ───────────────────────────────────────────────────────────

    def _device_badge(self, metric_stats: Dict[str, Any]) -> html.Div:
        devices = {ep["device"] for ep in metric_stats.values()}
        device_strs = {ep.get("device_str", "") for ep in metric_stats.values()}
        if len(devices) == 1:
            label = next(iter(devices))
            detail = next(iter(device_strs))
            text = f"Device: {label}" + (
                f" ({detail})" if detail and detail != label else ""
            )
            color = "#198754" if label == "gpu" else "#0d6efd"
        else:
            text = f"Device: mixed ({sorted(devices)})"
            color = "#dc3545"
        return html.Div(
            text,
            style={
                "backgroundColor": color,
                "color": "white",
                "padding": "8px 14px",
                "borderRadius": "6px",
                "marginBottom": "12px",
                "display": "inline-block",
                "fontWeight": "bold",
            },
        )

    @staticmethod
    def _stats_text(ep: Dict[str, Any]) -> str:
        """Single-line summary of an episode's latency stats (in ms)."""
        return (
            f"n={ep['num_timed']} | "
            f"mean={ep['mean_ms']:.3f} | "
            f"median={ep['median_ms']:.3f} | "
            f"stdev={ep['stdev_ms']:.3f} | "
            f"p95={ep['p95_ms']:.3f} | "
            f"p99={ep['p99_ms']:.3f} | "
            f"min={ep['min_ms']:.3f} | "
            f"max={ep['max_ms']:.3f} (ms)"
        )

    def _build_figure(self, latencies: list[float], episode_id: str) -> go.Figure:
        """Box plot + histogram for one episode's latency series."""
        if not latencies:
            return go.Figure()

        arr = np.asarray(latencies, dtype=float)
        mean_v = float(arr.mean())
        median_v = float(np.median(arr))
        p99_v = float(np.percentile(arr, 99))

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.18, 0.82],
            vertical_spacing=0.03,
        )

        fig.add_trace(
            go.Box(
                x=arr,
                orientation="h",
                name="",
                boxmean=True,
                marker_color="#0d6efd",
                line_color="#0d6efd",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Histogram(
                x=arr,
                histnorm="probability density",
                marker_color="#a8d4ff",
                marker_line_color="black",
                marker_line_width=1,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        line_specs = (
            ("Mean", mean_v, "#0d6efd", "dash"),
            ("Median", median_v, "#198754", "dashdot"),
            ("p99", p99_v, "#dc3545", "dot"),
        )
        for _name, value, color, dash in line_specs:
            fig.add_vline(
                x=value,
                line_dash=dash,
                line_color=color,
                line_width=2,
                row=2,
                col=1,
            )
        for name, value, color, dash in line_specs:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color=color, dash=dash, width=2),
                    name=f"{name}: {value:.3f}ms",
                    showlegend=True,
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            margin=dict(l=40, r=24, t=12, b=40),
            legend=dict(yanchor="top", y=0.95, xanchor="right", x=0.99),
            bargap=0.02,
        )
        fig.update_xaxes(title_text="Latency (ms)", row=2, col=1)
        fig.update_yaxes(title_text="Density", row=2, col=1)
        fig.update_yaxes(showticklabels=False, row=1, col=1)

        return fig

    # ── Layout ────────────────────────────────────────────────────────────

    def build_layout(self, metric_stats: Dict[str, Any]) -> html.Div:
        if not metric_stats:
            return html.Div([html.H3("No metric data available")])

        episode_ids = list(metric_stats.keys())

        # ── Precompute per-episode figures ─────────────────────────────
        episode_figs: Dict[str, go.Figure] = {
            ep_id: self._build_figure(
                list(metric_stats[ep_id].get("latencies_ms", [])), ep_id
            )
            for ep_id in episode_ids
        }

        # ── Sidebar: episode RadioItems (Collision-style) ─────────────
        episode_radio = dcc.RadioItems(
            options=[{"label": eid, "value": eid} for eid in episode_ids],
            value=episode_ids[0],
            id=f"lat-episode-radio-{self.page_id}",
            inputStyle={"marginRight": "8px"},
            style={
                "display": "flex",
                "flexDirection": "column",
                "overflowY": "auto",
                "maxHeight": self.CONTENT_HEIGHT,
                "paddingRight": "6px",
            },
            className="lat-inner-radio",
        )

        left_col_style = {
            "borderRight": "1px solid #ddd",
            "padding": "8px",
            "width": f"{self.LEFT_COL_PX}px",
            "minWidth": f"{self.LEFT_COL_PX}px",
            "maxWidth": f"{self.LEFT_COL_PX}px",
            "boxSizing": "border-box",
            "overflow": "hidden",
            "display": "flex",
            "flexDirection": "column",
            "height": "100%",
        }

        right_col_style = {
            "flex": "1 1 auto",
            "minWidth": "0",
            "display": "flex",
            "flexDirection": "column",
            "height": "100%",
            "boxSizing": "border-box",
            "paddingLeft": "12px",
            "overflowY": "auto",
        }

        sidebar = html.Div(
            [html.H6("Episodes", style={"marginTop": "0"}), episode_radio],
            style=left_col_style,
        )

        right_pane = html.Div(
            id=f"lat-content-{self.page_id}",
            style=right_col_style,
        )

        layout = html.Div(
            [
                self._device_badge(metric_stats),
                html.Div(
                    [sidebar, right_pane],
                    style={
                        "display": "flex",
                        "flexDirection": "row",
                        "flexWrap": "nowrap",
                        "alignItems": "stretch",
                        "width": "100%",
                        "height": self.CONTENT_HEIGHT,
                    },
                ),
            ],
            style={
                "padding": "12px",
                "boxSizing": "border-box",
                "height": self.CONTENT_HEIGHT,
                "display": "flex",
                "flexDirection": "column",
            },
        )

        # ── Callback: update right pane on episode change ─────────────
        @self.app.callback(
            Output(f"lat-content-{self.page_id}", "children"),
            [Input(f"lat-episode-radio-{self.page_id}", "value")],
        )
        def _update_content(ep_id):
            if not ep_id:
                return html.Div("Select an episode", style={"padding": "12px"})
            ep = metric_stats.get(ep_id, {})
            fig = episode_figs.get(ep_id, go.Figure())
            return html.Div(
                [
                    html.H4(
                        f"Latency — {ep_id}",
                        style={"marginTop": "0", "marginBottom": "8px"},
                    ),
                    html.Div(
                        [
                            html.Span(
                                "Statistics: ",
                                style={"fontWeight": "600", "fontSize": "20px"},
                            ),
                            html.Span(
                                self._stats_text(ep),
                                style={"color": "#333", "fontSize": "18px"},
                            ),
                        ],
                        style={"padding": "8px 4px"},
                    ),
                    dcc.Graph(
                        figure=fig,
                        style={"height": "calc(100vh - 280px)", "width": "100%"},
                        config={"responsive": True},
                    ),
                ],
                style={"display": "flex", "flexDirection": "column", "width": "100%"},
            )

        return layout
