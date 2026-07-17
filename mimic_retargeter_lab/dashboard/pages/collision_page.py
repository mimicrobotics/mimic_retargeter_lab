from typing import Any, Dict

import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output


class CollisionPage:
    """
    Dashboard page for the Collision metric.

    Shows summary stats (max/avg penetration, unique pairs, collision rate)
    and a per-frame max penetration depth time series, with a colliding-pair
    breakdown table. Episodes are selectable via a left sidebar.
    """

    LEFT_COL_PX = 220
    CONTENT_HEIGHT = "calc(100vh - 120px)"

    def __init__(self, app, page_id: str):
        self.app = app
        self.page_id = page_id

    def _build_tolerance_banner(self, ep_data: Dict[str, Any]) -> html.Div:
        tolerance = ep_data.get("penetration_tolerance_mm", 0.0)
        return html.Div(
            f"Penetration tolerance: {tolerance:.1f} mm "
            f"(contacts with depth \u2264 {tolerance:.1f} mm are ignored)",
            style={
                "backgroundColor": "#fff3cd",
                "border": "1px solid #ffc107",
                "borderRadius": "6px",
                "padding": "8px 14px",
                "marginBottom": "12px",
                "color": "#664d03",
            },
        )

    def _build_summary_cards(self, ep_data: Dict[str, Any]) -> html.Div:
        cards = [
            ("Max Penetration Depth", f"{ep_data['max_penetration_depth_mm']:.2f} mm"),
            ("Avg Penetration Depth", f"{ep_data['avg_penetration_depth_mm']:.2f} mm"),
            ("Unique Colliding Pairs", str(ep_data["num_unique_colliding_pairs"])),
            (
                "Collision Rate",
                f"{ep_data['collision_rate_pct']:.1f}% "
                f"({ep_data['frames_with_collision']}/{ep_data['num_frames']} frames)",
            ),
        ]

        card_style = {
            "border": "1px solid #ddd",
            "borderRadius": "8px",
            "padding": "16px",
            "minWidth": "180px",
            "flex": "1 1 180px",
            "textAlign": "center",
            "backgroundColor": "#f9f9f9",
        }

        return html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            label,
                            style={
                                "color": "#666",
                                "marginBottom": "4px",
                            },
                        ),
                        html.Div(
                            value,
                            style={"fontWeight": "bold"},
                        ),
                    ],
                    style=card_style,
                )
                for label, value in cards
            ],
            style={
                "display": "flex",
                "gap": "12px",
                "flexWrap": "wrap",
                "marginBottom": "16px",
            },
        )

    def _build_timeseries_figure(self, ep_data: Dict[str, Any]) -> go.Figure:
        per_frame = ep_data["per_frame_max_depth"]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(per_frame))),
                y=list(per_frame),
                mode="lines",
                name="Max penetration depth",
                line=dict(color="red"),
                fill="tozeroy",
                fillcolor="rgba(255, 0, 0, 0.1)",
            )
        )
        fig.update_layout(
            title="Per-Frame Max Penetration Depth",
            xaxis_title="Frame",
            yaxis_title="Penetration depth (mm)",
            margin=dict(l=40, r=24, t=36, b=40),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        return fig

    def _build_pair_table(self, ep_data: Dict[str, Any]) -> html.Div:
        collision_pairs = ep_data.get("collision_pairs", {})
        if not collision_pairs:
            return html.Div(
                "No collisions detected.",
                style={"padding": "12px", "color": "#666"},
            )

        sorted_pairs = sorted(
            collision_pairs.items(), key=lambda x: -x[1]["max_depth_mm"]
        )

        header = html.Tr(
            [
                html.Th("Geom 1", style={"textAlign": "left", "padding": "6px 12px"}),
                html.Th("Geom 2", style={"textAlign": "left", "padding": "6px 12px"}),
                html.Th("Count", style={"textAlign": "right", "padding": "6px 12px"}),
                html.Th(
                    "Max Depth (mm)",
                    style={"textAlign": "right", "padding": "6px 12px"},
                ),
                html.Th(
                    "Avg Depth (mm)",
                    style={"textAlign": "right", "padding": "6px 12px"},
                ),
            ]
        )

        rows = []
        for (g1, g2), info in sorted_pairs:
            rows.append(
                html.Tr(
                    [
                        html.Td(
                            g1, style={"padding": "4px 12px", "fontFamily": "monospace"}
                        ),
                        html.Td(
                            g2, style={"padding": "4px 12px", "fontFamily": "monospace"}
                        ),
                        html.Td(
                            str(info["count"]),
                            style={"textAlign": "right", "padding": "4px 12px"},
                        ),
                        html.Td(
                            f"{info['max_depth_mm']:.2f}",
                            style={"textAlign": "right", "padding": "4px 12px"},
                        ),
                        html.Td(
                            f"{info['avg_depth_mm']:.2f}",
                            style={"textAlign": "right", "padding": "4px 12px"},
                        ),
                    ]
                )
            )

        return html.Div(
            [
                html.H6("Colliding Pairs", style={"marginTop": "8px"}),
                html.Table(
                    [html.Thead(header), html.Tbody(rows)],
                    style={
                        "width": "100%",
                        "borderCollapse": "collapse",
                    },
                ),
            ]
        )

    def build_layout(self, metric_stats: Dict[str, Any]) -> html.Div:
        if not metric_stats:
            return html.Div([html.H3("No metric data available")])

        episode_ids = sorted(metric_stats.keys())

        episode_radio = dcc.RadioItems(
            options=[{"label": eid, "value": eid} for eid in episode_ids],
            value=episode_ids[0],
            id=f"col-radio-{self.page_id}",
            inputStyle={"marginRight": "8px"},
            style={
                "display": "flex",
                "flexDirection": "column",
                "overflowY": "auto",
                "maxHeight": self.CONTENT_HEIGHT,
                "paddingRight": "6px",
            },
            className="col-inner-radio",
        )

        # Pre-build content for each episode
        episode_content = {}
        for eid in episode_ids:
            ep_data = metric_stats[eid]
            episode_content[eid] = html.Div(
                [
                    self._build_tolerance_banner(ep_data),
                    self._build_summary_cards(ep_data),
                    dcc.Graph(
                        figure=self._build_timeseries_figure(ep_data),
                        style={"height": "350px", "width": "100%"},
                        config={"responsive": True},
                    ),
                    self._build_pair_table(ep_data),
                ]
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

        layout = html.Div(
            [
                html.Div(
                    [html.H6("Episodes", style={"marginTop": "0"}), episode_radio],
                    style=left_col_style,
                ),
                html.Div(
                    [html.Div(id=f"col-content-{self.page_id}")],
                    style=right_col_style,
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "row",
                "flexWrap": "nowrap",
                "alignItems": "stretch",
                "width": "100%",
                "height": self.CONTENT_HEIGHT,
            },
        )

        @self.app.callback(
            Output(f"col-content-{self.page_id}", "children"),
            [Input(f"col-radio-{self.page_id}", "value")],
        )
        def _update_content(selected_episode):
            if not selected_episode:
                return html.Div("No episode selected", style={"padding": "12px"})
            return episode_content.get(
                selected_episode,
                html.Div("Episode not found", style={"padding": "12px"}),
            )

        return layout
