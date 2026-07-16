from typing import Any, Dict, List, Tuple

import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output


class ResponseMetricPage:
    """
    Renderer for the Control Sensitivity metric.

    - Expects a Dash `app` instance passed to __init__ so it can register callbacks.
    - Left column: fixed-width joint navigator (vertical tabs).
    - Right column: flexible plot area that stays to the right of the navigator.
      The plot area is a vertical flex column so the two graphs (response &
      derivative) share the available height and resize together.
    """

    LEFT_COL_PX = 220
    # Height for the joint+plot area (subtract header/tooling if you have one
    # above the dashboard). Using calc keeps it responsive.
    CONTENT_HEIGHT = "calc(100vh - 120px)"

    def __init__(self, app, page_id: str):
        self.app = app
        self.page_id = page_id

    def _to_list_safe(self, v) -> List[Any]:
        if v is None:
            return []
        try:
            return list(v)
        except Exception:
            return [v]

    def _build_joint_figures(
        self, metric_stats: Dict[str, Any], joint_name: str
    ) -> Tuple[go.Figure, go.Figure]:
        """
        Build response and derivative figures for the given joint across episodes.
        """
        fig_response = go.Figure()
        fig_derivative = go.Figure()

        for episode_id, episode_data in metric_stats.items():
            js = episode_data.get("joint_space", {}) or {}
            js = js.get(joint_name)
            if not js:
                continue

            resp = js.get("response", {}) or {}
            deriv = js.get("derivative", {}) or {}

            in_arr = self._to_list_safe(resp.get("in", []))
            out_arr = self._to_list_safe(resp.get("out", []))
            deriv_vals = self._to_list_safe(deriv.get("values", []))

            if in_arr:
                fig_response.add_trace(
                    go.Scatter(
                        x=list(range(len(in_arr))),
                        y=in_arr,
                        mode="lines",
                        name=f"{episode_id} — human (in)",
                    )
                )
            if out_arr:
                fig_response.add_trace(
                    go.Scatter(
                        x=list(range(len(out_arr))),
                        y=out_arr,
                        mode="lines",
                        name=f"{episode_id} — robot (out)",
                    )
                )
            if deriv_vals:
                fig_derivative.add_trace(
                    go.Scatter(
                        x=list(range(len(deriv_vals))),
                        y=deriv_vals,
                        mode="lines",
                        name=f"{episode_id}",
                    )
                )

        # Tidy up layout, but don't hard-set pixel heights here - the container
        # will control rendering height via CSS and dcc.Graph style.
        common_legend = dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        )
        fig_response.update_layout(
            title=f"{joint_name} — response (in vs out)",
            xaxis_title="timesteps",
            yaxis_title="value",
            legend=common_legend,
            margin=dict(l=40, r=24, t=36, b=40),
        )
        fig_derivative.update_layout(
            title=f"{joint_name} — derivative (d[out]/d[in])",
            xaxis_title="timesteps",
            yaxis_title="derivative",
            legend=common_legend,
            margin=dict(l=40, r=24, t=36, b=40),
        )

        return fig_response, fig_derivative

    def _build_task_figures(
        self, metric_stats: Dict[str, Any], frame_name: str
    ) -> Dict[str, go.Figure]:
        """
        Build response and derivative figures for the given frame across episodes.
        """

        fig_response_pos = go.Figure()
        fig_response_rot = go.Figure()
        fig_derivative_pos = go.Figure()
        fig_derivative_rot = go.Figure()
        for episode_id, episode_data in metric_stats.items():
            for episode_id, episode_data in metric_stats.items():
                ts = episode_data.get("task_space")
                ts = ts.get(frame_name)
                if not ts:
                    continue

                responses = ts.get("frame_responses")
                derivatives = ts.get("derivative_estimates")

                # Add position responses
                for axis in ["x", "y", "z"]:
                    in_pos = self._to_list_safe(
                        responses.get("pos", {}).get("in").get(axis)
                    )
                    out_pos = self._to_list_safe(
                        responses.get("pos", {}).get("out").get(axis)
                    )
                    if in_pos:
                        fig_response_pos.add_trace(
                            go.Scatter(
                                x=list(range(len(in_pos))),
                                y=in_pos,
                                mode="lines",
                                name=f"{episode_id} — pos {axis} (in)",
                            )
                        )
                    if out_pos:
                        fig_response_pos.add_trace(
                            go.Scatter(
                                x=list(range(len(out_pos))),
                                y=out_pos,
                                mode="lines",
                                name=f"{episode_id} — pos {axis} (out)",
                            )
                        )

                # Add rotation responses
                for axis in ["x", "y", "z"]:
                    in_rot = self._to_list_safe(
                        responses.get("rot").get("in").get(axis)
                    )
                    out_rot = self._to_list_safe(
                        responses.get("rot").get("out").get(axis)
                    )
                    if in_rot:
                        fig_response_rot.add_trace(
                            go.Scatter(
                                x=list(range(len(in_rot))),
                                y=in_rot,
                                mode="lines",
                                name=f"{episode_id} — rot {axis} (in)",
                            )
                        )
                    if out_rot:
                        fig_response_rot.add_trace(
                            go.Scatter(
                                x=list(range(len(out_rot))),
                                y=out_rot,
                                mode="lines",
                                name=f"{episode_id} — rot {axis} (out)",
                            )
                        )

                # Add derivative estimates
                for fig, category in zip(
                    [fig_derivative_pos, fig_derivative_rot], ["pos", "rot"]
                ):
                    for axis in ["x", "y", "z"]:
                        deriv = self._to_list_safe(derivatives.get(category).get(axis))
                        if deriv:
                            fig.add_trace(
                                go.Scatter(
                                    x=list(range(len(deriv))),
                                    y=deriv,
                                    mode="lines",
                                    name=f"{episode_id} — {category} {axis}",
                                )
                            )

            # Tidy up layout
            common_legend = dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            )
            fig_response_pos.update_layout(
                title=f"{frame_name} — response (in vs out)",
                xaxis_title="timesteps",
                yaxis_title="value",
                legend=common_legend,
                margin=dict(l=40, r=24, t=36, b=40),
            )
            fig_response_rot.update_layout(
                title=f"{frame_name} — response (in vs out)",
                xaxis_title="timesteps",
                yaxis_title="value",
                legend=common_legend,
                margin=dict(l=40, r=24, t=36, b=40),
            )
            fig_derivative_pos.update_layout(
                title=f"{frame_name} — derivative (d[out]/d[in])",
                xaxis_title="timesteps",
                yaxis_title="derivative",
                legend=common_legend,
                margin=dict(l=40, r=24, t=36, b=40),
            )
            fig_derivative_rot.update_layout(
                title=f"{frame_name} — derivative (d[out]/d[in])",
                xaxis_title="timesteps",
                yaxis_title="derivative",
                legend=common_legend,
                margin=dict(l=40, r=24, t=36, b=40),
            )

        return (
            fig_response_pos,
            fig_response_rot,
            fig_derivative_pos,
            fig_derivative_rot,
        )

    def build_layout(self, metric_stats: Dict[str, Any]) -> html.Div:
        """
        Build the Control Sensitivity page layout.

        Layout specifics:
        - Metric-level tabs (Joint space / Task space)
        - Joint space: flex row with fixed-width left nav and flexible right plot area.
          The right plot area is a vertical flex column so the two graphs split the
          available height nicely and remain side-by-side with the nav.
        """
        if not metric_stats:
            return html.Div([html.H3("No metric data available")])

        # Discover joints/tasks
        first_episode = next(iter(metric_stats.values()))
        joint_space = (
            (first_episode.get("joint_space") or {})
            if isinstance(first_episode, dict)
            else {}
        )
        task_space = (
            (first_episode.get("task_space") or {})
            if isinstance(first_episode, dict)
            else {}
        )

        joint_names = sorted(joint_space.keys())
        frame_names = sorted(task_space.keys())

        # Joint navigator: vertical tabs with fixed visual width
        if joint_names:
            # Replace the vertical dcc.Tabs navigator with a RadioItems-based
            # left navigation. RadioItems behave predictably in fixed-width flex
            # containers and avoid the layout wrapping problems caused by dcc.Tabs.
            joint_radio_options = [
                {"label": jn, "value": f"cs-joints-{jn}"} for jn in joint_names
            ]
            joint_tabs = dcc.RadioItems(
                options=joint_radio_options,
                value=f"cs-{joint_names[0]}",
                id=f"cs-joints-radio-{self.page_id}",
                inputStyle={"marginRight": "8px"},
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "overflowY": "auto",
                    "maxHeight": self.CONTENT_HEIGHT,
                    "paddingRight": "6px",
                },
                className="cs-inner-radio",
            )
        else:
            joint_tabs = html.Div("No joints found", style={"padding": "8px"})

        if frame_names:
            frame_radio_options = [
                {"label": fn, "value": f"cs-frames-{fn}"} for fn in frame_names
            ]
            task_tabs = dcc.RadioItems(
                options=frame_radio_options,
                value=f"cs-{frame_names[0]}-{self.page_id}",
                id=f"cs-task-radio-{self.page_id}",
                inputStyle={"marginRight": "8px"},
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "overflowY": "auto",
                    "maxHeight": self.CONTENT_HEIGHT,
                    "paddingRight": "6px",
                },
                className="cs-inner-radio",
            )
        else:
            task_tabs = html.Div("No frames found", style={"padding": "8px"})

        # Precompute figures for joints (so callback lookup is cheap)
        joint_figs: Dict[str, Tuple[go.Figure, go.Figure]] = {}
        task_figs: Dict[str, Tuple[go.Figure, go.Figure, go.Figure, go.Figure]] = {}
        for jn in joint_names:
            joint_figs[jn] = self._build_joint_figures(metric_stats, jn)

        for fn in frame_names:
            task_figs[fn] = self._build_task_figures(metric_stats, fn)

        # Left column (fixed), right column (flex)
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

        # Right column: vertical flex container with two graph panes
        right_col_style = {
            "flex": "1 1 auto",
            "minWidth": "0",
            "display": "flex",
            "flexDirection": "column",
            "height": "100%",
            "boxSizing": "border-box",
            "paddingLeft": "12px",
        }

        right_col_style_tasks = {**right_col_style, "overflowY": "auto"}

        # Plot panes: each fills half the height (can be adjusted)
        plot_pane_style = {
            "flex": "1 1 50%",
            "minHeight": "0",
            "overflow": "hidden",
            "paddingBottom": "8px",
        }

        plot_pane_style_tasks = {
            "height": "600px",
            "minHeight": "600px",  # Ensure it doesn't shrink
            "paddingBottom": "8px",
            "boxSizing": "border-box",
        }

        # Build right-hand content with placeholders; callback will populate the graphs
        right_content_joints = html.Div(
            [
                html.Div(
                    id=f"cs-plot-joints-top-{self.page_id}", style=plot_pane_style
                ),
                html.Div(
                    id=f"cs-plot-joints-bottom-{self.page_id}", style=plot_pane_style
                ),
            ],
            style=right_col_style,
        )
        right_content_tasks = html.Div(
            [
                html.Div(
                    id=f"cs-plot-tasks-resp-pos-{self.page_id}",
                    style=plot_pane_style_tasks,
                ),
                html.Div(
                    id=f"cs-plot-tasks-resp-rot-{self.page_id}",
                    style=plot_pane_style_tasks,
                ),
                html.Div(
                    id=f"cs-plot-tasks-deriv-pos-{self.page_id}",
                    style=plot_pane_style_tasks,
                ),
                html.Div(
                    id=f"cs-plot-tasks-deriv-rot-{self.page_id}",
                    style=plot_pane_style_tasks,
                ),
            ],
            style=right_col_style_tasks,
        )

        joint_space_content = html.Div(
            [
                html.Div(
                    [html.H6("Joints", style={"marginTop": "0"}), joint_tabs],
                    style=left_col_style,
                ),
                right_content_joints,
            ],
            # Ensure the container is a horizontal flex row that never wraps so the
            # right-hand plot area stays to the right of the navigator at all sizes.
            style={
                "display": "flex",
                "flexDirection": "row",
                "flexWrap": "nowrap",
                "alignItems": "stretch",
                "width": "100%",
                "height": self.CONTENT_HEIGHT,
            },
        )

        task_space_content = html.Div(
            [
                html.Div(
                    [html.H6("frames", style={"marginTop": "0"}), task_tabs],
                    style=left_col_style,
                ),
                right_content_tasks,
            ],
            # Ensure the container is a horizontal flex row that never wraps so the
            # right-hand plot area stays to the right of the navigator at all sizes.
            style={
                "display": "flex",
                "flexDirection": "row",
                "flexWrap": "nowrap",
                "alignItems": "stretch",
                "width": "100%",
                "height": self.CONTENT_HEIGHT,
            },
        )

        metric_tabs = dbc.Tabs(
            [
                dbc.Tab(
                    joint_space_content,
                    label="Joint space",
                    tab_id=f"joint-space-{self.page_id}",
                ),
                dbc.Tab(
                    task_space_content,
                    label="Cartesian space",
                    tab_id=f"task-space-{self.page_id}",
                ),
            ],
            id=f"cs-metric-tabs-{self.page_id}",
            className=f"cs-metric-tabs-{self.page_id}",
            style={
                "boxSizing": "border-box",
            },
        )

        layout = html.Div(
            [metric_tabs],
            style={
                "width": "100%",
                "boxSizing": "border-box",
                "padding": "8px",
                "height": "600px",
                # "height": self.CONTENT_HEIGHT,
                "minHeight": "300px",
            },
        )

        # Callback to render figures into the top/bottom plot panes so they occupy the
        # available right-column vertical space and remain side-by-side with the nav.
        # Callback watches the RadioItems value (cs-joint-radio) instead of the old Tabs id.
        @self.app.callback(
            Output(f"cs-plot-joints-top-{self.page_id}", "children"),
            Output(f"cs-plot-joints-bottom-{self.page_id}", "children"),
            [Input(f"cs-joints-radio-{self.page_id}", "value")],
        )
        def _update_plots(selected_value):
            # selected_value like "cs-<joint_name>"
            if not selected_value:
                return (
                    html.Div("No joint selected", style={"padding": "12px"}),
                    html.Div("", style={"padding": "12px"}),
                )

            try:
                joint_name = selected_value.split("-", 2)[2]
            except Exception:
                joint_name = selected_value

            fig_resp, fig_deriv = joint_figs.get(joint_name, (go.Figure(), go.Figure()))

            # Put each graph inside a div that fills its pane; set graph height to 100%.
            top = dcc.Graph(
                figure=fig_resp,
                style={"height": "100%", "width": "100%"},
                config={"responsive": True},
            )
            bottom = dcc.Graph(
                figure=fig_deriv,
                style={"height": "100%", "width": "100%"},
                config={"responsive": True},
            )

            return top, bottom

        # Callback to render figures into the top/bottom plot panes for task space
        # Callback watches the RadioItems value (cs-task-radio) for task space plots.
        @self.app.callback(
            Output(f"cs-plot-tasks-resp-pos-{self.page_id}", "children"),
            Output(f"cs-plot-tasks-resp-rot-{self.page_id}", "children"),
            Output(f"cs-plot-tasks-deriv-pos-{self.page_id}", "children"),
            Output(f"cs-plot-tasks-deriv-rot-{self.page_id}", "children"),
            [Input(f"cs-task-radio-{self.page_id}", "value")],
        )
        def _update_task_plots(selected_value):
            # selected_value like "cs-frames-<frame_name>"
            if not selected_value:
                return (
                    html.Div("No task selected", style={"padding": "12px"}),
                    html.Div("", style={"padding": "12px"}),
                )

            try:
                frame_name = selected_value.split("-", 2)[2]
            except Exception:
                frame_name = selected_value

            fig_resp_pos, fig_resp_rot, fig_deriv_pos, fig_deriv_rot = task_figs.get(
                frame_name, (go.Figure(), go.Figure(), go.Figure(), go.Figure())
            )

            # Put each graph inside a div that fills its pane; set graph height to 100%.
            resp_pos = dcc.Graph(
                figure=fig_resp_pos,
                style={"height": "100%", "width": "100%"},
                config={"responsive": True},
            )
            resp_rot = dcc.Graph(
                figure=fig_resp_rot,
                style={"height": "100%", "width": "100%"},
                config={"responsive": True},
            )
            deriv_pos = dcc.Graph(
                figure=fig_deriv_pos,
                style={"height": "100%", "width": "100%"},
                config={"responsive": True},
            )

            deriv_rot = dcc.Graph(
                figure=fig_deriv_rot,
                style={"height": "100%", "width": "100%"},
                config={"responsive": True},
            )

            return resp_pos, resp_rot, deriv_pos, deriv_rot

        return layout
