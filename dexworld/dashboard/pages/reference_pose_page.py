from typing import Any, Dict, List, Set

import dash_bootstrap_components as dbc
import dash_vtk
import numpy as np
import plotly.graph_objs as go
from dash import callback_context, dcc, html
from dash.dependencies import Input, Output


class ReferencePosePage:
    """
    Renderer for the Reference Pose metric.

    This page visualizes frame positions (human vs robot) in 3D and draws line
    segments between corresponding frame positions to show per-timestep error.

    - Expects a Dash `app` instance passed to __init__ so it can register callbacks.
    - Left column: episode selector (vertical RadioItems).
    - Right column: 3D plot area that displays the precomputed figure for the
      selected episode.
    """

    LEFT_COL_PX = 220
    CONTENT_HEIGHT = "calc(100vh - 120px)"

    def __init__(self, app, page_id: str):
        self.app = app
        self.page_id = page_id

    @staticmethod
    def _rgba_to_hex(rgba: Any) -> str:
        r, g, b = (int(round(float(c) * 255)) for c in rgba[:3])
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def _build_vtk_skeleton_view(
        links: Any, view_id: str, diff_lines: Any = None
    ) -> Any:
        """Render a capsule skeleton from kinematic-tree links via dash-vtk."""
        if not links:
            return html.Div(
                "No skeleton data", style={"padding": "8px", "color": "#888"}
            )

        bone_color = (0.90, 0.85, 0.78)  # light cream
        joint_color = (0.85, 0.20, 0.20)  # red / muscle
        bone_radius = 0.003
        joint_radius = 0.005
        n_sides = 8

        actors: List[Any] = []
        joint_set: set = set()
        joint_list: List[np.ndarray] = []

        for li, link in enumerate(links):
            start, end = link
            s = np.asarray(start, dtype=np.float32).ravel()[:3]
            e = np.asarray(end, dtype=np.float32).ravel()[:3]
            if s.ndim == 2:
                s = s[0]
            if e.ndim == 2:
                e = e[0]

            for p in (s, e):
                key = tuple(np.round(p, 5))
                if key not in joint_set:
                    joint_set.add(key)
                    joint_list.append(p)

            d = e - s
            length = float(np.linalg.norm(d))
            if length < 1e-8:
                continue
            axis = d / length

            ref = np.array([1, 0, 0], dtype=np.float32)
            if abs(np.dot(axis, ref)) > 0.9:
                ref = np.array([0, 1, 0], dtype=np.float32)
            p1 = np.cross(axis, ref)
            p1 /= np.linalg.norm(p1) + 1e-8
            p2 = np.cross(axis, p1)

            ang = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
            ring = np.stack([np.cos(ang), np.sin(ang)], axis=1)
            off = bone_radius * (ring[:, 0:1] * p1 + ring[:, 1:2] * p2)

            pts = np.vstack([s + off, e + off])
            faces = []
            for i in range(n_sides):
                j = (i + 1) % n_sides
                faces.extend([3, i, j, j + n_sides])
                faces.extend([3, i, j + n_sides, i + n_sides])

            actors.append(
                dash_vtk.GeometryRepresentation(
                    id=f"{view_id}-bone-{li}",
                    property={"color": bone_color, "ambient": 0.3, "diffuse": 0.7},
                    children=[
                        dash_vtk.PolyData(points=pts.reshape(-1).tolist(), polys=faces)
                    ],
                )
            )

        # Joint spheres
        n_lat, n_lon = 4, 6
        for ji, c in enumerate(joint_list):
            verts: list = []
            tris: list = []
            for la in range(n_lat + 1):
                th = np.pi * la / n_lat
                for lo in range(n_lon):
                    ph = 2 * np.pi * lo / n_lon
                    verts.extend(
                        [
                            float(joint_radius * np.sin(th) * np.cos(ph) + c[0]),
                            float(joint_radius * np.sin(th) * np.sin(ph) + c[1]),
                            float(joint_radius * np.cos(th) + c[2]),
                        ]
                    )
            for la in range(n_lat):
                for lo in range(n_lon):
                    lo2 = (lo + 1) % n_lon
                    v0, v1 = la * n_lon + lo, la * n_lon + lo2
                    v2, v3 = v0 + n_lon, v1 + n_lon
                    tris.extend([3, v0, v2, v1, 3, v1, v2, v3])
            actors.append(
                dash_vtk.GeometryRepresentation(
                    id=f"{view_id}-jnt-{ji}",
                    property={"color": joint_color, "ambient": 0.3, "diffuse": 0.7},
                    children=[dash_vtk.PolyData(points=verts, polys=tris)],
                )
            )

        # Vector-diff line overlay
        if diff_lines:
            for dli, seg in enumerate(diff_lines):
                src = np.asarray(seg["src"], dtype=np.float32)
                tgt = np.asarray(seg["tgt"], dtype=np.float32)
                color = ReferencePosePage._LINE_COLORS[
                    dli % len(ReferencePosePage._LINE_COLORS)
                ]
                actors.append(
                    dash_vtk.GeometryRepresentation(
                        id=f"{view_id}-ln-{dli}",
                        property={
                            "color": color,
                            "lineWidth": 4,
                            "pointSize": 10,
                            "ambient": 1.0,
                            "diffuse": 0.0,
                        },
                        children=[
                            dash_vtk.PolyData(
                                points=np.vstack([src, tgt]).reshape(-1).tolist(),
                                lines=[2, 0, 1],
                                verts=[1, 0, 1, 1],
                            )
                        ],
                    )
                )

        return dash_vtk.View(
            id=view_id,
            background=[1.0, 1.0, 1.0],
            children=actors,
            style={"width": "100%", "height": "100%"},
        )

    # Colours for vector-diff line overlays (one per diff, cycled).
    _LINE_COLORS = [
        (0.90, 0.20, 0.20),  # red
        (0.20, 0.60, 0.90),  # blue
        (0.20, 0.80, 0.30),  # green
        (0.90, 0.55, 0.10),  # orange
        (0.60, 0.20, 0.80),  # purple
    ]

    @classmethod
    def _build_vtk_robot_view(
        cls,
        robot_meshes: Any,
        view_id: str,
        diff_lines: Any = None,
        show_edges: bool = False,
    ) -> Any:
        """Render mesh geoms via dash-vtk, with optional vector-diff line overlay.

        *diff_lines* is a list of ``{"src": (3,), "tgt": (3,)}`` dicts.
        Each pair is drawn as a thick coloured line + endpoint spheres.
        *show_edges* enables edge wireframe on the mesh for better finger
        readability.
        """
        if not robot_meshes:
            return html.Div(
                "No robot meshes available",
                style={"padding": "8px", "color": "#888"},
            )

        actors: List[Any] = []
        for idx, geom in enumerate(robot_meshes):
            V = np.asarray(geom["vertices"], dtype=np.float32)
            F = np.asarray(geom["faces"], dtype=np.int32)
            if V.size == 0 or F.size == 0:
                continue

            # VTK expects a flat polygons array: for each triangle, [3, i, j, k].
            polys = np.empty((F.shape[0], 4), dtype=np.int32)
            polys[:, 0] = 3
            polys[:, 1:] = F
            polys_flat = polys.reshape(-1).tolist()

            rgba = np.asarray(
                geom.get("rgba", (0.75, 0.75, 0.75, 1.0)), dtype=np.float32
            )
            color = (float(rgba[0]), float(rgba[1]), float(rgba[2]))

            prop = {
                "color": color,
                "opacity": float(rgba[3]) if rgba.size >= 4 else 1.0,
                "edgeVisibility": show_edges,
                "ambient": 0.3,
                "diffuse": 0.7,
                "specular": 0.1,
                "specularPower": 10,
            }
            if show_edges:
                prop["edgeColor"] = (0.55, 0.45, 0.40)
                prop["opacity"] = 1.0

            actors.append(
                dash_vtk.GeometryRepresentation(
                    id=f"{view_id}-actor-{idx}",
                    property=prop,
                    children=[
                        dash_vtk.PolyData(
                            points=V.reshape(-1).tolist(),
                            polys=polys_flat,
                        )
                    ],
                )
            )

        # Optional vector-diff line overlay
        if diff_lines:
            for li, seg in enumerate(diff_lines):
                src = np.asarray(seg["src"], dtype=np.float32)
                tgt = np.asarray(seg["tgt"], dtype=np.float32)
                color = cls._LINE_COLORS[li % len(cls._LINE_COLORS)]
                pts = np.vstack([src, tgt]).reshape(-1).tolist()
                actors.append(
                    dash_vtk.GeometryRepresentation(
                        id=f"{view_id}-line-{li}",
                        property={
                            "color": color,
                            "lineWidth": 4,
                            "pointSize": 10,
                            "ambient": 1.0,
                            "diffuse": 0.0,
                        },
                        children=[
                            dash_vtk.PolyData(
                                points=pts,
                                lines=[2, 0, 1],
                                verts=[1, 0, 1, 1],
                            )
                        ],
                    )
                )

        return dash_vtk.View(
            id=view_id,
            background=[1.0, 1.0, 1.0],
            children=actors,
            style={"width": "100%", "height": "100%"},
        )

    def _build_pose_figure(
        self,
        human_links: Any,
        robot_links: Any,
        episode_id: str,
        robot_meshes: Any = None,
    ) -> go.Figure:
        """
        Build a simple skeleton figure showing links for human and robot hands.

        The inputs `human_links` and `robot_links` are expected to be iterables of
        link pairs (start, end). Each start/end can be array-like (shape (3,) or (N,3))
        or a torch tensor; we convert defensively using numpy. We ignore transforms
        and only draw the raw links. Human links are drawn in one color, robot links
        in another; within each hand all colors are the same.
        """
        fig = go.Figure()

        human_color = "#7f7f7f"  # neutral/darker for human
        robot_color = "#1f77b4"  # blue for robot

        # Track extents so we can ensure equal axis scales
        mins = np.array([float("inf"), float("inf"), float("inf")])
        maxs = np.array([-float("inf"), -float("inf"), -float("inf")])

        def _as_xyz(p):
            """Convert a (T,3) array to a (3,) array."""
            a = np.asarray(p)
            if a.ndim == 2:  # e.g. (T,3) -> first row
                a = a[0]
            if a.ndim != 1 or a.shape[0] != 3:
                return None
            return a

        # Draw human links; only the first link shows a legend entry for the hand
        human_legend_shown = False
        if human_links:
            for link in human_links:
                start, end = link
                # defensive indexing similar to previous implementation
                s = _as_xyz(start)
                e = _as_xyz(end)
                # Update bounds if possible
                mins = np.minimum(mins, s)
                maxs = np.maximum(maxs, e)
                mins = np.minimum(mins, e)
                maxs = np.maximum(maxs, e)
                showlegend = not human_legend_shown
                fig.add_trace(
                    go.Scatter3d(
                        x=[s[0], e[0]],
                        y=[s[1], e[1]],
                        z=[s[2], e[2]],
                        mode="lines+markers",
                        line=dict(color=human_color, width=4),
                        marker=dict(size=3, color=human_color),
                        name="human",
                        showlegend=showlegend,
                    )
                )
                human_legend_shown = human_legend_shown or showlegend

        # Draw robot links; only the first link shows a legend entry for the hand
        robot_legend_shown = False
        if robot_links:
            for link in robot_links:
                start, end = link
                s = _as_xyz(start)
                e = _as_xyz(end)
                if s is None or e is None:
                    continue
                # Update bounds if possible
                mins = np.minimum(mins, s)
                maxs = np.maximum(maxs, e)
                mins = np.minimum(mins, e)
                maxs = np.maximum(maxs, e)

                showlegend = not robot_legend_shown
                robot_skel_opacity = 0.4 if robot_meshes else 1.0
                fig.add_trace(
                    go.Scatter3d(
                        x=[s[0], e[0]],
                        y=[s[1], e[1]],
                        z=[s[2], e[2]],
                        mode="lines+markers",
                        line=dict(color=robot_color, width=4),
                        marker=dict(size=3, color=robot_color),
                        opacity=robot_skel_opacity,
                        name="robot",
                        showlegend=showlegend,
                    )
                )
                robot_legend_shown = robot_legend_shown or showlegend

        # Robot meshes are rendered separately via dash-vtk (see _build_vtk_robot_view);
        # we still update the Plotly scene bounds from them so the skeleton view
        # frames the same volume as the mesh view.
        if robot_meshes:
            for geom in robot_meshes:
                V = np.asarray(geom["vertices"])
                if V.size == 0:
                    continue
                mins = np.minimum(mins, V.min(axis=0))
                maxs = np.maximum(maxs, V.max(axis=0))

        # Compute symmetric ranges if we collected any points
        min_x, min_y, min_z = mins[0], mins[1], mins[2]
        max_x, max_y, max_z = maxs[0], maxs[1], maxs[2]
        if min_x != float("inf") and max_x != float("-inf"):
            span_x = max_x - min_x
            span_y = max_y - min_y
            span_z = max_z - min_z
            max_span = max(span_x, span_y, span_z, 1e-6)
            cx = 0.5 * (max_x + min_x)
            cy = 0.5 * (max_y + min_y)
            cz = 0.5 * (max_z + min_z)
            half = 0.5 * max_span
            x_range = [cx - half, cx + half]
            y_range = [cy - half, cy + half]
            z_range = [cz - half, cz + half]
        else:
            x_range = [-1, 1]
            y_range = [-1, 1]
            z_range = [-1, 1]

        fig.update_layout(
            title=f"Episode {episode_id} — skeletons (links only)",
            margin=dict(l=0, r=0, t=36, b=0),
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="z",
                xaxis=dict(range=x_range),
                yaxis=dict(range=y_range),
                zaxis=dict(range=z_range),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=1),
                camera=dict(projection=dict(type="orthographic")),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0,
                xanchor="right",
                x=1,
                font=dict(color="white"),
            ),
        )

        return fig

    def _collect_columns(self, obj: Any, prefix: str = "") -> Set[str]:
        """
        Recursively collect flattened column keys from a nested dict-like object.

        Example: {'a': {'x': 1}, 'b': 2} -> {'a.x', 'b'}
        """
        cols: Set[str] = set()
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    cols.update(self._collect_columns(v, key))
                else:
                    cols.add(key)
        else:
            # non-dict objects: treat the whole prefix as a column if present
            if prefix:
                cols.add(prefix)
        return cols

    def _get_nested(self, obj: Dict[str, Any], dotted_key: str):
        """
        Retrieve a nested value from obj following dotted path; return None if missing.
        """
        parts = dotted_key.split(".")
        cur = obj
        for p in parts:
            if not isinstance(cur, dict):
                return None
            if p not in cur:
                return None
            cur = cur[p]
        return cur

    def build_layout(self, metric_stats: Dict[str, Any]) -> html.Div:
        """
        Build the Reference Pose page layout and register callbacks.

        Notes:
        - Precomputes 3D figures for each episode so callbacks are lightweight.
        - RadioItems values are of the form 'cs-episode-<episode_id>-<page_id>'.
        """
        if not metric_stats:
            return html.Div([html.H3("No metric data available")])

        # Discover episodes from the metric_stats
        episode_ids = sorted(metric_stats.keys())

        # Left column: episode selector. Split into "Pinches" (thumb_to_*_tip)
        # and "Other" so the tip-to-tip closures are visually grouped.
        def _is_pinch(eid: str) -> bool:
            return eid.startswith("thumb_to_") and eid.endswith("_tip")

        pinch_eids = [eid for eid in episode_ids if _is_pinch(eid)]
        other_eids = [eid for eid in episode_ids if not _is_pinch(eid)]

        def _to_options(eids):
            return [
                {"label": eid, "value": f"cs-episode-{eid}-{self.page_id}"}
                for eid in eids
            ]

        pinch_options = _to_options(pinch_eids)
        other_options = _to_options(other_eids)

        # Default selection: prefer the first "Other" episode (preserves prior
        # alphabetical-first behavior, e.g. "rest"); fall back to first pinch.
        if other_options:
            default_pinch_value = None
            default_other_value = other_options[0]["value"]
        elif pinch_options:
            default_pinch_value = pinch_options[0]["value"]
            default_other_value = None
        else:
            default_pinch_value = default_other_value = None

        radio_style = {
            "display": "flex",
            "flexDirection": "column",
            "paddingRight": "6px",
        }
        group_header_style = {
            "marginTop": "8px",
            "marginBottom": "4px",
            "fontSize": "13px",
            "fontWeight": "600",
            "color": "#555",
            "textTransform": "uppercase",
            "letterSpacing": "0.04em",
        }

        if episode_ids:
            radio_groups = []
            if pinch_options:
                radio_groups.extend(
                    [
                        html.Div("Pinches", style=group_header_style),
                        dcc.RadioItems(
                            options=pinch_options,
                            value=default_pinch_value,
                            id=f"cs-task-radio-pinch-{self.page_id}",
                            inputStyle={"marginRight": "8px"},
                            style=radio_style,
                            className="cs-inner-radio",
                        ),
                    ]
                )
            else:
                radio_groups.append(
                    dcc.RadioItems(
                        options=[],
                        value=None,
                        id=f"cs-task-radio-pinch-{self.page_id}",
                        style={"display": "none"},
                    )
                )

            if other_options:
                radio_groups.extend(
                    [
                        html.Div("Other", style=group_header_style),
                        dcc.RadioItems(
                            options=other_options,
                            value=default_other_value,
                            id=f"cs-task-radio-other-{self.page_id}",
                            inputStyle={"marginRight": "8px"},
                            style=radio_style,
                            className="cs-inner-radio",
                        ),
                    ]
                )
            else:
                radio_groups.append(
                    dcc.RadioItems(
                        options=[],
                        value=None,
                        id=f"cs-task-radio-other-{self.page_id}",
                        style={"display": "none"},
                    )
                )

            task_tabs = html.Div(
                radio_groups,
                style={
                    "overflowY": "auto",
                    "maxHeight": self.CONTENT_HEIGHT,
                },
            )
        else:
            task_tabs = html.Div("No episodes found", style={"padding": "8px"})

        # Cache the raw per-episode inputs; defer figure construction to the callback
        # so that editing `_build_pose_figure` takes effect on the next radio click
        # without needing to restart the Python process / recompute metrics.
        episode_figure_inputs: Dict[str, Dict[str, Any]] = {}
        for eid in episode_ids:
            episode_data = metric_stats.get(eid, {}) or {}
            episode_figure_inputs[eid] = {
                "human_links": episode_data.get("human_links"),
                "robot_meshes": episode_data.get("robot_meshes"),
                "human_meshes": episode_data.get("human_meshes"),
                "vector_diff_lines": episode_data.get("vector_diff_lines"),
            }

        # Precompute scalar metric tables per episode (below the skeleton plot)
        scalar_tables: Dict[str, Any] = {}
        for eid in episode_ids:
            episode_data = metric_stats.get(eid, {}) or {}
            ref_metrics = episode_data.get("reference_pose_metrics", {}) or {}
            error_metrics = ref_metrics.get("error_metrics", {}) or {}

            if not error_metrics:
                scalar_tables[eid] = html.Div(
                    "No scalar metrics available", style={"padding": "8px"}
                )
                continue

            # Discover all columns by scanning every named metric and flattening keys
            columns: Set[str] = set()
            for m in error_metrics.values():
                columns.update(self._collect_columns(m))
            # Ensure deterministic order
            sorted_columns = sorted(columns)

            # Build header row dynamically
            header_cells = [html.Th("Keyvector")]
            for col in sorted_columns:
                header_cells.append(html.Th(col))
            header = html.Tr(header_cells)

            # Build rows for each named metric
            rows = [header]
            for name in sorted(error_metrics.keys()):
                m = error_metrics.get(name, {}) or {}
                cells = [html.Td(name)]
                for col in sorted_columns:
                    v = self._get_nested(m, col)

                    # Format numbers, otherwise represent as string or N/A
                    def fmt(x):
                        if x is None:
                            return "N/A"
                        try:
                            return f"{float(x):.4f}"
                        except Exception:
                            return str(x)

                    cells.append(html.Td(fmt(v)))
                rows.append(html.Tr(cells))

            table = html.Table(
                rows, style={"width": "100%", "borderCollapse": "collapse"}
            )
            scalar_tables[eid] = table

        # Left column style
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

        # Right column style
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

        plot_pane_style_tasks = {
            "height": "calc(100vh - 260px)",
            "minHeight": "720px",
            "paddingBottom": "8px",
            "boxSizing": "border-box",
        }

        # Display options (toggle vector-diff lines)
        display_options = dcc.Checklist(
            id=f"cs-display-opts-{self.page_id}",
            options=[{"label": " Show vector-diff lines", "value": "lines"}],
            value=[],
            style={
                "fontSize": "20px",
                "paddingBottom": "4px",
                "display": "inline-block",
            },
            inputStyle={"marginRight": "6px", "width": "18px", "height": "18px"},
        )

        # This container will be replaced with the Graph + table for the selected episode
        right_content_tasks = html.Div(
            [
                html.Div(
                    display_options,
                    style={"paddingBottom": "4px"},
                ),
                html.Div(
                    id=f"cs-plot-tasks-skel-{self.page_id}",
                    style=plot_pane_style_tasks,
                ),
            ],
            style=right_col_style_tasks,
        )

        task_space_content = html.Div(
            [
                html.Div(
                    [html.H6("Episodes", style={"marginTop": "0"}), task_tabs],
                    style=left_col_style,
                ),
                right_content_tasks,
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

        metric_tabs = dbc.Tabs(
            [
                dbc.Tab(
                    task_space_content,
                    label="Cartesian space",
                    tab_id=f"task-space-{self.page_id}",
                ),
            ],
            id=f"cs-metric-tabs-{self.page_id}",
            className="cs-metric-tabs",
            style={"boxSizing": "border-box"},
        )

        layout = html.Div(
            [metric_tabs],
            style={
                "width": "100%",
                "boxSizing": "border-box",
                "padding": "8px",
                "height": "calc(100vh - 160px)",
                "minHeight": "800px",
            },
        )

        # Sync callback: when one group is selected, deselect the other so the
        # two grouped RadioItems behave as a single mutually-exclusive selection.
        @self.app.callback(
            Output(f"cs-task-radio-pinch-{self.page_id}", "value"),
            Output(f"cs-task-radio-other-{self.page_id}", "value"),
            Input(f"cs-task-radio-pinch-{self.page_id}", "value"),
            Input(f"cs-task-radio-other-{self.page_id}", "value"),
            prevent_initial_call=True,
        )
        def _sync_episode_radios(pinch_val, other_val):
            ctx = callback_context
            if not ctx.triggered:
                return pinch_val, other_val
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if trigger_id == f"cs-task-radio-pinch-{self.page_id}":
                return pinch_val, None
            return None, other_val

        # Register callback to render the precomputed 3D figure for the selected episode,
        # and render the skeleton figure into the third pane along with scalar metrics table below.
        @self.app.callback(
            Output(f"cs-plot-tasks-skel-{self.page_id}", "children"),
            [
                Input(f"cs-task-radio-pinch-{self.page_id}", "value"),
                Input(f"cs-task-radio-other-{self.page_id}", "value"),
                Input(f"cs-display-opts-{self.page_id}", "value"),
            ],
        )
        def _update_task_plots(pinch_val, other_val, display_opts):
            selected_value = pinch_val or other_val
            display_opts = display_opts or []
            show_lines = "lines" in display_opts

            # selected_value is of the form 'cs-episode-<episode_id>-<page_id>'
            if not selected_value:
                return (
                    html.Div("No episode selected", style={"padding": "12px"}),
                    html.Div("", style={"padding": "12px"}),
                )

            # Extract episode id robustly: remove prefix and suffix
            prefix = "cs-episode-"
            episode_id = selected_value
            if selected_value.startswith(prefix):
                remainder = selected_value[len(prefix) :]
                # remainder may be "<episode_id>-<page_id>" — isolate episode_id by removing final "-<page_id>"
                parts = remainder.rsplit("-", 1)
                episode_id = parts[0] if parts else remainder

            inputs = episode_figure_inputs.get(episode_id)

            # Extract per-embodiment vector-diff line segments
            vdl = inputs.get("vector_diff_lines") if inputs else None
            robot_lines = None
            human_lines = None
            if show_lines and vdl:
                robot_lines = [seg["robot"] for seg in vdl if "robot" in seg]
                human_lines = [seg["human"] for seg in vdl if "human" in seg]

            robot_meshes = inputs["robot_meshes"] if inputs else None
            vtk_view = self._build_vtk_robot_view(
                robot_meshes,
                view_id=f"cs-vtk-view-{self.page_id}-{episode_id}",
                diff_lines=robot_lines,
            )

            human_meshes = inputs["human_meshes"] if inputs else None
            human_vtk_view = self._build_vtk_robot_view(
                human_meshes,
                view_id=f"cs-vtk-human-{self.page_id}-{episode_id}",
                diff_lines=human_lines,
                show_edges=True,
            )

            human_links = inputs.get("human_links") if inputs else None
            skel_vtk_view = self._build_vtk_skeleton_view(
                human_links,
                view_id=f"cs-vtk-skel-{self.page_id}-{episode_id}",
                diff_lines=human_lines,
            )

            # Side-by-side: robot mesh | MANO mesh | capsule skeleton
            side_pane_style = {
                "flex": "1 1 0",
                "minWidth": "0",
                "height": "calc(100vh - 500px)",
                "minHeight": "640px",
                "border": "4px solid #ddd",
                "boxSizing": "border-box",
            }
            panes = [
                html.Div(vtk_view, style=side_pane_style),
            ]
            if human_meshes:
                panes.append(
                    html.Div(
                        human_vtk_view,
                        style={**side_pane_style, "marginLeft": "8px"},
                    )
                )
            panes.append(
                html.Div(
                    skel_vtk_view,
                    style={**side_pane_style, "marginLeft": "8px"},
                )
            )
            plot_row = html.Div(
                panes,
                style={"display": "flex", "flexDirection": "row", "width": "100%"},
            )

            # Get precomputed scalar table for this episode (or a placeholder)
            table = scalar_tables.get(
                episode_id,
                html.Div("No scalar metrics available", style={"padding": "8px"}),
            )

            container = html.Div(
                [
                    html.Div(plot_row, style={"paddingBottom": "8px"}),
                    html.Div([html.H6("Keyvector Metrics"), table]),
                ],
                style={"display": "flex", "flexDirection": "column", "height": "100%"},
            )

            return container

        return layout
