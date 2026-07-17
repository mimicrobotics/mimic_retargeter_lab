from typing import Any, Dict

import dash_bootstrap_components as dbc
from dash import html


class Dashboard:
    def __init__(self, metric_stats: Dict[str, Any]):
        """
        High-level dashboard composition with top-level tabs for metrics.

        Accepts a mapping of metric_key -> metric_stats (where metric_stats is the
        per-metric output returned by metric.compute()).

        The build() method will produce a top-level `dcc.Tabs` with one tab per
        metric, delegating rendering to specialized pages like ControlSensitivityPage.
        """
        self.metric_stats = metric_stats
        self.app = None

    def build(self, app, hand_type, retargeter_type):
        """
        Build and return a Dash layout composed of dcc.Tabs, one tab per metric.

        Args:
            app: Dash app instance for registering callbacks.

        """
        from mimic_retargeter_lab.dashboard.pages import (
            CollisionPage,
            FlatnessPage,
            KeyvectorMatchingPage,
            LatencyPage,
            MotionPreservationPage,
            ReferencePosePage,
            ResponseMetricPage,
            WorkspacePage,
        )

        self.app = app
        stats = self.metric_stats

        # Map base metric names to their page classes and tab emojis.
        page_class_map = {
            "Response": ResponseMetricPage,
            "Motion Preservation": MotionPreservationPage,
            "Keyvector Matching": KeyvectorMatchingPage,
            "Pinch Grasps": ReferencePosePage,
            "Flatness": FlatnessPage,
            "Workspace": WorkspacePage,
            "Collision": CollisionPage,
            "Latency": LatencyPage,
        }
        tab_emoji = {
            "Motion Preservation": "\u27a1\ufe0f",
            "Keyvector Matching": "\u270b",
            "Pinch Grasps": "\ud83d\udc4c",
            "Flatness": "\ud83e\uded3",
            "Workspace": "\ud83d\udee0",
            "Collision": "\ud83d\udca5",
            "Response": "\ud83d\udcc8",
            "Latency": "\u23f1\ufe0f",
        }

        pages = {}
        for metric_key, metric_val in stats.items():
            page_cls = None
            for name, cls in page_class_map.items():
                if name in metric_key:
                    page_cls = cls
                    break
            if page_cls is None:
                raise ValueError(f"Unknown metric: {metric_key}")
            pages[metric_key] = page_cls(self.app, metric_key).build_layout(metric_val)

        metric_tabs = []
        for metric_key, metric_page in pages.items():
            emoji = ""
            for name, e in tab_emoji.items():
                if name in metric_key:
                    emoji = e + " "
                    break
            metric_tabs.append(
                dbc.Tab(
                    label=f"{emoji}{metric_key}",
                    tab_id=f"{metric_key}-tab",
                    children=[metric_page],
                )
            )

        # Compose top-level tabs and return
        layout = html.Div(
            [
                html.H2(
                    f"RetargetBench - {hand_type.replace('_', ' ')} - {retargeter_type.replace('_', ' ')} retargeter"
                ),
                dbc.Tabs(metric_tabs, id="top-metric-tabs", className="top-tabs"),
            ]
        )
        return layout
