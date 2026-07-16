"""Serve the metrics dashboard for a *single* (hand, retargeter) pair.

Companion to ``compute_hand_retargeter_pair_metrics.py``. Loads exactly one
``reports/metrics-stats_<dataset>_<hand>_<retargeter>.pkl`` — the pair named by
the Hydra overrides — and starts the Dash app. No metrics are recomputed, and
hot-reload is safe to enable because restarting the Python process is cheap.

To compare *every* (hand, retargeter) pair for a dataset side by side, use
``scripts/serve_all_pairs_dashboard.py`` instead.

Usage:
    # Default config (matches compute_hand_retargeter_pair_metrics.py defaults: shadow_hand + keyvector)
    python scripts/serve_single_pair_dashboard.py hand=shadow_dexee_hand retargeter=dexpilot

    # Manus config: same overrides as the matching compute_hand_retargeter_pair_metrics.py invocation
    python scripts/serve_single_pair_dashboard.py \\
        dataset=manus retargeter=sampling_based

Run ``compute_hand_retargeter_pair_metrics.py`` at least once with the same ``hand`` /
``retargeter`` / ``dataset`` / ``reports_dir`` first so the
pickle exists.
"""

import pickle
from pathlib import Path

import dash_bootstrap_components as dbc
import hydra
from dash import Dash
from hydra.core.hydra_config import HydraConfig

from dexworld.dashboard.dashboard import Dashboard
from dexworld.types import Retargeter, RobotHandType
from dexworld.utils import configure_logging, get_logger


@hydra.main(config_path="../config", config_name="compute_metrics", version_base="1.2")
def main(cfg):
    configure_logging(level=cfg.logging.level)
    logger = get_logger(Path(__file__).stem)

    reports_dir = Path(hydra.utils.to_absolute_path(cfg.reports_dir)).resolve()
    dataset_name = HydraConfig.get().runtime.choices["dataset"]
    pkl_path = (
        reports_dir
        / f"metrics-stats_{dataset_name}_{cfg.hand.name}_{cfg.retargeter.name}.pkl"
    )
    if not pkl_path.is_file():
        raise SystemExit(
            f"Missing metrics pickle: {pkl_path}. Run compute_hand_retargeter_pair_metrics.py first "
            f"with the same hand / retargeter / config-name."
        )

    logger.info(f"Loading cached metrics from {pkl_path}")
    with open(pkl_path, "rb") as f:
        metrics_stats = pickle.load(f)

    robot_hand_model_type = RobotHandType(cfg.hand.name)
    retargeter_type = Retargeter(cfg.retargeter.name)

    app = Dash("Retargeting Dashboard", external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = Dashboard(metrics_stats).build(
        app, robot_hand_model_type.value, retargeter_type.value
    )

    dash_cfg = cfg.dashboard
    logger.info(
        f"Starting Dash server on {dash_cfg.host}:{dash_cfg.port} (debug={dash_cfg.debug}). "
        "Saving Python files will auto-restart this process (metrics are loaded from disk, so reload is cheap)."
    )
    # `use_reloader=True` restarts the Python process on any file change.
    # That's fine here because we don't recompute metrics — we just re-load
    # the pickle. This is the lever that makes code edits take effect.
    app.run(
        host=dash_cfg.host,
        port=dash_cfg.port,
        debug=dash_cfg.debug,
        use_reloader=True,
        dev_tools_hot_reload=True,
    )


if __name__ == "__main__":
    main()
