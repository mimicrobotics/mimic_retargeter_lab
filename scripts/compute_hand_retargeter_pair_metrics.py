# Import mimic_retargeter_lab first — its package init pins ``JAX_PLATFORMS``
# and silences MJX's misleading "Using JAX default device" log. Must come
# before ``import jax`` because JAX caches platform priority at import time.
import mimic_retargeter_lab  # noqa: F401

import json
import pickle
from datetime import datetime
from pathlib import Path

import dash_bootstrap_components as dbc
import hydra
from dash import Dash
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import jax

# Set the directory (relative to the script so it works anywhere)
cache_dir = Path(__file__).parent.parent / ".jax_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(cache_dir))
# Force JAX to cache EVERYTHING, ignoring the 1-second rule
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

from mimic_retargeter_lab.dashboard.dashboard import Dashboard
from mimic_retargeter_lab.hand_models import create_human_hand, create_robot_hand
from mimic_retargeter_lab.retargeting.online import create_retargeter
from mimic_retargeter_lab.types import Chirality, HumanHandType, RobotHandType, Retargeter
from mimic_retargeter_lab.utils import RetargetCache, configure_logging, get_logger


@hydra.main(config_path="../config", config_name="compute_metrics", version_base="1.2")
def main(cfg):
    configure_logging(level=cfg.logging.level)
    logger = get_logger(Path(__file__).stem)
    hand_path = Path(__file__).parent.parent / "assets" / "mjcf" / cfg.hand.name
    reports_dir = Path(hydra.utils.to_absolute_path(cfg.reports_dir)).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = HydraConfig.get().runtime.choices["dataset"]
    out_path = (
        reports_dir
        / f"metrics-stats_{dataset_name}_{cfg.hand.name}_{cfg.retargeter.name}.pkl"
    )
    # print(f"Resolved cfg: {OmegaConf.to_yaml(cfg, resolve=True)}")
    # Load robot hand model
    robot_hand_model_type = RobotHandType(cfg.hand.name)
    robot_hand_model = create_robot_hand(
        robot_hand_model_type, hand_path, Chirality(cfg.chirality)
    )
    human_hand_type = HumanHandType(cfg.get("human_hand", "mano_keypoint_hand"))
    human_hand_model = create_human_hand(
        human_hand_type,
        chirality=Chirality(cfg.chirality),
    )

    retargeter_type = Retargeter(cfg.retargeter.name)
    retargeter_cfg = OmegaConf.to_container(cfg.retargeter.config, resolve=True)

    retargeter = create_retargeter(
        retargeter_type,
        from_model=human_hand_model,
        to_model=robot_hand_model,
        **retargeter_cfg,
    )

    # Shared cache so metrics consuming the same (dataset, episode) retarget once.
    retarget_cache = RetargetCache(retargeter)

    # Collect computed stats for possible interactive inspection.
    # `metrics_meta` mirrors `metrics_stats` but holds provenance — which
    # data source each metric pulled from and which episodes it consumed
    # — so the sidecar JSON written below can answer "what went into this
    # pkl?" without re-loading the (large) pickle itself.
    metrics_stats = {}
    metrics_meta: dict[str, dict] = {}

    for metric_cfg in cfg.metrics.values():
        logger.info(f"Trying to compute {metric_cfg.config.display_name}")
        # Reset once per metric: episodes within a dataset still warm-start
        # from the prior episode's qpos (helps static reference poses), while
        # each metric starts from neutral so order doesn't bleed across datasets.
        retargeter.reset()

        metric = hydra.utils.instantiate(
            metric_cfg,
            human_hand_model=human_hand_model,
            robot_hand_model=robot_hand_model,
            retargeter=retargeter,
            retarget_cache=retarget_cache,
            _recursive_=False,
        )
        stats = metric.compute()
        logger.info(f"Computed stats for {stats.keys()} episodes")
        metrics_stats[metric.display_name] = stats
        # Capture the data-source provenance that fed this metric. Every
        # current metric has `self.data_source.data_path` but we degrade
        # gracefully (dataset_dir / str(...) / None) for any that don't.
        ds = getattr(metric, "data_source", None)
        data_path = (
            getattr(ds, "data_path", None) or getattr(ds, "dataset_dir", None)
            if ds is not None
            else None
        )
        episode_ids = [str(k) for k in stats.keys()]
        metrics_meta[metric.display_name] = {
            "data_path": str(data_path) if data_path is not None else None,
            "episode_ids": episode_ids,
            "num_episodes": len(episode_ids),
        }
        logger.info(f"Computed {metric.display_name}")

    pkl_saved = False
    try:
        with open(out_path, "wb") as f:
            pickle.dump(metrics_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
        pkl_saved = True
        logger.info(f"Saved computed metrics to {out_path}")

    except Exception as e:
        logger.warning(f"Failed to save metrics to {out_path}: {e}")

    # ── Sidecar metadata JSON ────────────────────────────────────────────
    # Lives next to the pkl with the same stem (`<...>.json`) so readers
    # can pair them by name. Records the run's timestamp and per-metric
    # data-source provenance — enough to backtrack which raw episodes
    # contributed to any number in the pkl without unpickling.
    meta_path = out_path.with_suffix(".json")
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "dataset": dataset_name,
            "hand": cfg.hand.name,
            "retargeter": cfg.retargeter.name,
            "chirality": str(cfg.chirality),
            "human_hand": cfg.get("human_hand", "mano_keypoint_hand"),
        },
        "metrics": metrics_meta,
        "pkl_path": out_path.name,
        "pkl_saved": pkl_saved,
        "pkl_size_bytes": (
            out_path.stat().st_size if pkl_saved and out_path.exists() else None
        ),
    }
    try:
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metrics metadata to {meta_path}")
    except Exception as e:
        logger.warning(f"Failed to save metadata to {meta_path}: {e}")

    # Optionally serve an interactive Dash dashboard to browse metrics now.
    # Controlled by config: set `serve_dashboard: true` and optionally provide `cfg.dashboard` settings.

    if cfg.serve_dashboard:
        # Build the Dash app and serve the dashboard layout.
        app = Dash("Retargeting Dashboard", external_stylesheets=[dbc.themes.BOOTSTRAP])
        app.layout = Dashboard(metrics_stats).build(
            app, robot_hand_model_type.value, retargeter_type.value
        )

        # Read port/debug from cfg.dashboard if present
        dash_cfg = cfg.dashboard
        logger.info(
            f"Starting Dash server on {dash_cfg.host}:{dash_cfg.port} (debug={dash_cfg.debug}) — press Ctrl+C to stop"
        )
        # `debug=True` + `use_reloader=False` gives browser-side hot reload
        # (edits in dashboard code refresh the tab) without restarting the
        # Python process — which matters because restarting would recompute
        # every metric. The callback lazily rebuilds figures, so clicking an
        # episode after saving will pick up the new code.
        app.run(
            host=dash_cfg.host,
            port=dash_cfg.port,
            debug=dash_cfg.debug,
            use_reloader=False,
            dev_tools_hot_reload=True,
        )


if __name__ == "__main__":
    main()
