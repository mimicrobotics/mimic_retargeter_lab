#!/usr/bin/env python3
"""
Run offline retargeting using Hydra configuration.

This script expects `mimic_retargeter_lab/config/offline_retargeting.yaml` to use Hydra
interpolation so that `retargeter_cfg` is already the mapping for the chosen
retargeter (e.g. `retargeter_cfg: ${retargeters.${retargeter}}`).

Behavior:
- If `cfg.retargeter_cfg` is a mapping, it will be converted to a plain dict
  and passed to `RetargetingScene`.
- If `cfg.retargeter_cfg` is a string (legacy), the runner will not attempt to
  load files and will treat it as unspecified (empty dict). Prefer Hydra
  interpolation (recommended).
"""

# Import mimic_retargeter_lab first — its package init pins ``JAX_PLATFORMS``
# and silences MJX's misleading "Using JAX default device" log. Must come
# before ``import jax`` because JAX caches platform priority at import time.
import mimic_retargeter_lab  # noqa: F401

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import jax

# Set the directory (relative to the script so it works anywhere)
cache_dir = Path(__file__).parent.parent / ".jax_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(cache_dir))
# Force JAX to cache EVERYTHING, ignoring the 1-second rule
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

from mimic_retargeter_lab.data_sources import HandDatasetReader
from mimic_retargeter_lab.data_sources.datasets import ManusNpzReader
from mimic_retargeter_lab.scenes import KinematicRetargetingScene
from mimic_retargeter_lab.types import Chirality, RobotHandType, Retargeter, Simulator
from mimic_retargeter_lab.types.types import HandDataset
from mimic_retargeter_lab.utils import configure_logging, get_logger


@hydra.main(
    config_path="../config", config_name="offline_retargeting", version_base="1.2"
)
def main(cfg: DictConfig) -> None:
    configure_logging(level=cfg.logging.level)
    logger = get_logger(Path(__file__).stem)

    # Resolve the robot hand assets path relative to the repository layout
    hand_path = Path(__file__).parent.parent / "assets" / "mjcf" / cfg.hand.name

    # Resolve data path relative to the original working directory (Hydra may change cwd)
    # keyboard_handler = KBHit()

    if cfg.use_offline_data:
        data_base_path = Path(
            hydra.utils.to_absolute_path(cfg.data_base_path)
        ).resolve()
        offline_source = getattr(cfg, "offline_source", "wilor")
        if offline_source == "manus":
            hand_data_source = ManusNpzReader(
                data_path=data_base_path,
                num_episodes=getattr(cfg, "num_episodes", None),
            )
        elif offline_source == "wilor":
            hand_data_source = HandDatasetReader(
                data_path=data_base_path, dataset=HandDataset(cfg.hand_dataset)
            )
        else:
            raise ValueError(
                f"Unknown offline_source={offline_source!r}; expected 'wilor' or 'manus'."
            )
    else:
        hand_data_source = hydra.utils.instantiate(cfg.hand_tracker)

    retargeter_cfg = OmegaConf.to_container(cfg.retargeter.config, resolve=True)

    # Create the RetargetingScene with resolved config values
    scene = KinematicRetargetingScene(
        hand_type=RobotHandType(cfg.hand.name),
        robot_hand_base_path=hand_path,
        chirality=Chirality(cfg.chirality),
        hand_data_source=hand_data_source,
        simulator=Simulator(cfg.simulator),
        retargeter=Retargeter(cfg.retargeter.name),
        retargeter_cfg=retargeter_cfg,
        num_episodes=getattr(cfg, "num_episodes", 1),
    )

    # Run loop: either for a fixed number of steps or until interrupted.
    num_steps = cfg.get("num_steps", None)
    idx = 0
    paused = False
    try:
        logger.info("Starting retargeting loop. Press Ctrl+C to stop.")
        while True:
            if num_steps is not None and idx > num_steps:
                break

            # if keyboard_handler.kbhit():
            #     key = keyboard_handler.getch()
            #     match key:
            #         case "q":
            #             logger.info("Exiting...")
            #             break
            #         case "p":
            #             paused = not paused
            #             logger.info(f"Paused: {paused}")
            #         case _:
            #             pass
            scene.step(update_data=not paused)
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting.")


if __name__ == "__main__":
    main()
