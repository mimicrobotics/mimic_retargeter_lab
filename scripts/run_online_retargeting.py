#!/usr/bin/env python3
"""
Run online retargeting using Hydra configuration.

Streams live hand data from a hand tracker (e.g. OAK-D, MANUS) through a
retargeter and visualises the result in MuJoCo.

Usage:
    python scripts/run_online_retargeting.py                           # defaults (OAK-D + joint_angle)
    python scripts/run_online_retargeting.py hand_tracker=manus_metagloves_pro retargeter=dexpilot
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from mimic_retargeter_lab.scenes import KinematicRetargetingScene
from mimic_retargeter_lab.types import Chirality, RobotHandType, Retargeter, Simulator
from mimic_retargeter_lab.utils import KBHit


@hydra.main(
    config_path="../config", config_name="online_retargeting", version_base="1.2"
)
def main(cfg: DictConfig) -> None:
    hand_path = Path(__file__).parent.parent / "assets" / "mjcf" / cfg.hand.name

    # Instantiate the hand tracker from Hydra config (_target_ in hand_tracker yaml)
    hand_data_source = hydra.utils.instantiate(cfg.hand_tracker)

    # Optionally instantiate a wrist controller (e.g., SpaceMouse) for wrist control.
    wrist_data_source = None
    if cfg.get("wrist_controller") is not None:
        wrist_data_source = hydra.utils.instantiate(cfg.wrist_controller)

    keyboard_handler = KBHit()

    scene = KinematicRetargetingScene(
        hand_type=RobotHandType(cfg.hand.name),
        robot_hand_base_path=hand_path,
        chirality=Chirality(cfg.chirality),
        hand_data_source=hand_data_source,
        simulator=Simulator(cfg.simulator),
        retargeter=Retargeter(cfg.retargeter.name),
        retargeter_cfg=OmegaConf.to_container(cfg.retargeter.config),
        wrist_data_source=wrist_data_source,
    )

    num_steps = cfg.get("num_steps", None)
    idx = 0
    paused = False
    try:
        while True:
            if num_steps is not None and idx > num_steps:
                break

            if keyboard_handler.kbhit():
                key = keyboard_handler.getch()
                match key:
                    case "q":
                        print("Exiting...")
                        break
                    case "p":
                        paused = not paused
                        print(f"Paused: {paused}")
                    case _:
                        pass
            scene.step(update_data=not paused)
            idx += 1
    except KeyboardInterrupt:
        print("\n[run_retargeting] Interrupted by user. Exiting.")


if __name__ == "__main__":
    main()
