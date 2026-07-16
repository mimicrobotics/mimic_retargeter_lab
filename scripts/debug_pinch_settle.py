"""Debug script: check how many settle iterations the retargeter needs to converge on a pinch."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf

from dexworld.data_sources import HandDatasetReader
from dexworld.hand_models import create_human_hand, create_robot_hand
from dexworld.retargeting.online import create_retargeter
from dexworld.types import (
    Chirality,
    HandDataset,
    HandLandmark,
    HumanHandType,
    Retargeter,
    RobotHandType,
)


@hydra.main(config_path="../config", config_name="compute_metrics", version_base="1.2")
def main(cfg):
    hand_path = Path(hydra.utils.get_original_cwd()) / "assets" / "mjcf" / cfg.hand.name
    robot = create_robot_hand(
        RobotHandType(cfg.hand.name), hand_path, Chirality(cfg.chirality)
    )
    human = create_human_hand(
        HumanHandType.MANO_KEYPOINT_HAND,
        chirality=Chirality(cfg.chirality),
    )
    ret_cfg = OmegaConf.to_container(cfg.retargeter.config, resolve=True)
    retargeter = create_retargeter(
        Retargeter(cfg.retargeter.name),
        from_model=human,
        to_model=robot,
        **ret_cfg,
    )

    ds = HandDatasetReader(
        data_path=Path(hydra.utils.get_original_cwd()) / "dataset/pinch_grasps",
        dataset=HandDataset("pinch_grasps_test"),
    )

    # Fingertip landmarks to measure distances
    tips = [
        ("index", HandLandmark.INDEX_TIP),
        ("middle", HandLandmark.MIDDLE_TIP),
        ("ring", HandLandmark.RING_TIP),
        ("pinky", HandLandmark.PINKY_TIP),
    ]

    for ep in ds.get_episode_iter():
        episode_id = ep["episode_id"]
        joints = ep["joints"]
        frame = np.asarray(joints[0], dtype=np.float32)

        print(f"\n{'=' * 70}")
        print(f"Episode: {episode_id}")
        print(f"{'=' * 70}")
        print(f"{'Iter':>4}  ", end="")
        for name, _ in tips:
            print(f"{'thumb-' + name:>16} mm", end="")
        print("   grasp states")
        print("-" * 100)

        retargeter.reset()
        for i in range(15):
            q, _ = retargeter.retarget(frame)
            tgt_lm = robot.get_landmarks(qpos=np.asarray(q[0], dtype=np.float32))
            thumb = tgt_lm[HandLandmark.THUMB_TIP]

            print(f"{i:4d}  ", end="")
            for name, lm in tips:
                tip_pos = tgt_lm[lm]
                dist_mm = float(np.linalg.norm(thumb - tip_pos)) * 1000
                print(f"{dist_mm:16.1f}  ", end="")

            grasp = {k.split("_to_")[1]: v for k, v in retargeter._grasp_states.items()}
            print(f"  {grasp}")


if __name__ == "__main__":
    main()
