#!/usr/bin/env python3
"""
Sanity check: ensure all Mimic keyvector YAML tgt/src keys exist.

Usage:
  python scripts/check_mimic_keyvector_cfg.py
"""

from pathlib import Path

import numpy as np
import yaml

from mimic_retargeter_lab.hand_models import ManoKeypointHandModel, MimicP050HandModel
from mimic_retargeter_lab.types import Chirality
from mimic_retargeter_lab.utils import configure_logging, get_logger


def main():
    configure_logging()
    logger = get_logger(Path(__file__).stem)
    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = (
        repo_root
        / "config"
        / "retargeter"
        / "keyvector"
        / "human_hand_to_mimic_p050_hand.yaml"
    )
    mjcf_dir = repo_root / "assets" / "mjcf" / "mimic_p050_hand"

    # 1) Load config
    cfg = yaml.safe_load(cfg_path.read_text())
    kv_cfg = cfg["config"]["keyvectors_cfg"]

    # 2) Build robot keyvectors at zero pose
    robot = MimicP050HandModel(mjcf_dir, Chirality.RIGHT)
    robot_ctrl = np.zeros((1, robot.num_actuated_dofs), dtype=np.float32)
    robot_kv = robot.compute_keyvectors(robot_ctrl, joint_space="ctrl")
    robot_keys = set(robot_kv.keys())

    # 3) Build human keyvectors on a dummy joint batch
    human = ManoKeypointHandModel(
        chirality=Chirality.RIGHT,
    )
    human_joints = np.zeros((1, 21, 3), dtype=np.float32)
    human_kv = human.compute_keyvectors(human_joints)
    human_keys = set(human_kv.keys())

    missing_tgt = []
    missing_src = []
    logger.info("=== Checking keyvectors_cfg entries ===")
    for item in kv_cfg:
        name = item["name"]
        src = item["src_key"]
        tgt = item["tgt_key"]

        src_ok = src in human_keys
        tgt_ok = tgt in robot_keys

        if not src_ok:
            missing_src.append((name, src))
        if not tgt_ok:
            missing_tgt.append((name, tgt))

        logger.info(
            f"- {name:24s} src={'OK' if src_ok else 'MISSING'} "
            f"tgt={'OK' if tgt_ok else 'MISSING'}"
        )

    logger.info("\n=== Summary ===")
    logger.info(f"human keys available: {len(human_keys)}")
    logger.info(f"robot keys available: {len(robot_keys)}")
    logger.info(f"missing src keys: {len(missing_src)}")
    logger.info(f"missing tgt keys: {len(missing_tgt)}")

    if missing_src:
        logger.info("\nMissing src keys:")
        for name, key in missing_src:
            logger.info(f"  - {name}: {key}")

    if missing_tgt:
        logger.info("\nMissing tgt keys:")
        for name, key in missing_tgt:
            logger.info(f"  - {name}: {key}")

    logger.info("\nSample robot keys:")
    for k in sorted(list(robot_keys))[:20]:
        logger.info(f"  {k}")

    if missing_src or missing_tgt:
        raise SystemExit(1)

    logger.info("\nAll keyvector mappings look valid.")


if __name__ == "__main__":
    main()
