"""Debug script: visualize self-collisions in retargeted poses.

For each frame in a retargeted sequence, runs mj_forward and reports
any active contacts (penetrating geom pairs). Helps verify whether
collision detection works smoothly before adding it to the loss.

Usage:
    python scripts/debug_collisions.py hand=shadow_hand retargeter=dexpilot
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import hydra
import mujoco
import numpy as np
from omegaconf import OmegaConf

from mimic_retargeter_lab.data_sources import HandDatasetReader
from mimic_retargeter_lab.hand_models import create_human_hand, create_robot_hand
from mimic_retargeter_lab.retargeting.online import create_retargeter
from mimic_retargeter_lab.types import (
    Chirality,
    HandDataset,
    HumanHandType,
    Retargeter,
    RobotHandType,
)
from mimic_retargeter_lab.utils import retarget_points_sequence


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
    print(f"collision_cfg in config: {'collision_cfg' in ret_cfg}")
    if "collision_cfg" in ret_cfg:
        print(f"  weight: {ret_cfg['collision_cfg']['weight']}")
        print(f"  pairs: {len(ret_cfg['collision_cfg']['pairs'])}")
    retargeter = create_retargeter(
        Retargeter(cfg.retargeter.name),
        from_model=human,
        to_model=robot,
        **ret_cfg,
    )

    # Load the MuJoCo model for contact detection
    orig_cwd = os.getcwd()
    os.chdir(robot.robot_hand_base_path)
    try:
        mj_model = mujoco.MjModel.from_xml_path(str(robot.hand_model_path))
    finally:
        os.chdir(orig_cwd)
    mj_data = mujoco.MjData(mj_model)

    # Helper: get a readable label for a geom (geom_name or body_name/geom_id)
    def geom_label(geom_id):
        gname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body_id = mj_model.geom_bodyid[geom_id]
        bname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if gname:
            return f"{bname}/{gname}"
        return f"{bname}/geom_{geom_id}"

    # Load pinch grasps data
    ds = HandDatasetReader(
        data_path=Path(hydra.utils.get_original_cwd()) / "dataset/pinch_grasps",
        dataset=HandDataset("pinch_grasps_test"),
    )

    for ep in ds.get_episode_iter():
        episode_id = ep["episode_id"]
        joints = ep["joints"]

        print(f"\n{'=' * 80}")
        print(f"Episode: {episode_id}")
        print(f"{'=' * 80}")

        # Retarget
        retargeter.reset()
        robot_q = retarget_points_sequence(retargeter, joints)
        robot_q = np.asarray(robot_q, dtype=np.float32)

        # Check contacts at each frame
        for t in range(robot_q.shape[0]):
            ctrl = robot_q[t]
            qpos = ctrl @ robot.joint_map.T
            mj_data.qpos[:] = qpos
            mujoco.mj_forward(mj_model, mj_data)

            n_contacts = mj_data.ncon
            penetrating = []
            for c in range(n_contacts):
                contact = mj_data.contact[c]
                if contact.dist < 0:  # negative = penetration
                    g1 = geom_label(contact.geom1)
                    g2 = geom_label(contact.geom2)
                    penetrating.append((g1, g2, contact.dist))

            if penetrating:
                print(f"\n  Frame {t}: {len(penetrating)} penetrating contact(s)")
                for g1, g2, dist in sorted(penetrating, key=lambda x: x[2]):
                    print(f"    {g1:>30} <-> {g2:<30}  depth = {-dist * 1000:.2f} mm")
            else:
                print(f"  Frame {t}: no collisions")

    # Also test with video data if available
    video_path = Path(hydra.utils.get_original_cwd()) / "dataset/wilor"
    if video_path.exists():
        print(f"\n\n{'#' * 80}")
        print("Testing with video data (first episode, sampled frames)")
        print(f"{'#' * 80}")

        ds_video = HandDatasetReader(
            data_path=video_path,
            dataset=HandDataset("wilor_test_long"),
            num_episodes=1,
        )

        for ep in ds_video.get_episode_iter():
            joints = ep["joints"]
            print(f"\nEpisode: {ep['episode_id']} ({joints.shape[0]} frames)")

            retargeter.reset()
            robot_q = retarget_points_sequence(retargeter, joints)
            robot_q = np.asarray(robot_q, dtype=np.float32)

            total_penetrations = 0
            max_depth_mm = 0.0
            collision_pairs = {}

            for t in range(robot_q.shape[0]):
                ctrl = robot_q[t]
                qpos = ctrl @ robot.joint_map.T
                mj_data.qpos[:] = qpos
                mujoco.mj_forward(mj_model, mj_data)

                for c in range(mj_data.ncon):
                    contact = mj_data.contact[c]
                    if contact.dist < 0:
                        g1 = geom_label(contact.geom1)
                        g2 = geom_label(contact.geom2)
                        pair = (min(g1, g2), max(g1, g2))
                        depth_mm = -contact.dist * 1000
                        total_penetrations += 1
                        max_depth_mm = max(max_depth_mm, depth_mm)
                        if pair not in collision_pairs:
                            collision_pairs[pair] = {"count": 0, "max_depth_mm": 0.0}
                        collision_pairs[pair]["count"] += 1
                        collision_pairs[pair]["max_depth_mm"] = max(
                            collision_pairs[pair]["max_depth_mm"], depth_mm
                        )

            print(
                f"  Total penetrating contacts across all frames: {total_penetrations}"
            )
            print(f"  Max penetration depth: {max_depth_mm:.2f} mm")
            if collision_pairs:
                print(f"  Unique colliding pairs: {len(collision_pairs)}")
                print(
                    f"\n  {'Geom 1':>30}  {'Geom 2':<30}  {'Count':>6}  {'Max Depth':>10}"
                )
                print(f"  {'-' * 30}  {'-' * 30}  {'-' * 6}  {'-' * 10}")
                for (g1, g2), info in sorted(
                    collision_pairs.items(), key=lambda x: -x[1]["max_depth_mm"]
                ):
                    print(
                        f"  {g1:>30}  {g2:<30}  {info['count']:>6}  "
                        f"{info['max_depth_mm']:>8.2f} mm"
                    )


if __name__ == "__main__":
    main()
