"""Generate golden robot hand data for robot hand model regression tests.

Produces an .npz file containing:
  - pose_labels    : (N,) human-readable label per pose
  - ctrl_poses     : (N, A) ctrl-space joint angle sets
  - qpos_poses     : (N, Q) corresponding qpos-space joint angles (via joint_map)
  - fingertips     : (N, T, 4, 4) fingertip transforms per pose
  - fingertip_landmarks : (T,) fingertip landmark names (ordering for jacobians)
  - fingertip_jacobians : (N, T, 4, 4, A) d(T_tip)/d(ctrl) per pose (ordered by fingertip_landmarks)
  - keyvector_keys : list of keyvector dict keys (consistent ordering)
  - keyvectors     : (N, K, 3) keyvector values per pose
  - frame_names    : list of FK frame names
  - frame_poses    : (N, F, 4, 4) all FK frame transforms per pose
  - hand_type      : string identifier
  - chirality      : string identifier
  - qpos_joint_names     : (Q,) qpos joint names in joint_map row order
  - actuated_joint_names : (A,) actuated joint names in joint_map column order
  - joint_map            : (Q, A) mapping from actuated controls to qpos joints

Usage:
    python scripts/generate_robot_hand_golden_data.py --hand shadow_hand --chirality right
    python scripts/generate_robot_hand_golden_data.py --hand mimic_p050_hand --chirality right
    python scripts/generate_robot_hand_golden_data.py --all
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexworld.hand_models import ROBOT_HAND_REGISTRY, create_robot_hand
from dexworld.types import Chirality, HandLandmark, RobotHandType
from dexworld.utils import configure_logging, get_logger

ASSETS_PATH = REPO_ROOT / "assets" / "mjcf"
FIXTURES_PATH = REPO_ROOT / "tests" / "fixtures"

# Per-hand chirality coverage for golden generation. Hand types themselves
# come from ROBOT_HAND_REGISTRY; any registered hand missing here defaults
# to RIGHT only.
SUPPORTED_CHIRALITIES: dict[RobotHandType, list[Chirality]] = {
    hand_type: [Chirality.RIGHT] for hand_type in ROBOT_HAND_REGISTRY
}

NUM_RANDOM_POSES = 11


def generate_data(hand_type: RobotHandType, chirality: Chirality) -> Path:
    configure_logging()
    logger = get_logger(Path(__file__).stem)

    if hand_type not in ROBOT_HAND_REGISTRY:
        raise ValueError(
            f"Unsupported hand type: {hand_type}. "
            f"Supported: {list(ROBOT_HAND_REGISTRY.keys())}"
        )

    available = SUPPORTED_CHIRALITIES.get(hand_type, [Chirality.RIGHT])
    if chirality not in available:
        raise ValueError(
            f"{hand_type.value} does not have {chirality.value} chirality. "
            f"Available: {[c.value for c in available]}"
        )

    hand_path = ASSETS_PATH / hand_type.value
    model = create_robot_hand(hand_type, hand_path, chirality)

    rng = np.random.default_rng(seed=42)
    act_limits = model.get_actuated_joint_limits()
    act_names = model.get_actuated_joint_names()
    lo = np.array([act_limits[n][0] for n in act_names], dtype=np.float32)
    hi = np.array([act_limits[n][1] for n in act_names], dtype=np.float32)

    pose_labels = [
        "zero",
        "neutral",
        "min_limits",
        "max_limits",
        "midrange",
    ]
    ctrl_poses = [
        np.zeros(model.num_actuated_dofs, dtype=np.float32),
        model.get_neutral_ctrl_pose(),
        lo,
        hi,
        (lo + hi) / 2,
    ]

    for i in range(NUM_RANDOM_POSES):
        pose_labels.append(f"random_{i}")
        ctrl_poses.append(rng.uniform(lo, hi).astype(np.float32))

    ctrl_poses = np.stack(ctrl_poses)

    all_fingertips: list[np.ndarray] = []
    all_fingertip_jacobians: list[np.ndarray] = []
    all_keyvectors = []
    all_frame_poses = []
    all_qpos = []
    kv_keys = None
    frame_names = None
    # Canonical model-driven ordering (pinky optional depending on embodiment).
    fingertip_landmarks: list[HandLandmark] = model.get_fingertip_landmarks()
    if len(fingertip_landmarks) != model.get_num_fingertips():
        raise ValueError(
            "Fingertip landmark count mismatch: "
            f"landmarks={len(fingertip_landmarks)} num_fingertips={model.get_num_fingertips()} "
            f"for {hand_type.value}/{chirality.value}"
        )

    for i in range(ctrl_poses.shape[0]):
        ctrl = ctrl_poses[i : i + 1]  # (1, A)

        qpos = ctrl @ model.joint_map.T
        all_qpos.append(qpos.squeeze(0))

        # Fingertip positions via MJX FK body positions
        tip_pos = model.mjx_fk_body_positions(ctrl, joint_space="ctrl")
        tip_frames_4x4 = []
        for lm in fingertip_landmarks:
            link_name = model._landmark_config[lm][0]
            pos = np.asarray(tip_pos[link_name]).squeeze(0)
            frame = np.eye(4, dtype=np.float32)
            frame[:3, 3] = pos
            tip_frames_4x4.append(frame)
        all_fingertips.append(np.stack(tip_frames_4x4))

        J_dict = model.compute_fingertip_jacobians(ctrl)
        J = np.stack([np.asarray(J_dict[k]) for k in fingertip_landmarks], axis=0)
        all_fingertip_jacobians.append(J.astype(np.float32))

        kv = model.compute_keyvectors_jax(ctrl, joint_space="ctrl")
        if kv_keys is None:
            kv_keys = sorted(kv.keys())
        kv_arr = np.stack([np.asarray(kv[k]).squeeze(0) for k in kv_keys])
        all_keyvectors.append(kv_arr)

        fk_out, _ = model.to_kinematic_tree(
            ctrl, joint_space="ctrl", return_frame_dict=True
        )
        if frame_names is None:
            frame_names = sorted(fk_out.keys())
        frames = np.stack([fk_out[name] for name in frame_names])
        all_frame_poses.append(frames)

    FIXTURES_PATH.mkdir(parents=True, exist_ok=True)
    filename = f"{hand_type.value}_{chirality.value}_golden.npz"
    out_path = FIXTURES_PATH / filename

    np.savez(
        out_path,
        pose_labels=np.array(pose_labels),
        ctrl_poses=ctrl_poses,
        qpos_poses=np.stack(all_qpos),
        fingertips=np.stack(all_fingertips),
        fingertip_landmarks=np.array([str(lm) for lm in fingertip_landmarks]),
        fingertip_jacobians=np.stack(all_fingertip_jacobians),
        keyvector_keys=np.array(kv_keys),
        keyvectors=np.stack(all_keyvectors),
        frame_names=np.array(frame_names),
        frame_poses=np.stack(all_frame_poses),
        hand_type=hand_type.value,
        chirality=chirality.value,
        qpos_joint_names=np.array(model.get_qpos_joint_names()),
        actuated_joint_names=np.array(model.get_actuated_joint_names()),
        joint_map=np.asarray(model.joint_map, dtype=np.float32),
    )

    logger.info(
        f"[{hand_type.value} / {chirality.value}] Saved {ctrl_poses.shape[0]} poses to {out_path}"
    )
    logger.info(f"  pose_labels: {pose_labels}")
    logger.info(f"  ctrl_poses:  {ctrl_poses.shape}")
    logger.info(f"  qpos_poses:  {np.stack(all_qpos).shape}")
    logger.info(f"  fingertips:  {np.stack(all_fingertips).shape}")
    logger.info(
        f"  fingertip_jacobians:  {np.stack(all_fingertip_jacobians).shape}  "
        f"landmarks={[str(lm) for lm in fingertip_landmarks]}"
    )
    logger.info(f"  keyvectors:  {np.stack(all_keyvectors).shape}  keys={kv_keys}")
    logger.info(
        f"  frame_poses: {np.stack(all_frame_poses).shape}  ({len(frame_names)} frames)"
    )
    jm = np.asarray(model.joint_map)
    logger.info(
        f"  joint_map: {(jm.shape[0], jm.shape[1])}  "
        f"qpos_names/child_joint={len(model.get_qpos_joint_names())} "
        f"actuated_names/parent_joint={len(model.get_actuated_joint_names())}"
    )

    return out_path


def main():
    valid_hands = [h.value for h in ROBOT_HAND_REGISTRY]
    valid_chiralities = [c.value for c in Chirality]

    parser = argparse.ArgumentParser(
        description="Generate golden robot hand data for robot hand regression tests."
    )
    parser.add_argument(
        "--hand",
        type=str,
        choices=valid_hands,
        help="Hand model to generate data for.",
    )
    parser.add_argument(
        "--chirality",
        type=str,
        choices=valid_chiralities,
        default="right",
        help="Hand chirality (default: right).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate data for all supported hand/chirality combinations.",
    )
    args = parser.parse_args()

    if args.all:
        for hand_type in ROBOT_HAND_REGISTRY:
            for chirality in SUPPORTED_CHIRALITIES.get(hand_type, [Chirality.RIGHT]):
                generate_data(hand_type, chirality)
                print()
    elif args.hand:
        hand_type = RobotHandType(args.hand)
        chirality = Chirality(args.chirality)
        generate_data(hand_type, chirality)
    else:
        parser.error("Provide --hand <name> or --all")


if __name__ == "__main__":
    main()
