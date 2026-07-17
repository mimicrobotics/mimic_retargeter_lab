"""mimic_retargeter_lab-side training entrypoint for GeoRT with the MJCF backend.

Reads `config/train_geort.yaml` (which itself pulls in the per-hand retargeter
yaml at `config/retargeter_cfg/geort/human_hand_to_<hand>.yaml`) to assemble:
  - to_model        : RobotHandModel built from the mimic_retargeter_lab MJCF
  - human_data      : (T, 25, 3) keypoint stream from the human mocap file
  - GeoRT config    : synthesized in memory (no urdf_path, no on-disk JSON)

The orchestration:
  1. Load + concatenate human mocap episodes into one (T, 25, 3) array.
  2. Compute the canonical Kabsch-Umeyama target (dataset mean of the
     configured `alignment.landmarks` on the human side).
  3. Align every frame to that canonical target (Umeyama scale optional).
  4. Synthesize the GeoRT-shaped config dict (joint_order from
     to_model.get_actuated_joint_names(); fingertip_link from the per-hand
     yaml + per-tracker `human_hand_ids`; alignment metadata for inference).
  5. cd into `third_party/geort/` so GeoRT's trainer writes its `data/<name>.npz`
     and `checkpoint/<name>_<datetime>_<exp>/` to the package's existing dirs.
  6. Run `MjcfGeoRTTrainer(config, to_model).train(aligned_data, ...)`.
  7. Optionally publish `last.pth` + `config.json` to `checkpoints/geort/<run>/`.

Run from repo root:
    JAX_PLATFORMS=cpu python -m mimic_retargeter_lab.retargeting.online.geort.train \\
        hand=mimic_p050_hand chirality=right tracker=manus \\
        human_data=./dataset/manus/manus_right_subject-RJM_run-010.npz \\
        exp_id=001
"""

import mimic_retargeter_lab  # noqa: F401  pins JAX_PLATFORMS

import json
import os
import shutil
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from mimic_retargeter_lab.data_sources.datasets import ManusNpzReader
from mimic_retargeter_lab.hand_models import HumanHandModel, create_robot_hand
from mimic_retargeter_lab.retargeting.online.geort import MjcfGeoRTTrainer
from mimic_retargeter_lab.types import Chirality, HandLandmark, RobotHandType
from mimic_retargeter_lab.utils import configure_logging, get_logger
from mimic_retargeter_lab.utils.retarget_utils import align_pcloud_kabsch_umeyama

REPO_ROOT = Path(__file__).resolve().parents[4]
GEORT_PACKAGE_ROOT = REPO_ROOT / "third_party" / "geort"
CHECKPOINTS_ROOT = REPO_ROOT / "checkpoints" / "geort"


def _load_human_data(
    human_data_path: Path, tracker: str
) -> tuple[np.ndarray, HumanHandModel, dict]:
    """Return (joints (T, 25, 3), from_model, dataset_metadata)."""
    if tracker == "manus":
        reader = ManusNpzReader(data_path=human_data_path)
        episodes = list(reader.get_episode_iter())
        if not episodes:
            raise ValueError(f"No episodes found in {human_data_path}")
        joints = np.concatenate([ep["joints"] for ep in episodes], axis=0)
        metadata_path = (
            human_data_path.parent / f"{human_data_path.stem}.json"
            if human_data_path.is_file()
            else None
        )
        metadata = (
            json.loads(metadata_path.read_text())
            if metadata_path and metadata_path.exists()
            else {}
        )
        return joints, reader.hand_model, metadata
    raise NotImplementedError(
        f"tracker={tracker!r} not wired up. Add a loader to _load_human_data."
    )


def _build_fingertip_link(
    fingertips_yaml: list[dict], human_hand_ids: dict[str, int], to_model
) -> list[dict]:
    """Assemble GeoRT's `fingertip_link` list from the per-hand yaml."""
    fingertip_link = []
    for f in fingertips_yaml:
        finger_name = f["name"]
        landmark = HandLandmark(str(f["landmark"]).lower())
        if landmark not in to_model._landmark_config:
            raise ValueError(
                f"Landmark {landmark.value!r} (for finger {finger_name!r}) "
                f"is not in to_model._landmark_config."
            )
        if finger_name not in human_hand_ids:
            raise ValueError(
                f"Tracker mapping is missing human_hand_id for finger {finger_name!r}."
            )
        fingertip_link.append(
            {
                "name": finger_name,
                "link": to_model._landmark_config[landmark].name,
                "joint": list(f["joint"]),
                "center_offset": list(f.get("center_offset", [0.0, 0.0, 0.0])),
                "human_hand_id": int(human_hand_ids[finger_name]),
            }
        )
    return fingertip_link


def _publish_checkpoint(
    run_dir: Path, dest_root: Path, target_name: str, logger
) -> Path:
    """Copy {config.json, last.pth} from a freshly-trained run into
    checkpoints/geort/<target_name>/. The target name is independent of the
    upstream run dir (which carries upstream's verbose datetime + auto-tag);
    we pick a clean cache_name + run-<exp_id> form here."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Trainer output dir not found: {run_dir}")
    target = dest_root / target_name
    target.mkdir(parents=True, exist_ok=True)
    for fname in ("config.json", "last.pth"):
        src = run_dir / fname
        if not src.exists():
            raise FileNotFoundError(f"Expected {src} after training; not present.")
        shutil.copy2(src, target / fname)
    logger.info(f"Published checkpoint to {target}")
    return target


@hydra.main(
    config_path="../../../../config", config_name="train_geort", version_base="1.2"
)
def main(cfg: DictConfig) -> None:
    configure_logging(level=cfg.logging.level)
    logger = get_logger("train_geort")

    hand_path = REPO_ROOT / "assets" / "mjcf" / cfg.hand.name
    to_model = create_robot_hand(
        RobotHandType(cfg.hand.name),
        hand_path,
        Chirality(cfg.chirality),
    )
    logger.info(
        f"Built to_model: {cfg.hand.name} ({cfg.chirality}); "
        f"actuated DoFs={to_model.num_actuated_dofs}"
    )

    # Per-hand retargeter yaml (carries fingertips, tracker indices, alignment).
    retargeter_cfg = OmegaConf.to_container(cfg.retargeter, resolve=True)
    inference_cfg = retargeter_cfg["config"]
    training_cfg = retargeter_cfg["training"]

    human_hand_ids = training_cfg["trackers"][cfg.tracker]["human_hand_ids"]
    fingertip_link = _build_fingertip_link(
        training_cfg["fingertips"], human_hand_ids, to_model
    )

    # Load human mocap stream and the matching from_model.
    human_data_path = Path(hydra.utils.to_absolute_path(cfg.human_data)).resolve()
    joints, from_model, dataset_metadata = _load_human_data(
        human_data_path, cfg.tracker
    )
    logger.info(
        f"Loaded human data {human_data_path.name}: shape={joints.shape}, "
        f"tracker={cfg.tracker}"
    )

    # Per-frame Kabsch-Umeyama alignment — matches GeortRetargeter at runtime.
    # Source: human landmarks for the configured `alignment_landmarks`.
    # Target: robot landmarks at NEUTRAL pose (deterministic across all
    # training frames). At inference, GeortRetargeter aligns to robot
    # landmarks at `_qpos_prev`, which starts at neutral, so frame 1
    # matches training exactly; later frames drift as `_qpos_prev` evolves.
    alignment_landmarks = list(inference_cfg["alignment_landmarks"])
    use_scale = bool(inference_cfg.get("alignment_use_scale", True))
    landmark_enums = [HandLandmark(str(lm).lower()) for lm in alignment_landmarks]

    neutral_landmarks = to_model.get_landmarks(qpos=to_model.get_neutral_qpos_pose())
    target_stack = np.stack(
        [np.asarray(neutral_landmarks[lm], dtype=np.float32) for lm in landmark_enums]
    )

    human_transforms = from_model.get_landmark_transforms(joints_3d=joints)
    aligned = np.empty_like(joints, dtype=np.float32)
    precomputed_scale = None if use_scale else 1.0
    for t in range(joints.shape[0]):
        src_stack = np.stack(
            [
                np.asarray(human_transforms[lm][t, :3, 3], dtype=np.float32)
                for lm in landmark_enums
            ]
        )
        aligned[t], _, _ = align_pcloud_kabsch_umeyama(
            points=joints[t],
            source_landmarks=src_stack,
            target_landmarks=target_stack,
            precomputed_scale=precomputed_scale,
        )
    logger.info(
        f"Aligned {joints.shape[0]} frames against robot neutral-pose landmarks "
        f"using {len(alignment_landmarks)} alignment_landmarks (use_scale={use_scale})."
    )

    # Synthesize the GeoRT-shaped config dict. Note: no urdf_path. The cache
    # name carries `_mjcf` so we don't collide with URDF-trained FK MLPs.
    # joint_order uses the unprefixed qpos-equivalent names — matches the
    # existing GeoRT URDF-trained checkpoint convention and the per-finger
    # `joint` lists in the yaml. MjcfHandKinematicModel accepts either form.
    joint_order = [
        to_model._joint_name_from_actuated_name(n)
        for n in to_model.get_actuated_joint_names()
    ]

    cache_name = f"{cfg.hand.name}_{cfg.chirality}_{cfg.tracker}_mjcf"
    geort_config = {
        "name": cache_name,
        "joint_order": joint_order,
        "fingertip_link": fingertip_link,
        # Recorded for run reproducibility; the inference adapter reads
        # alignment_landmarks from the yaml directly (not from this block).
        "alignment": {
            "landmarks": alignment_landmarks,
            "use_scale": use_scale,
            "target": "robot_neutral_pose_landmarks",
        },
    }

    # GeoRT's trainer writes outputs cwd-relative. cd into the package root
    # so its `data/` and `checkpoint/` dirs are reused.
    original_cwd = os.getcwd()
    os.chdir(GEORT_PACKAGE_ROOT)
    try:
        trainer = MjcfGeoRTTrainer(geort_config, to_model=to_model)
        train_kwargs = OmegaConf.to_container(cfg.training_kwargs, resolve=True)
        train_kwargs["exp_id"] = cfg.exp_id
        logger.info(
            f"Starting MjcfGeoRTTrainer.train (cache={cache_name}, kwargs={train_kwargs})"
        )
        trainer.train(
            dataset_positions_data=aligned,
            dataset_metadata=dataset_metadata or None,
            **train_kwargs,
        )
    finally:
        os.chdir(original_cwd)

    if cfg.auto_publish:
        # The trainer creates `checkpoint/<name>_<datetime>_<exp_tag>/` —
        # we don't know the timestamp it picked, but the directory matching
        # `<name>_<datetime>` and *not* ending in `_last` is the run we want.
        ckpt_root = GEORT_PACKAGE_ROOT / "checkpoint"
        candidates = [
            d
            for d in ckpt_root.iterdir()
            if d.is_dir()
            and d.name.startswith(cache_name)
            and not d.name.endswith("_last")
        ]
        if not candidates:
            logger.warning(
                f"auto_publish: no fresh run dir found under {ckpt_root} "
                f"matching {cache_name}_*. Skipping publish."
            )
            return
        latest = max(candidates, key=lambda d: d.stat().st_mtime)
        target_name = f"{cache_name}_run-{cfg.exp_id}"
        _publish_checkpoint(latest, CHECKPOINTS_ROOT, target_name, logger)


if __name__ == "__main__":
    main()
