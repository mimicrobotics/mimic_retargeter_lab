import hydra
from omegaconf import DictConfig

from mimic_retargeter_lab.data_sources import OakDHandTracker


@hydra.main(
    config_path="../config/hand_tracker",
    config_name="oakd_mediapipe",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    tracker = OakDHandTracker(cfg)
    # tracker.run_blocking_viz()
    for hands in tracker.run_non_blocking():
        if hands is None:
            continue
        print(f"detected hands with shape {hands.shape}")


if __name__ == "__main__":
    main()
