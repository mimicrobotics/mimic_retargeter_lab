# Measure trajectory flatness (smoothness) via second-order finite differences.
#
# Flatness = E[ || FK∘f(x+d) + FK∘f(x-d) - 2·FK∘f(x) ||² ]
#
# This is the squared acceleration magnitude. Lower values indicate smoother
# trajectories with fewer jumps or jitter.


import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.ndimage import gaussian_filter1d

from dexworld.hand_models import HumanHandModel, RobotHandModel
from dexworld.retargeting.online import BaseOnlineRetargeter
from dexworld.types.types import HandLandmark
from dexworld.utils import RetargetCache

from ._stats import summarize_array
from .base_metric import BaseMetric


class FlatnessMetric(BaseMetric):
    def __init__(
        self,
        config,
        human_hand_model: HumanHandModel,
        robot_hand_model: RobotHandModel,
        retargeter: BaseOnlineRetargeter,
        data_source_cfg: DictConfig,
        retarget_cache: RetargetCache | None = None,
    ):
        self.display_name = config.display_name
        self.task_space_mapping = config.task_space_mapping

        self.retargeter = retargeter
        self.retarget_cache = retarget_cache
        self.human_hand_model = human_hand_model
        self.robot_hand_model = robot_hand_model

        self.data_source = hydra.utils.instantiate(data_source_cfg)

    def compute(self):
        episode_metrics = {}
        for episode_data in self.data_source.get_episode_iter():
            human_joints_3d = episode_data["joints"]
            cache_key = (
                str(self.data_source.data_path),
                str(episode_data["episode_id"]),
            )
            self.retargeter.reset()
            robot_joint_angles_actuated = self.retarget_cache.get(
                cache_key, human_joints_3d
            )
            robot_joint_angles_actuated = np.asarray(
                robot_joint_angles_actuated, dtype=np.float32
            )

            human_landmarks = self.human_hand_model.get_landmark_transforms(
                joints_3d=human_joints_3d
            )
            robot_landmarks = self.robot_hand_model.get_landmark_transforms(
                joint_angles=robot_joint_angles_actuated,
                joint_space="ctrl",
            )

            flatness_per_frame = {}
            for fmap in self.task_space_mapping:
                lm = HandLandmark(fmap["landmark"].lower())
                human_T = human_landmarks[lm]
                robot_T = robot_landmarks[lm]

                human_pos = (
                    human_T[:, :3, 3] if human_T.ndim == 3 else human_T[:3, 3][None, :]
                )
                robot_pos = (
                    robot_T[:, :3, 3] if robot_T.ndim == 3 else robot_T[:3, 3][None, :]
                )

                flatness_per_frame[lm.value] = {
                    "human": _compute_flatness(human_pos),
                    "robot": _compute_flatness(robot_pos),
                }

            episode_metrics[episode_data["episode_id"]] = flatness_per_frame

        return episode_metrics


def _compute_flatness(positions, sigma=2.0):
    """Compute second-order finite difference (acceleration) magnitude.

    positions: (T, 3)

    Returns dict with:
        accel_norm: (T-2,) squared norm of acceleration at each interior timestep
        smoothed: (T-2,) Gaussian-smoothed version
        mean: scalar mean squared acceleration
        max: scalar max squared acceleration
    """
    if len(positions) < 3:
        return {
            "accel_norm_sq": np.array([]),
            "smoothed": np.array([]),
            **summarize_array(np.array([])),
        }

    # Second-order central difference: pos[t+1] + pos[t-1] - 2*pos[t]
    accel = positions[2:] + positions[:-2] - 2 * positions[1:-1]  # (T-2, 3)
    accel_norm_sq = np.sum(accel**2, axis=1)  # (T-2,)

    smoothed = gaussian_filter1d(accel_norm_sq, sigma=sigma, mode="nearest")

    return {
        "accel_norm_sq": accel_norm_sq,
        "smoothed": smoothed,
        **summarize_array(accel_norm_sq),
    }
