# Measure directional alignment of velocity vectors between human and robot in task space


import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.ndimage import gaussian_filter1d

from mimic_retargeter_lab.hand_models import HumanHandModel, RobotHandModel
from mimic_retargeter_lab.retargeting.online import BaseOnlineRetargeter
from mimic_retargeter_lab.types.types import HandLandmark
from mimic_retargeter_lab.utils import RetargetCache
from mimic_retargeter_lab.utils.retarget_utils import compute_kabsch_umeyama_transform

from ._stats import summarize_array
from .base_metric import BaseMetric


class MotionPreservationMetric(BaseMetric):
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

    def _align_human_to_robot(self, human_landmarks, robot_landmarks):
        """Compute per-frame Kabsch-Umeyama alignment from human frame to robot frame.

        Uses shared landmark positions (wrist, fingertip bases, tips) as
        correspondences to find rotation + scale that maps human coordinates
        into the MuJoCo reference frame.

        Returns:
            rotations: (T, 3, 3) rotation matrices
            scales: (T,) scale factors
            translations: (T, 3) translation vectors (robot_centroid - scale * R @ human_centroid)
        """
        # Collect shared landmarks present in both
        shared_landmarks = [
            lm for lm in HandLandmark if lm in human_landmarks and lm in robot_landmarks
        ]

        def _get_pos(transforms, lm):
            T = transforms[lm]
            if T.ndim == 3:
                return T[:, :3, 3]
            return T[:3, 3][None, :]

        # Stack shared landmark positions: (T, M, 3)
        human_pts = np.stack(
            [_get_pos(human_landmarks, lm) for lm in shared_landmarks], axis=1
        )
        robot_pts = np.stack(
            [_get_pos(robot_landmarks, lm) for lm in shared_landmarks], axis=1
        )

        T = human_pts.shape[0]
        rotations = np.zeros((T, 3, 3))
        scales = np.zeros(T)
        translations = np.zeros((T, 3))

        for t in range(T):
            h_pts = human_pts[t]  # (M, 3)
            r_pts = robot_pts[t]  # (M, 3)

            h_centroid = h_pts.mean(axis=0)
            r_centroid = r_pts.mean(axis=0)

            R, s = compute_kabsch_umeyama_transform(
                h_pts - h_centroid, r_pts - r_centroid
            )
            rotations[t] = R
            scales[t] = s
            translations[t] = r_centroid - s * (R @ h_centroid)

        return rotations, scales, translations

    def _transform_positions(self, positions, rotations, scales, translations):
        """Apply per-frame Kabsch-Umeyama transform to a position trajectory.

        positions: (T, 3)
        Returns: (T, 3) aligned positions
        """
        aligned = np.zeros_like(positions)
        for t in range(len(positions)):
            aligned[t] = scales[t] * (rotations[t] @ positions[t]) + translations[t]
        return aligned

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

            # Get landmark transforms for Kabsch alignment and per-finger tracking
            human_landmarks = self.human_hand_model.get_landmark_transforms(
                joints_3d=human_joints_3d
            )
            robot_landmarks = self.robot_hand_model.get_landmark_transforms(
                joint_angles=robot_joint_angles_actuated,
                joint_space="ctrl",
            )

            # Compute per-frame alignment from human to robot (MuJoCo) frame
            rotations, scales, translations = self._align_human_to_robot(
                human_landmarks, robot_landmarks
            )

            alignment_per_frame = {}
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

                # Align human positions into robot's MuJoCo frame
                human_pos_aligned = self._transform_positions(
                    human_pos, rotations, scales, translations
                )

                alignment_per_frame[lm.value] = {
                    "pos_alignment": self._compute_directional_alignment(
                        human_pos_aligned, robot_pos
                    ),
                }

            episode_metrics[episode_data["episode_id"]] = alignment_per_frame

        return episode_metrics

    def _compute_directional_alignment(self, pos_a, pos_b, sigma=2.0):
        """Compute cosine similarity between displacement vectors at each timestep.

        pos_a: (T, 3) — e.g. human positions (aligned to robot frame)
        pos_b: (T, 3) — e.g. robot positions

        Returns dict with raw/smoothed cosine similarity time series and summary stats.
        """
        if len(pos_a) < 2:
            empty = np.array([])
            return {
                "raw": empty,
                "smoothed": empty,
                **summarize_array(empty),
            }

        da = np.diff(pos_a, axis=0)  # (T-1, 3)
        db = np.diff(pos_b, axis=0)  # (T-1, 3)

        norm_a = np.linalg.norm(da, axis=1)
        norm_b = np.linalg.norm(db, axis=1)

        # Add small epsilon to avoid division by zero, but compute every timestep
        eps = 1e-8
        dot = np.sum(da * db, axis=1)
        cos_sim = dot / (norm_a * norm_b + eps)

        # Clip to [-1, 1] to handle numerical edge cases
        cos_sim = np.clip(cos_sim, -1.0, 1.0)

        smoothed = gaussian_filter1d(cos_sim, sigma=sigma, mode="nearest")

        return {
            "raw": cos_sim,
            "smoothed": smoothed,
            **summarize_array(cos_sim),
        }
