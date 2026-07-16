"""Keyvector Matching as a per-frame time-series metric.

Mirrors the structure of ``MotionPreservationMetric`` and ``FlatnessMetric``:
runs over the same continuous dataset that the other stream metrics consume,
applies per-frame Kabsch–Umeyama alignment of the human landmarks into the
robot's MuJoCo frame, and emits per-frame time series for each configured
``vector_diffs`` entry.

Output schema (per ``episode_id``):

    {
        "vector_metrics": {
            "thumb_to_index_tip": {
                "cosine_similarity": {"raw": (T,), "smoothed": (T,), "mean": float, "std": float},
                "angle_error_deg":   {"raw": (T,), "smoothed": (T,), "mean": float, "std": float},
                "length_error_mm":   {"raw": (T,), "smoothed": (T,), "mean": float, "std": float},
                "scale_ratio":       {"raw": (T,), "smoothed": (T,), "mean": float, "std": float},
            },
            ...
        },
    }

Pinch Grasps still uses the static-pose ``ReferencePoseMetric``; this class
does not replace that — it replaces the static-pose Keyvector Matching
formulation, which collapsed everything to scalars.
"""

from typing import Any, Dict

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.ndimage import gaussian_filter1d

from dexworld.hand_models import HumanHandModel, RobotHandModel
from dexworld.retargeting.online import BaseOnlineRetargeter
from dexworld.types import HandLandmark
from dexworld.utils import RetargetCache
from dexworld.utils.retarget_utils import compute_kabsch_umeyama_transform

from ._stats import summarize_array
from .base_metric import BaseMetric


_SMOOTH_SIGMA_FRAMES = 3.0
_ERROR_KEYS = (
    "cosine_similarity",
    "angle_error_deg",
    "length_error_mm",
    "scale_ratio",
)


class KeyvectorMatchingMetric(BaseMetric):
    def __init__(
        self,
        config: DictConfig,
        human_hand_model: HumanHandModel,
        robot_hand_model: RobotHandModel,
        retargeter: BaseOnlineRetargeter,
        data_source_cfg: DictConfig,
        retarget_cache: RetargetCache | None = None,
    ):
        self.display_name = config.display_name
        self.vector_diff_config = config.vector_diffs

        self.retargeter = retargeter
        self.retarget_cache = retarget_cache
        self.human_hand_model = human_hand_model
        self.robot_hand_model = robot_hand_model

        self.data_source = hydra.utils.instantiate(data_source_cfg)

    # ── Per-episode driver ────────────────────────────────────────────

    def compute(self):
        episode_metrics: Dict[str, Any] = {}
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

            episode_metrics[episode_data["episode_id"]] = {
                "vector_metrics": self._compute_vector_metrics(
                    human_joints_3d, robot_joint_angles_actuated
                ),
            }
        return episode_metrics

    # ── Time-series error computation ─────────────────────────────────

    def _compute_vector_metrics(
        self, human_joints_3d: np.ndarray, robot_joint_angles_actuated: np.ndarray
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        human_transforms = self.human_hand_model.get_landmark_transforms(
            joints_3d=human_joints_3d
        )
        robot_transforms = self.robot_hand_model.get_landmark_transforms(
            joint_angles=robot_joint_angles_actuated, joint_space="ctrl"
        )

        def _get_pos(lm_transforms, lm):
            T = lm_transforms[lm]
            if T.ndim == 3:
                return T[:, :3, 3]
            return T[:3, 3][None, :]

        # Per-frame Kabsch–Umeyama: align human landmarks into the robot's
        # frame, applied independently each timestep. Same pattern as
        # `ReferencePoseMetric._compute_reference_pose_metrics`.
        shared_landmarks = [
            lm
            for lm in HandLandmark
            if lm in human_transforms and lm in robot_transforms
        ]
        human_pts = np.stack(
            [_get_pos(human_transforms, lm) for lm in shared_landmarks], axis=1
        )  # (T, M, 3)
        robot_pts = np.stack(
            [_get_pos(robot_transforms, lm) for lm in shared_landmarks], axis=1
        )  # (T, M, 3)
        T_frames = human_pts.shape[0]

        aligned_human_positions: Dict[HandLandmark, np.ndarray] = {
            lm: np.zeros((T_frames, 3), dtype=np.float64) for lm in shared_landmarks
        }
        for t in range(T_frames):
            h_pts = human_pts[t]
            r_pts = robot_pts[t]
            h_centroid = h_pts.mean(axis=0)
            r_centroid = r_pts.mean(axis=0)
            R, s = compute_kabsch_umeyama_transform(
                h_pts - h_centroid, r_pts - r_centroid
            )
            for i, lm in enumerate(shared_landmarks):
                aligned_human_positions[lm][t] = s * (R @ human_pts[t, i]) + (
                    r_centroid - s * (R @ h_centroid)
                )

        def keyvector_for(diff_cfg, embodiment):
            src_lm = HandLandmark(diff_cfg["src"].lower())
            tgt_lm = HandLandmark(diff_cfg["tgt"].lower())
            if embodiment == "human":
                return (
                    aligned_human_positions[src_lm] - aligned_human_positions[tgt_lm]
                )  # (T, 3) — already aligned into robot frame
            return _get_pos(robot_transforms, src_lm) - _get_pos(
                robot_transforms, tgt_lm
            )

        result: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for diff_cfg in self.vector_diff_config:
            human_kv = keyvector_for(diff_cfg, "human")  # (T, 3) meters
            robot_kv = keyvector_for(diff_cfg, "robot")  # (T, 3) meters

            human_len_m = np.linalg.norm(human_kv, axis=-1)  # (T,)
            robot_len_m = np.linalg.norm(robot_kv, axis=-1)  # (T,)
            denom = human_len_m * robot_len_m + 1e-8

            cos_sim = np.clip(
                np.einsum("td,td->t", human_kv, robot_kv) / denom, -1.0, 1.0
            )
            angle_err_deg = np.degrees(np.arccos(cos_sim))
            length_err_mm = (robot_len_m - human_len_m) * 1000.0
            scale_ratio = robot_len_m / (human_len_m + 1e-8)

            result[diff_cfg["name"]] = {
                "cosine_similarity": _summarize(cos_sim),
                "angle_error_deg": _summarize(angle_err_deg),
                "length_error_mm": _summarize(length_err_mm),
                "scale_ratio": _summarize(scale_ratio),
            }
        return result


def _summarize(arr: np.ndarray) -> Dict[str, Any]:
    """Wrap a (T,) time series with smoothed series + standard 12-stat block.

    Same shape as Motion Preservation's `pos_alignment` and Flatness's
    per-embodiment dicts, so dashboard helpers can be shared.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size:
        smoothed = gaussian_filter1d(arr, sigma=_SMOOTH_SIGMA_FRAMES, mode="nearest")
    else:
        smoothed = arr
    return {
        "raw": arr,
        "smoothed": smoothed,
        **summarize_array(arr),
    }
