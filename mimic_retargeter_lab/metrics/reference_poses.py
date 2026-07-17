# Given a set of reference robot poses and a sequence of robot poses,
# compute the minimum pose error in joint angles and frame poses for each reference pose

import hydra
import numpy as np
import tqdm
from omegaconf import DictConfig

from mimic_retargeter_lab.hand_models import HumanHandModel, RobotHandModel
from mimic_retargeter_lab.retargeting.online import BaseOnlineRetargeter
from mimic_retargeter_lab.types import HandLandmark
from mimic_retargeter_lab.utils import RetargetCache
from mimic_retargeter_lab.utils.retarget_utils import compute_kabsch_umeyama_transform

from .base_metric import BaseMetric


class ReferencePoseMetric(BaseMetric):
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

    # ------------------------------------------------------------------
    # Vector-diff line helpers
    # ------------------------------------------------------------------

    # Hand-mesh tip vertex indices (right hand) keyed by HandLandmark.
    _HUMAN_LANDMARK_TO_VERT = {
        HandLandmark.THUMB_TIP: 745,
        HandLandmark.INDEX_TIP: 317,
        HandLandmark.MIDDLE_TIP: 444,
        HandLandmark.RING_TIP: 556,
        HandLandmark.PINKY_TIP: 673,
    }

    def _extract_vector_diff_lines(
        self,
        human_landmark_transforms,
        robot_landmark_transforms,
        human_verts=None,
    ):
        """Return per-embodiment line segments for each vector diff.

        If *human_verts* is provided, human endpoints come from the mesh
        surface so the dots sit exactly on the rendered fingertips.
        """
        lines = []
        for diff_cfg in self.vector_diff_config:
            src_lm = HandLandmark(diff_cfg["src"].lower())
            tgt_lm = HandLandmark(diff_cfg["tgt"].lower())
            entry = {"name": diff_cfg["name"]}

            # Human side: prefer mesh vertices when available
            if (
                human_verts is not None
                and src_lm in self._HUMAN_LANDMARK_TO_VERT
                and tgt_lm in self._HUMAN_LANDMARK_TO_VERT
            ):
                entry["human"] = {
                    "src": human_verts[self._HUMAN_LANDMARK_TO_VERT[src_lm]].copy(),
                    "tgt": human_verts[self._HUMAN_LANDMARK_TO_VERT[tgt_lm]].copy(),
                }
            elif human_landmark_transforms is not None:
                src_T = human_landmark_transforms.get(src_lm)
                tgt_T = human_landmark_transforms.get(tgt_lm)
                if src_T is not None and tgt_T is not None:
                    entry["human"] = {
                        "src": src_T[:3, 3].astype(np.float32).copy(),
                        "tgt": tgt_T[:3, 3].astype(np.float32).copy(),
                    }

            # Robot side: from landmark transforms
            if robot_landmark_transforms is not None:
                src_T = robot_landmark_transforms.get(src_lm)
                tgt_T = robot_landmark_transforms.get(tgt_lm)
                if src_T is not None and tgt_T is not None:
                    entry["robot"] = {
                        "src": src_T[:3, 3].astype(np.float32).copy(),
                        "tgt": tgt_T[:3, 3].astype(np.float32).copy(),
                    }

            lines.append(entry)
        return lines

    def compute(self):
        episode_metrics = {}
        episodes = list(self.data_source.get_episode_iter())
        for episode_data in tqdm.tqdm(episodes, desc=f"MANO fit ({self.display_name})"):
            human_joints_3d = episode_data["joints"]
            cache_key = (
                str(self.data_source.data_path),
                str(episode_data["episode_id"]),
            )
            robot_joint_angles_actuated = self.retarget_cache.get(
                cache_key, human_joints_3d
            )
            robot_joint_angles_actuated = np.asarray(
                robot_joint_angles_actuated, dtype=np.float32
            )

            # Human kinematic tree for skeleton visualization (full 21-joint MANO tree)
            _, human_links = self.human_hand_model.to_kinematic_tree(
                joints_3d=human_joints_3d, return_frame_dict=False
            )

            reference_pose_metrics = self._compute_reference_pose_metrics(
                human_joints_3d,
                robot_joint_angles_actuated,
            )

            robot_meshes = self.robot_hand_model.get_mesh_geoms(
                joint_angles=robot_joint_angles_actuated[0],
                joint_space="ctrl",
            )

            # Human hand mesh is not rendered: the dashboard shows the human as a
            # skeleton (``human_links``). Reinstating a mesh overlay would mean
            # fitting a parametric hand model (e.g. MANO) to the keypoints, which
            # this repo deliberately does not depend on.
            human_meshes = None

            # Extract vector-diff line endpoints for dashboard overlay.
            human_lm = reference_pose_metrics["landmark_transforms"]["human"]
            robot_lm = reference_pose_metrics["landmark_transforms"]["robot"]
            vector_diff_lines = self._extract_vector_diff_lines(human_lm, robot_lm)

            episode_metrics[episode_data["episode_id"]] = {
                "reference_pose_metrics": reference_pose_metrics,
                "human_links": human_links,
                "robot_meshes": robot_meshes,
                "human_meshes": human_meshes,
                "vector_diff_lines": vector_diff_lines,
            }

        return episode_metrics

    def _compute_reference_pose_metrics(
        self, human_joints_3d, robot_joint_angles_actuated
    ):
        human_transforms = self.human_hand_model.get_landmark_transforms(
            joints_3d=human_joints_3d
        )
        robot_transforms = self.robot_hand_model.get_landmark_transforms(
            joint_angles=robot_joint_angles_actuated,
            joint_space="ctrl",
        )

        def _get_pos(lm_transforms, lm):
            """Extract (N, 3) position from landmark transforms."""
            T = lm_transforms[lm]
            if T.ndim == 3:
                return T[:, :3, 3]
            return T[:3, 3][None, :]

        # Kabsch-Umeyama: align human landmarks into robot's MuJoCo frame
        shared_landmarks = [
            lm
            for lm in HandLandmark
            if lm in human_transforms and lm in robot_transforms
        ]

        human_pts = np.stack(
            [_get_pos(human_transforms, lm) for lm in shared_landmarks], axis=1
        )  # (N, M, 3)
        robot_pts = np.stack(
            [_get_pos(robot_transforms, lm) for lm in shared_landmarks], axis=1
        )  # (N, M, 3)

        # Compute alignment per frame and build aligned human positions
        N = human_pts.shape[0]
        aligned_human_positions = {}
        for i, lm in enumerate(shared_landmarks):
            aligned_human_positions[lm] = np.zeros((N, 3))

        for t in range(N):
            h_pts = human_pts[t]  # (M, 3)
            r_pts = robot_pts[t]  # (M, 3)

            h_centroid = h_pts.mean(axis=0)
            r_centroid = r_pts.mean(axis=0)

            R, s = compute_kabsch_umeyama_transform(
                h_pts - h_centroid, r_pts - r_centroid
            )

            # Align all shared landmarks for this frame
            for i, lm in enumerate(shared_landmarks):
                aligned_human_positions[lm][t] = s * (R @ human_pts[t, i]) + (
                    r_centroid - s * (R @ h_centroid)
                )

        def compute_frame_diff_aligned(diff_cfg, embodiment):
            src_lm = HandLandmark(diff_cfg["src"].lower())
            tgt_lm = HandLandmark(diff_cfg["tgt"].lower())
            if embodiment == "human":
                src_pos = aligned_human_positions[src_lm]
                tgt_pos = aligned_human_positions[tgt_lm]
            else:
                src_pos = _get_pos(robot_transforms, src_lm)
                tgt_pos = _get_pos(robot_transforms, tgt_lm)
            return src_pos - tgt_pos

        def compute_metrics(human_keyvector, robot_keyvector):
            human_kv = human_keyvector.squeeze()
            robot_kv = robot_keyvector.squeeze()

            # Calculate base norms (These are in METERS)
            human_kv_len_m = np.linalg.norm(human_kv)
            robot_kv_len_m = np.linalg.norm(robot_kv)

            # Distance Metrics (Convert to MILLIMETERS)
            length_error_mm = (robot_kv_len_m - human_kv_len_m) * 1000.0

            # Angle & Similarity Metrics (Reusing calculated lengths for efficiency)
            # Added 1e-8 to prevent division by zero if hands perfectly close
            cosine_similarity = np.dot(human_kv, robot_kv) / (
                human_kv_len_m * robot_kv_len_m + 1e-8
            )

            # Clip to [-1, 1] to prevent np.arccos from crashing due to float imprecision
            cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
            angular_error_deg = np.degrees(np.arccos(cosine_similarity))

            # Scale Metric (Robot / Human makes the human the baseline '1.0')
            scale_ratio = robot_kv_len_m / (human_kv_len_m + 1e-8)

            # Return clean dictionary with updated, descriptive keys
            return {
                "Length Error [mm]": length_error_mm,
                "Cosine Similarity": float(cosine_similarity),
                "Angle Error [deg]": float(angular_error_deg),
                "Scale Ratio [robot/human]": float(scale_ratio),
            }

        metrics = {
            "error_metrics": {
                diff_cfg["name"]: compute_metrics(
                    compute_frame_diff_aligned(diff_cfg, "human"),
                    compute_frame_diff_aligned(diff_cfg, "robot"),
                )
                for diff_cfg in self.vector_diff_config
            },
            "landmark_transforms": {
                "human": human_transforms,
                "robot": robot_transforms,
            },
        }

        return metrics
