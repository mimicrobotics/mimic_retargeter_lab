"""Monte-Carlo sphere-union workspace utilization metric.

For each fingertip, the metric estimates what fraction of the robot's full
reachable Cartesian workspace was actually exercised by the retargeted
trajectory:

    1. Obtain ``P_robot`` (shape: ``num_samples x 3``) — a dense point cloud of
       the robot's reachable workspace for that fingertip. Loaded from
       ``assets/workspace_cache/<hand>.npz`` if present (precomputed by
       ``scripts/precompute_workspace.py``); otherwise sampled live by drawing
       ``num_samples`` random joint configurations and FK-ing them.
    2. Build a ``scipy.spatial.cKDTree`` from the retargeted trajectory
       ``P_traj`` (shape: ``T x 3``).
    3. For each ``P_robot`` sample, find its nearest trajectory point. If the
       distance is ≤ ``radius``, the sample is a *hit* — i.e. covered by the
       sphere union of radius ``radius`` placed around every trajectory point.
    4. ``utilization = hits / num_samples``.

``P_robot`` is computed once per fingertip and reused across all episodes
within a run.
"""

from pathlib import Path
from typing import Any, Dict, Iterable

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.spatial import cKDTree

from dexworld.hand_models import HumanHandModel, RobotHandModel
from dexworld.metrics import BaseMetric
from dexworld.metrics._stats import summarize_array
from dexworld.retargeting.online import BaseOnlineRetargeter
from dexworld.types import HandLandmark
from dexworld.utils import RetargetCache, get_logger


WORKSPACE_CACHE_DIR = Path(__file__).parent.parent.parent / "assets" / "workspace_cache"


class WorkspaceMetric(BaseMetric):
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
        self.task_space_mapping = config.task_space_mapping
        self.num_samples = int(config.get("num_samples", 100_000))
        self.radius = float(config.get("radius", 0.005))
        self.seed = int(config.get("seed", 42))
        self.fk_chunk_size = int(config.get("fk_chunk_size", 10_000))

        self.human_hand_model = human_hand_model
        self.robot_hand_model = robot_hand_model
        self.retargeter = retargeter
        self.retarget_cache = retarget_cache
        self.data_source = hydra.utils.instantiate(data_source_cfg)

        self._logger = get_logger(__name__)

        # Per-fingertip Monte-Carlo robot workspace clouds; lazy-computed,
        # shared across episodes within a single metric run.
        self._p_robot_cache: Dict[HandLandmark, np.ndarray] = {}

    # ── Monte Carlo P_robot ─────────────────────────────────────────────

    def _ensure_p_robot(self, landmarks: Iterable[HandLandmark]) -> None:
        """Populate ``self._p_robot_cache`` for every requested landmark.

        Tries the precomputed cache first
        (``assets/workspace_cache/<hand>.npz`` — produced by
        ``scripts/precompute_workspace.py``); falls back to live Monte Carlo
        sampling for any landmarks that aren't covered by the cache.
        """
        needed = [lm for lm in landmarks if lm not in self._p_robot_cache]
        if not needed:
            return

        cached_landmarks = self._load_p_robot_from_cache(needed)
        still_needed = [lm for lm in needed if lm not in cached_landmarks]
        if not still_needed:
            return

        self._sample_p_robot_live(still_needed)

    def _load_p_robot_from_cache(
        self, landmarks: Iterable[HandLandmark]
    ) -> set[HandLandmark]:
        """Best-effort load of P_robot from
        ``assets/workspace_cache/<hand>.npz``.

        Returns the set of landmarks that were successfully resolved. Any
        landmark not in the cache is left for the live-sampling fallback.
        """
        hand_name = self.robot_hand_model.robot_hand_base_path.name
        cache_path = WORKSPACE_CACHE_DIR / f"{hand_name}.npz"
        if not cache_path.exists():
            self._logger.info(
                f"WorkspaceMetric: no precomputed P_robot at {cache_path} — "
                f"will sample live (run scripts/precompute_workspace.py "
                f"--hands {hand_name} to cache it)."
            )
            return set()

        archive = np.load(cache_path)
        resolved: set[HandLandmark] = set()
        first_count: int | None = None
        for lm in landmarks:
            try:
                body_name = self.robot_hand_model._landmark_config[lm].name
            except (AttributeError, KeyError):
                continue
            if body_name not in archive.files:
                continue
            arr = np.asarray(archive[body_name], dtype=np.float32)
            self._p_robot_cache[lm] = arr
            resolved.add(lm)
            if first_count is None:
                first_count = arr.shape[0]

        if not resolved:
            self._logger.warning(
                f"WorkspaceMetric: cache {cache_path} has no entries matching "
                f"the configured fingertips — falling back to live sampling."
            )
            return set()

        # Honor the cache's actual sample count when reporting hits/N so the
        # numbers in the dashboard reflect what was actually queried.
        if first_count is not None and first_count != self.num_samples:
            self._logger.info(
                f"WorkspaceMetric: cache contains {first_count:,} samples; "
                f"config requested {self.num_samples:,} — using cached count."
            )
            self.num_samples = first_count

        self._logger.info(
            f"WorkspaceMetric: loaded P_robot for {len(resolved)} fingertip(s) "
            f"from {cache_path} ({first_count:,} samples each)."
        )
        return resolved

    def _sample_p_robot_live(self, landmarks: Iterable[HandLandmark]) -> None:
        """Fallback live Monte Carlo sampling when the precomputed cache is
        missing or doesn't cover all needed landmarks."""
        needed = list(landmarks)
        joint_names = self.robot_hand_model.get_actuated_joint_names()
        limits = self.robot_hand_model.get_actuated_joint_limits()
        lows = np.array([limits[n][0] for n in joint_names], dtype=np.float32)
        highs = np.array([limits[n][1] for n in joint_names], dtype=np.float32)

        rng = np.random.default_rng(self.seed)
        qpos = rng.uniform(
            lows, highs, size=(self.num_samples, len(joint_names))
        ).astype(np.float32)

        self._logger.info(
            f"WorkspaceMetric: Monte-Carlo sampling {self.num_samples:,} robot "
            f"poses live for {len(needed)} fingertip(s)."
        )

        positions: Dict[HandLandmark, np.ndarray] = {
            lm: np.empty((self.num_samples, 3), dtype=np.float32) for lm in needed
        }
        for start in range(0, self.num_samples, self.fk_chunk_size):
            end = min(start + self.fk_chunk_size, self.num_samples)
            transforms = self.robot_hand_model.get_landmark_transforms(
                joint_angles=qpos[start:end], joint_space="ctrl"
            )
            for lm in needed:
                T = transforms[lm]
                positions[lm][start:end] = (
                    T[:, :3, 3] if T.ndim == 3 else T[:3, 3][None, :]
                )

        for lm, arr in positions.items():
            self._p_robot_cache[lm] = arr

    # ── Per-(episode, fingertip) utilization ───────────────────────────

    def compute(self):
        episode_metrics: Dict[str, Any] = {}

        # Up-front: figure out which landmarks we need P_robot for, and sample
        # them all in one pass so FK work is shared.
        landmarks_needed = [
            HandLandmark(fmap["landmark"].lower()) for fmap in self.task_space_mapping
        ]
        self._ensure_p_robot(landmarks_needed)

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

            workspace_pts: Dict[str, Dict[str, np.ndarray]] = {}
            utilization: Dict[str, Dict[str, Any]] = {}
            for fmap in self.task_space_mapping:
                lm = HandLandmark(fmap["landmark"].lower())
                human_T = human_landmarks[lm]
                robot_T = robot_landmarks[lm]
                p_traj = (
                    robot_T[:, :3, 3] if robot_T.ndim == 3 else robot_T[:3, 3][None, :]
                )
                workspace_pts[lm.value] = {
                    "human": (
                        human_T[:, :3, 3]
                        if human_T.ndim == 3
                        else human_T[:3, 3][None, :]
                    ),
                    "robot": p_traj,
                }
                utilization[lm.value] = _compute_sphere_union_utilization(
                    p_traj=p_traj,
                    p_robot=self._p_robot_cache[lm],
                    radius=self.radius,
                )

            episode_metrics[episode_data["episode_id"]] = {
                "workspace_pts": workspace_pts,
                "utilization": utilization,
            }

        return episode_metrics


def _compute_sphere_union_utilization(
    p_traj: np.ndarray,
    p_robot: np.ndarray,
    radius: float,
) -> Dict[str, Any]:
    """Monte-Carlo sphere-union coverage of a Cartesian workspace.

    Builds a cKDTree on ``p_traj`` (the retargeted trajectory) and queries each
    ``p_robot`` sample. A sample is a "hit" iff the nearest trajectory point is
    within ``radius`` — equivalently, the sample lies inside the union of
    radius-``radius`` spheres centered at every trajectory point.

    Returns
    -------
    dict
        ``utilization`` (float in [0, 1]), ``hits`` (int), ``num_samples`` (int),
        ``radius`` (float), ``distance_stats`` (12-stat block over the
        per-sample nearest-trajectory-point distances).
    """
    num_samples = int(p_robot.shape[0])
    if p_traj.size == 0 or num_samples == 0:
        return {
            "utilization": 0.0,
            "hits": 0,
            "num_samples": num_samples,
            "radius": radius,
            "distance_stats": summarize_array(np.array([])),
        }

    tree = cKDTree(p_traj)
    # k=1 NN distance from each robot sample to the trajectory; sample is a hit
    # iff that distance ≤ radius.
    distances, _ = tree.query(p_robot, k=1, workers=-1)
    hits = int(np.sum(distances <= radius))
    return {
        "utilization": hits / num_samples,
        "hits": hits,
        "num_samples": num_samples,
        "radius": radius,
        "distance_stats": summarize_array(distances),
    }
