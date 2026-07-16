# Measure self-collision severity in retargeted robot hand poses.
#
# For each frame, runs mj_forward and inspects MuJoCo contacts to find
# penetrating geom pairs. Reports:
#   - max_penetration_depth_mm
#   - avg_penetration_depth_mm
#   - num_unique_colliding_pairs
#   - collision_rate_pct  (fraction of frames with at least one collision)

import os

import hydra
import mujoco
import numpy as np
from omegaconf import DictConfig

from dexworld.hand_models import HumanHandModel, RobotHandModel
from dexworld.retargeting.online import BaseOnlineRetargeter
from dexworld.utils import RetargetCache

from ._stats import summarize_array
from .base_metric import BaseMetric


class CollisionMetric(BaseMetric):
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
        self.penetration_tolerance_mm = config.get("penetration_tolerance_mm", 0.0)
        self.retargeter = retargeter
        self.retarget_cache = retarget_cache
        self.human_hand_model = human_hand_model
        self.robot_hand_model = robot_hand_model

        self.data_source = hydra.utils.instantiate(data_source_cfg)

        # Load MuJoCo model for contact detection
        orig_cwd = os.getcwd()
        os.chdir(robot_hand_model.robot_hand_base_path)
        try:
            self.mj_model = mujoco.MjModel.from_xml_path(
                str(robot_hand_model.hand_model_path)
            )
        finally:
            os.chdir(orig_cwd)
        self.mj_data = mujoco.MjData(self.mj_model)

    def _geom_label(self, geom_id: int) -> str:
        gname = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body_id = self.mj_model.geom_bodyid[geom_id]
        bname = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if gname:
            return f"{bname}/{gname}"
        return f"{bname}/geom_{geom_id}"

    def compute(self):
        episode_metrics = {}

        for episode_data in self.data_source.get_episode_iter():
            human_joints_3d = episode_data["joints"]
            cache_key = (
                str(self.data_source.data_path),
                str(episode_data["episode_id"]),
            )
            self.retargeter.reset()
            robot_q = self.retarget_cache.get(cache_key, human_joints_3d)
            robot_q = np.asarray(robot_q, dtype=np.float32)

            num_frames = robot_q.shape[0]
            frames_with_collision = 0
            all_penetration_depths = []
            collision_pairs = {}
            per_frame_max_depth = np.zeros(num_frames, dtype=np.float32)

            for t in range(num_frames):
                ctrl = robot_q[t]
                qpos = ctrl @ self.robot_hand_model.joint_map.T
                self.mj_data.qpos[:] = qpos
                mujoco.mj_forward(self.mj_model, self.mj_data)

                frame_has_collision = False
                frame_max = 0.0

                for c in range(self.mj_data.ncon):
                    contact = self.mj_data.contact[c]
                    if contact.dist < 0:
                        depth_mm = -contact.dist * 1000
                        if depth_mm <= self.penetration_tolerance_mm:
                            continue
                        frame_has_collision = True
                        frame_max = max(frame_max, depth_mm)
                        all_penetration_depths.append(depth_mm)

                        g1 = self._geom_label(contact.geom1)
                        g2 = self._geom_label(contact.geom2)
                        pair = (min(g1, g2), max(g1, g2))
                        if pair not in collision_pairs:
                            collision_pairs[pair] = {
                                "count": 0,
                                "max_depth_mm": 0.0,
                                "total_depth_mm": 0.0,
                            }
                        collision_pairs[pair]["count"] += 1
                        collision_pairs[pair]["max_depth_mm"] = max(
                            collision_pairs[pair]["max_depth_mm"], depth_mm
                        )
                        collision_pairs[pair]["total_depth_mm"] += depth_mm

                if frame_has_collision:
                    frames_with_collision += 1
                per_frame_max_depth[t] = frame_max

            # Compute average depth per pair
            for info in collision_pairs.values():
                info["avg_depth_mm"] = info["total_depth_mm"] / info["count"]

            depths = np.array(all_penetration_depths, dtype=np.float32)
            episode_metrics[episode_data["episode_id"]] = {
                "penetration_tolerance_mm": self.penetration_tolerance_mm,
                "max_penetration_depth_mm": float(depths.max())
                if len(depths) > 0
                else 0.0,
                "avg_penetration_depth_mm": float(depths.mean())
                if len(depths) > 0
                else 0.0,
                "num_unique_colliding_pairs": len(collision_pairs),
                "collision_rate_pct": (frames_with_collision / num_frames * 100)
                if num_frames > 0
                else 0.0,
                "num_frames": num_frames,
                "frames_with_collision": frames_with_collision,
                "per_frame_max_depth": per_frame_max_depth,
                "collision_pairs": collision_pairs,
                # Standard 12-stat summaries for downstream aggregation.
                # `per_frame_max_depth_stats` covers every frame (zeros for
                # collision-free frames); `penetration_depth_stats` covers
                # only contacts that breached the tolerance.
                "per_frame_max_depth_stats": summarize_array(per_frame_max_depth),
                "penetration_depth_stats": summarize_array(depths),
            }

        return episode_metrics
