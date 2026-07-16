# Measure deviation from a linear response in joint space and task space


import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

from dexworld.hand_models import HumanHandModel, RobotHandModel
from dexworld.retargeting.online import BaseOnlineRetargeter
from dexworld.types.types import HandLandmark
from dexworld.utils import RetargetCache

from .base_metric import BaseMetric


class ResponseMetric(BaseMetric):
    def __init__(
        self,
        config,
        human_hand_model: HumanHandModel,
        robot_hand_model: RobotHandModel,
        retargeter: BaseOnlineRetargeter,
        data_source_cfg: DictConfig,
        retarget_cache: RetargetCache | None = None,
        *,
        compute_derivatives: bool = True,
        downsampling_factor: int = 2,
    ):
        self.display_name = config.display_name
        self.joint_space_mapping = config.joint_space_mapping
        self.task_space_mapping = config.task_space_mapping
        self.reference_type = config.reference_type

        self.retargeter = retargeter
        self.retarget_cache = retarget_cache
        self.human_hand_model = human_hand_model
        self.robot_hand_model = robot_hand_model
        self.compute_derivatives = compute_derivatives
        self.downsampling_factor = downsampling_factor

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

            human_kintree, _ = self.human_hand_model.to_kinematic_tree(
                joints_3d=human_joints_3d, return_frame_dict=False
            )
            robot_kintree, _ = self.robot_hand_model.to_kinematic_tree(
                joint_angles=robot_joint_angles_actuated,
                joint_space="ctrl",
                return_frame_dict=False,
            )

            joint_space_metrics = self.compute_joint_space_metrics(
                human_joints_3d,
                robot_joint_angles_actuated,
                compute_derivatives=self.compute_derivatives,
                downsampling_factor=2,
            )
            task_space_metrics = self.compute_task_space_metrics(
                human_joints_3d,
                robot_joint_angles_actuated,
                compute_derivatives=self.compute_derivatives,
                downsampling_factor=2,
            )

            episode_metrics[episode_data["episode_id"]] = {
                "joint_space": joint_space_metrics,
                "task_space": task_space_metrics,
                "kinematic_trees": {"human": human_kintree, "robot": robot_kintree},
            }

        return episode_metrics

    def compute_joint_space_metrics(
        self,
        human_joints_3d,
        robot_joint_angles_actuated,
        compute_derivatives=False,
        downsampling_factor: int = 2,
    ):
        human_hand_joint_angles = self.human_hand_model.to_joint_angles(
            joints_3d=human_joints_3d
        )
        robot_joint_angles_qpos = (
            robot_joint_angles_actuated @ self.robot_hand_model.joint_map.T
        )
        joint_responses = {}
        for joint_map_cfg in self.joint_space_mapping:
            joint_responses[joint_map_cfg["name"]] = {
                "in": human_hand_joint_angles[joint_map_cfg["src_key"]][
                    ::downsampling_factor, ...
                ],
                "out": robot_joint_angles_qpos[
                    :,
                    self.robot_hand_model.get_qpos_joint_names().index(
                        joint_map_cfg["tgt_key"]
                    ),
                ][::downsampling_factor, ...],
            }

        if not compute_derivatives:
            return {
                joint_name: {"response": joint_responses[joint_name]}
                for joint_name in joint_responses
            }

        # linear response should have a constant derivative - this is what we will be visualizing against
        # TODO: try to fit a curve/line?
        derivative_estimates = {}
        for joint_name in joint_responses:
            grad = self._compute_sequence_derivative(
                joint_responses[joint_name]["out"], joint_responses[joint_name]["in"]
            )
            derivative_estimates[joint_name] = {
                "values": grad,
                "mean": np.mean(grad),
                "std": np.std(grad),
                "min": np.min(grad),
                "max": np.max(grad),
            }

        out = {
            joint_name: {
                "response": joint_responses[joint_name],
                "derivative": derivative_estimates[joint_name],
            }
            for joint_name in joint_responses
        }

        return out

    def _compute_sequence_derivative(
        self, f_x, x, method: str = "first_order", sigma=2
    ):
        """
        f_x: (T,)
        x: (T,)
        """
        match method:
            case "np.gradient":
                derivative = np.gradient(f_x, x)
            case "first_order":
                # zeros_division = np.where(np.diff(x) == 0)
                # print(f"WARNING: Division by zero at {len(zeros_division)} indices")
                derivative = np.diff(f_x) / (np.diff(x) + 1e-6)
                derivative = np.concatenate([derivative, np.zeros(1)], axis=0)

            case _:
                raise ValueError(f"Unknown method: {method}")

        return gaussian_filter1d(derivative, sigma=2, mode="nearest")

    def compute_task_space_metrics(
        self,
        human_joints_3d,
        robot_joint_angles_actuated,
        compute_derivatives=True,
        downsampling_factor=2,
    ):
        human_landmarks = self.human_hand_model.get_landmark_transforms(
            joints_3d=human_joints_3d
        )
        robot_landmarks = self.robot_hand_model.get_landmark_transforms(
            joint_angles=robot_joint_angles_actuated,
            joint_space="ctrl",
        )

        frame_responses = {}
        derivative_estimates = {}
        for fmap in self.task_space_mapping:
            lm = HandLandmark(fmap["landmark"].lower())
            # 4x4 matrices
            in_frames = human_landmarks[lm]
            out_frames = robot_landmarks[lm]
            if in_frames.ndim == 2:
                in_frames = in_frames[None, :]
            if out_frames.ndim == 2:
                out_frames = out_frames[None, :]

            human_pos = in_frames[:, :3, 3]
            robot_pos = out_frames[:, :3, 3]

            human_euler = R.from_matrix(in_frames[:, :3, :3]).as_euler(
                "xyz", degrees=True
            )
            robot_euler = R.from_matrix(out_frames[:, :3, :3]).as_euler(
                "xyz", degrees=True
            )

            frame_responses[lm.value] = {
                "pos": {
                    "in": {
                        "x": human_pos[:, 0],
                        "y": human_pos[:, 1],
                        "z": human_pos[:, 2],
                    },
                    "out": {
                        "x": robot_pos[:, 0],
                        "y": robot_pos[:, 1],
                        "z": robot_pos[:, 2],
                    },
                },
                "rot": {
                    "in": {
                        "x": human_euler[:, 0],
                        "y": human_euler[:, 1],
                        "z": human_euler[:, 2],
                    },
                    "out": {
                        "x": robot_euler[:, 0],
                        "y": robot_euler[:, 1],
                        "z": robot_euler[:, 2],
                    },
                },
            }

            if not compute_derivatives:
                continue

            derivative_estimates[lm.value] = {
                elem: {
                    char: self._compute_sequence_derivative(
                        frame_responses[lm.value][elem]["out"][char],
                        frame_responses[lm.value][elem]["in"][char],
                    )
                    for char in ["x", "y", "z"]
                }
                for elem in ["pos", "rot"]
            }

        if not compute_derivatives:
            out = {
                frame_name: {
                    "frame_responses": frame_responses[frame_name],
                }
                for frame_name in frame_responses.keys()
            }
        else:
            out = {
                frame_name: {
                    "frame_responses": frame_responses[frame_name],
                    "derivative_estimates": derivative_estimates[frame_name],
                }
                for frame_name in frame_responses.keys()
            }
        return out
