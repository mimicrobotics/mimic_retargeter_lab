from pathlib import Path

import numpy as np
import zarr

from mimic_retargeter_lab.hand_models import ManoKeypointHandModel
from mimic_retargeter_lab.types import HandDataset
from mimic_retargeter_lab.utils.logger_utils import get_logger

# from ..base_hand_data_source import BaseHandDataSource
from ..base_hand_data_reader import BaseHandDataReader


class HandDatasetReader(BaseHandDataReader):
    def __init__(
        self,
        data_path: Path | str,
        dataset: HandDataset | str,
        num_episodes: int | None = None,
        episode_id: str | None = None,
        seed: int = 42,
    ):
        super().__init__()
        self._logger = get_logger(__name__)
        if isinstance(dataset, str):
            dataset = HandDataset(dataset)
        data_path = Path(data_path) if not isinstance(data_path, Path) else data_path
        self._logger.info(f"Loading dataset from {data_path.resolve()}")

        self.data_path = data_path.resolve() / f"{dataset.value}.zarr"
        self._logger.info(f"Data path: {self.data_path}")

        self.data = zarr.open(str(self.data_path), mode="r")
        episode_keys = list(self.data.keys())
        rng = np.random.default_rng(seed)

        if num_episodes is None:
            # Choose all
            self.chosen_episodes = episode_keys
        else:
            self.chosen_episodes = rng.choice(episode_keys, num_episodes, replace=False)

        # TODO: move to config
        self.hand_model = ManoKeypointHandModel()

        # override chosen_episodes if episode_id is provided
        if episode_id is not None:
            self.chosen_episodes = [episode_id]

    def get_episode_iter(self):
        for episode_id in self.chosen_episodes:
            local_repr = self.data[episode_id]["local_representation"][:]
            joints = self.hand_model.local_repr_to_joints(local_repr)
            keyvectors = self.hand_model.compute_keyvectors(joints)
            joint_angles = self.hand_model.to_joint_angles(joints)

            ep = self.data[episode_id]
            mano_pose = ep["pose"][:] if "pose" in ep else None
            mano_shape = ep["shape"][:] if "shape" in ep else None

            yield {
                "episode_id": episode_id,
                "keyvectors": keyvectors,
                "joints": joints,
                "joint_angles": joint_angles,
                "mano_pose": mano_pose,
                "mano_shape": mano_shape,
            }

    def get_iter(self):
        for episode_id in self.chosen_episodes:
            local_repr = self.data[episode_id]["local_representation"][:]
            joints = self.hand_model.local_repr_to_joints(local_repr)
            frame_dict, links = self.hand_model.to_kinematic_tree(
                joints, return_frame_dict=True
            )
            keyvectors = self.hand_model.compute_keyvectors(joints)
            joint_angles = self.hand_model.to_joint_angles(joints)

            for idx in range(joints.shape[0]):
                transforms_t = np.array([v[idx] for v in frame_dict.values()])
                keyvectors_t = {k: v[idx, :] for k, v in keyvectors.items()}
                # joint_angles is now np.ndarray, index directly
                joint_angles_t = {k: v[idx] for k, v in joint_angles.items()}

                yield {
                    "transforms": transforms_t,
                    "keyvectors": keyvectors_t,
                    "links": [],
                    "joints": joints[idx : idx + 1],
                    "joint_angles": joint_angles_t,
                }
