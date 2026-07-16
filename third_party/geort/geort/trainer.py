# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# import sapien.core as sapien
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from geort.utils.path import get_human_data
from geort.utils.config_utils import get_config, save_json
from geort.model import FKModel, IKModel

# from geort.env.hand import HandKinematicModel
from geort.loss import chamfer_distance
from geort.formatter import HandFormatter
from geort.dataset import RobotKinematicsDataset, MultiPointDataset
from tqdm import tqdm
import os
from pathlib import Path
import math
import json
import argparse
import time

# Custom
from geort._legacy_utils.helpers import get_current_datetime
from geort.env.hand_min import HandKinematicModel
from geort._legacy_utils.dataset import (
    extract_positions_and_orientations_from_dataset,
    extract_hand_tracker_from_filename,
)


def merge_dict_list(dl):
    keys = dl[0].keys()

    result = {k: [] for k in keys}
    for data in dl:
        for k in keys:
            result[k].append(data[k])

    result = {k: np.array(v) for k, v in result.items()}
    return result


def format_loss(value):
    return f"{value:.4e}" if math.fabs(value) < 1e-3 else f"{value:.4f}"


def get_float_list_from_np(np_vector):
    float_list = np_vector.tolist()
    float_list = [float(x) for x in float_list]
    return float_list


def generate_current_datetime() -> str:
    """
    Utility Function. Generate a current timestring in the format 'YYYY-MM-DD_HH-MM-SS_<timezone>'.
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S_%Z", time.localtime())


class GeoRTTrainer:
    def __init__(self, config):
        self.config = config
        self.hand = HandKinematicModel.build_from_config(self.config)

    def get_robot_pointcloud(self, keypoint_names):
        """
        Utility getter function. Return the robot fingertip point cloud.
        """
        kinematics_dataset = self.get_robot_kinematics_dataset()
        return kinematics_dataset.export_robot_pointcloud(keypoint_names)

    def get_robot_kinematics_dataset(self):
        """
        Utility getter function. Return the robot kinematics dataset
        """
        dataset_path = self.get_robot_kinematics_dataset_path(postfix=True)
        if not os.path.exists(dataset_path):
            dataset = self.generate_robot_kinematics_dataset(n_total=100000, save=True)
            dataset_path = self.get_robot_kinematics_dataset_path(postfix=True)

        keypoint_names = self.get_keypoint_info()["link"]

        kinematics_dataset = RobotKinematicsDataset(
            dataset_path, keypoint_names=keypoint_names
        )
        return kinematics_dataset

    def get_robot_kinematics_dataset_path(self, postfix=False):
        """
        Utility getter function. Return the path to the robot kinematics dataset.
        """
        data_name = self.config["name"]

        out = f"data/{data_name}"
        if postfix:
            out += ".npz"
        return out

    def get_keypoint_info(self):
        keypoint_links = []
        keypoint_offsets = []
        keypoint_joints = []
        keypoint_human_ids = []

        joint_order = self.config["joint_order"]

        for info in self.config["fingertip_link"]:
            keypoint_links.append(info["link"])
            keypoint_offsets.append(info["center_offset"])
            keypoint_human_ids.append(info["human_hand_id"])

            keypoint_joint = []
            for joint in info["joint"]:
                keypoint_joint.append(joint_order.index(joint))

            keypoint_joints.append(keypoint_joint)

        out = {
            "link": keypoint_links,
            "offset": keypoint_offsets,
            "joint": keypoint_joints,
            "human_id": keypoint_human_ids,
        }

        return out

    def generate_robot_kinematics_dataset(self, n_total=100000, save=True):
        """
        This function will generate a (joint position, keypoint position) dataset.
        - The joint order is specified by "joint_order" in configuration.
        - The keypoint order is specified by "fingertip_link" field in configuration.
        """
        info = self.get_keypoint_info()

        self.hand.initialize_keypoint(
            keypoint_link_names=info["link"], keypoint_offsets=info["offset"]
        )

        joint_range_low, joint_range_high = (
            self.hand.get_joint_limit()
        )  # joint order is based on user config specification.
        joint_range_low = np.array(joint_range_low)
        joint_range_high = np.array(joint_range_high)

        all_data_qpos = []
        all_data_keypoint = []

        for _ in tqdm(range(n_total)):
            qpos = (
                np.random.uniform(0, 1, len(joint_range_low))
                * (joint_range_high - joint_range_low)
                + joint_range_low
            )
            keypoint = self.hand.keypoint_from_qpos(qpos)
            all_data_qpos.append(qpos)
            all_data_keypoint.append(keypoint)

        all_data_keypoint = merge_dict_list(all_data_keypoint)

        dataset = {"qpos": all_data_qpos, "keypoint": all_data_keypoint}

        if save:
            # save data to disk for future use.
            os.makedirs("data", exist_ok=True)
            np.savez(self.get_robot_kinematics_dataset_path(), **dataset)

        return dataset

    def get_fk_checkpoint_path(self):
        name = self.config["name"]
        os.makedirs("checkpoint", exist_ok=True)
        return f"checkpoint/fk_model_{name}.pth"

    def get_robot_neural_fk_model(self, force_train=False):
        """
        This function will return a forward kinematics model.
        If the fk model does not exist, this function will train one first.
        """

        # Normalizer.
        joint_lower_limit, joint_upper_limit = self.hand.get_joint_limit()
        qpos_normalizer = HandFormatter(joint_lower_limit, joint_upper_limit)

        # Model.
        print(self.get_keypoint_info()["joint"])
        fk_model = FKModel(keypoint_joints=self.get_keypoint_info()["joint"]).cuda()

        # If the model exists, load it.
        fk_checkpoint_path = self.get_fk_checkpoint_path()
        if os.path.exists(fk_checkpoint_path) and not force_train:
            fk_model.load_state_dict(torch.load(fk_checkpoint_path))

        else:
            # If the model does not exist, train it.
            print("Train Neural Forward Kinematics (FK) from Scratch")

            fk_dataset = self.get_robot_kinematics_dataset()
            fk_dataloader = DataLoader(fk_dataset, batch_size=256, shuffle=True)
            fk_optim = optim.Adam(fk_model.parameters(), lr=5e-4)

            criterion_fk = nn.MSELoss()
            for epoch in range(200):
                all_fk_error = 0
                for batch_idx, batch in enumerate(fk_dataloader):
                    keypoint = batch["keypoint"].cuda().float()
                    qpos = batch["qpos"].cuda().float()
                    # breakpoint()
                    qpos = qpos_normalizer.normalize_torch(qpos)
                    predicted_keypoint = fk_model(qpos)
                    fk_optim.zero_grad()
                    loss = criterion_fk(predicted_keypoint, keypoint)
                    loss.backward()
                    fk_optim.step()

                    all_fk_error += loss.item()

                avg_fk_error = all_fk_error / (batch_idx + 1)
                print(
                    f"Neural FK Training Epoch: {epoch}; Training Loss: {avg_fk_error}"
                )

            torch.save(fk_model.state_dict(), fk_checkpoint_path)

        fk_model.eval()
        return fk_model

    def train(
        self, dataset_positions_data: np.ndarray, dataset_metadata=None, **kwargs
    ):
        """
        This is the main trainer.

        Args:
            dataset_positions_data: dataset of human fingertip positions (N_samples x N_fingertips x 3).
            dataset_metadata: (Optional) Metadata dictionary for the dataset.
            kwargs: Additional keyword arguments for training configuration.
        Returns:
            None
        """

        fk_model = self.get_robot_neural_fk_model()
        ik_model = IKModel(keypoint_joints=self.get_keypoint_info()["joint"]).cuda()
        os.makedirs("./checkpoint", exist_ok=True)
        learning_rate = 1e-4
        ik_optim = optim.AdamW(ik_model.parameters(), lr=learning_rate)

        # Workspace.
        exp_tag = kwargs.get("tag", "")
        exp_id = kwargs.get("exp_id", "000")

        # Create tag from dataset_metadata if not provided.
        if not exp_tag and dataset_metadata:
            try:
                session_config = dataset_metadata.get("session_config", {})

                exp_tag = (
                    f"geort_"
                    f"{session_config.get('hand_chirality', 'unknown')}_"
                    f"{session_config.get('hand_tracker', 'unknown')}_"
                    f"subject-{session_config.get('subject_id', 'unknown')}_"
                    f"exp-{exp_id}"
                )
                print(f"Auto-generated experiment tag from dataset metadata: {exp_tag}")
            except Exception as e:
                print(
                    f"Failed to auto-generate experiment tag from dataset metadata: {e}"
                )

        n_epoch = kwargs.get("epoch", 200)
        hand_model_name = self.config["name"]

        w_chamfer = kwargs.get("w_chamfer", 80.0)
        w_curvature = kwargs.get("w_curvature", 0.1)
        w_collision = kwargs.get("w_collision", 0.0)
        w_pinch = kwargs.get("w_pinch", 1.0)

        datetime_trained = get_current_datetime()

        save_dir = f"./checkpoint/{hand_model_name}_{datetime_trained}"
        if exp_tag != "":
            save_dir += f"_{exp_tag}"
        last_save_dir = f"./checkpoint/{hand_model_name}_last"

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(last_save_dir, exist_ok=True)

        # Save the config including robot joint info to the checkpoint directory.
        joint_lower_limit, joint_upper_limit = self.hand.get_joint_limit()

        export_config = self.config.copy()
        export_config["joint"] = {
            "lower": get_float_list_from_np(joint_lower_limit),
            "upper": get_float_list_from_np(joint_upper_limit),
        }
        export_config["datetime_trained"] = datetime_trained
        export_config["training_hyperparameters"] = {
            "tag": exp_tag,
            "experiment_id": exp_id,
            "n_epoch": n_epoch,
            "learning_rate": learning_rate,
            "w_chamfer": w_chamfer,
            "w_curvature": w_curvature,
            "w_collision": w_collision,
            "w_pinch": w_pinch,
        }

        if dataset_metadata:
            export_config["dataset_info"] = dataset_metadata

        save_json(export_config, Path(save_dir) / "config.json")
        save_json(export_config, Path(last_save_dir) / "config.json")

        # Dataset.
        robot_keypoint_names = self.get_keypoint_info()["link"]
        n_keypoints = len(robot_keypoint_names)

        robot_points = self.get_robot_pointcloud(robot_keypoint_names)

        human_finger_idxes = self.get_keypoint_info()["human_id"]
        for robot_keypoint_name, human_id in zip(
            robot_keypoint_names, human_finger_idxes
        ):
            print(f"Robot Keypoint {robot_keypoint_name}: Human Id: {human_id}")

        dataset_positions_data = np.array(
            [dataset_positions_data[:, idx, :3] for idx in human_finger_idxes]
        )  # [N_finger, N, 3]

        point_dataset_human = MultiPointDataset.from_points(
            dataset_positions_data, n=20000
        )
        point_dataloader = DataLoader(
            point_dataset_human, batch_size=2048, shuffle=True
        )

        # Training / Optimization
        for epoch in range(n_epoch):
            for batch_idx, batch in enumerate(point_dataloader):
                direction_loss = 0

                point = batch.cuda()  # [B, N, 3]
                joint = ik_model(point)  # [B, DOF]
                embedded_point = fk_model(joint)  # [B, N, 3]

                # [Pinch Loss]
                # We find it sufficient to only consider thumb-X pinch. Note that idx 0 is thumb.
                n_finger = point.size(1)
                pinch_loss = 0

                for i in range(n_finger):
                    for j in range(i + 1, n_finger):
                        distance = point[:, i, ...] - point[:, j, ...]
                        mask = (torch.norm(distance, dim=-1) < 0.015).float()
                        e_distance = (
                            (embedded_point[:, i, ...] - embedded_point[:, j, ...]) ** 2
                        ).sum(dim=-1)
                        pinch_loss += (
                            (mask * e_distance).mean()
                            / (mask.sum() + 1e-7)
                            * point.size(0)
                        )

                # [Curvature loss] -- Ensuring flatness.
                direction = F.normalize(torch.randn_like(point), dim=-1, p=2)
                scale = 0.002
                delta1 = direction * scale
                point_delta_1p = point + delta1
                point_delta_1n = point - delta1

                embedded_point_p = fk_model(ik_model(point_delta_1p))
                embedded_point_n = fk_model(ik_model(point_delta_1n))
                curvature_loss = (
                    (embedded_point_p + embedded_point_n - 2 * embedded_point) ** 2
                ).mean()

                # [Chamfer loss]
                selected_idx = np.random.randint(0, robot_points.shape[1], 2048)
                target = (
                    torch.from_numpy(robot_points[:, selected_idx, :])
                    .permute(1, 0, 2)
                    .float()
                    .cuda()
                )

                chamfer_loss = 0
                for i in range(n_keypoints):
                    chamfer_loss += chamfer_distance(
                        embedded_point[:, i, :].unsqueeze(0),
                        target[:, i, :].unsqueeze(0),
                    )

                # [Direction Loss]
                direction = F.normalize(torch.randn_like(point), dim=-1, p=2)
                scale = (
                    0.001
                    + torch.rand(point.size(0)).cuda().unsqueeze(-1).unsqueeze(-1)
                    * 0.01
                )
                point_delta = point + direction * scale

                joint_delta = ik_model(point_delta)
                embedded_point_delta = fk_model(joint_delta)

                d1 = (point_delta - point).reshape(-1, 3)
                d2 = (embedded_point_delta - embedded_point).reshape(-1, 3)
                direction_loss = -(
                    (
                        F.normalize(d1, dim=-1, p=2, eps=1e-5)
                        * F.normalize(d2, dim=-1, p=2, eps=1e-5)
                    ).sum(-1)
                ).mean()

                # [Collision loss]
                # if classifier is not None:
                #     real_labels = torch.ones(joint.size(0), dtype=torch.long).to(joint.device)
                #     # Discriminator's output for generated data
                #     safe_logits = classifier(joint)
                #     criterion = nn.CrossEntropyLoss()
                #     # Generator loss is the cross-entropy loss between the fake outputs and the label 1 (real)
                #     collision_loss = criterion(safe_logits, real_labels)

                # collision Loss integration pending.
                collision_loss = torch.tensor([0.0]).cuda()

                loss = (
                    direction_loss
                    + chamfer_loss * w_chamfer
                    + curvature_loss * w_curvature
                    + collision_loss * w_collision
                    + pinch_loss * w_pinch
                )

                ik_optim.zero_grad()
                loss.backward()
                ik_optim.step()

                if batch_idx % 50 == 0:
                    print(
                        f"Epoch {epoch} | Losses"
                        f" - Direction: {format_loss(direction_loss.item())}"
                        f" - Chamfer: {format_loss(chamfer_loss.item())}"
                        f" - Curvature: {format_loss(curvature_loss.item())}"
                        f" - Collision: {format_loss(collision_loss.item())}"
                        f" - Pinch: {format_loss(pinch_loss.item())}"
                    )

            # Saving the checkpoint.
            torch.save(ik_model.state_dict(), Path(save_dir) / f"epoch_{epoch}.pth")
            torch.save(ik_model.state_dict(), Path(save_dir) / "last.pth")

            # This is just for dev phase convenience and can be removed.
            torch.save(
                ik_model.state_dict(), Path(last_save_dir) / f"epoch_{epoch}.pth"
            )
            torch.save(ik_model.state_dict(), Path(last_save_dir) / "last.pth")

        return


# --- HELPER FUNCTIONS ---
def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", type=str, default="allegro")
    parser.add_argument(
        "--human_data",
        type=str,
        default="human",
        help="Path to human pointcloud data (.npy file).",
    )
    parser.add_argument(
        "--exp_id", type=str, default="000", help="Experiment ID (e.g. 001, 002)"
    )
    parser.add_argument(
        "--ckpt_tag", type=str, default="", help="Manually override the entire tag"
    )

    parser.add_argument("--w_chamfer", type=float, default=80.0)
    parser.add_argument("--w_curvature", type=float, default=0.1)
    parser.add_argument("--w_collision", type=float, default=0.0)
    parser.add_argument("--w_pinch", type=float, default=1.0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()

    hand_tracker = extract_hand_tracker_from_filename(args.human_data)
    config = get_config(args.hand, hand_tracker)
    trainer = GeoRTTrainer(config)

    human_data_path = get_human_data(args.human_data)
    human_dataset = np.load(human_data_path)
    human_data_positions, human_data_orientations = (
        extract_positions_and_orientations_from_dataset(human_dataset)
    )
    print("Training with human data:", human_data_path.as_posix())

    # Retrieve dataset metadata
    metadata_filename = human_data_path.name.replace("dataset_", "metadata_", 1)
    metadata_path = human_data_path.with_name(metadata_filename).with_suffix(".json")
    dataset_metadata = None
    if metadata_path.exists():
        print(f"Found metadata file: {metadata_path.as_posix()}")
        with open(metadata_path, "r") as f:
            dataset_metadata = json.load(f)
    else:
        print("No metadata file found.")
        print(f"  Expected at: {metadata_path}")
        print("  Checkpoint will NOT be autotagged.")

    trainer.train(
        dataset_positions_data=human_data_positions,
        dataset_metadata=dataset_metadata,
        tag=args.ckpt_tag,
        exp_id=args.exp_id,
        w_chamfer=args.w_chamfer,
        w_curvature=args.w_curvature,
        w_collision=args.w_collision,
        w_pinch=args.w_pinch,
    )
