# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import open3d as o3d


def upsample_array(x, K=50000):
    n = x.shape[0]
    ind = np.random.randint(0, n - 1, K)
    return x[ind]


class MultiPointDataset:
    def __init__(self, points):
        self.points = np.array(points)  # [Num_Fingers, Num_Samples, 3]

    @staticmethod
    def from_points(points, n, resample_to=50000, resample_resolution=0.001):
        """
        This is the actual initialization function.
        """
        num_fingers = points.shape[0]
        all_points = []

        # Resampling to reduce spatial imbalance.
        for finger_id in range(num_fingers):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[finger_id])
            downpcd = pcd.voxel_down_sample(voxel_size=resample_resolution)
            resampled_points = np.asarray(downpcd.points)
            all_points.append(upsample_array(resampled_points, K=resample_to))

        return MultiPointDataset(np.array(all_points).astype(np.float32))

    def __len__(self):
        return self.points.shape[1]

    def __getitem__(self, idx):
        return self.points[:, idx]


class RobotKinematicsDataset:
    def __init__(self, qpos_keypoint_file, keypoint_names):
        np_array = np.load(qpos_keypoint_file, allow_pickle=True)
        self.qpos = np_array["qpos"]
        self.keypoints = np_array["keypoint"].item()
        self.keypoint_names = keypoint_names
        print("Keypoint Names", self.keypoint_names)
        self.n = len(self.qpos)
        return

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        keypoint_data = []
        for name in self.keypoint_names:
            keypoint_data.append(self.keypoints[name][idx][:3])

        return {
            "qpos": self.qpos[idx].astype(np.float32),
            "keypoint": np.array(keypoint_data).astype(np.float32),
        }

    def export_robot_pointcloud(self, keypoint_names):
        all_keypoint_data = []
        for keypoint_name in keypoint_names:
            all_keypoint_data.append(self.keypoints[keypoint_name])
        return np.array(all_keypoint_data)


if __name__ == "__main__":
    dataset = RobotJointKeypointDataset(
        "../data/allegro_native.npz", ["link_3.0_tip", "link_3.0_tip"]
    )
    print(dataset[0])
