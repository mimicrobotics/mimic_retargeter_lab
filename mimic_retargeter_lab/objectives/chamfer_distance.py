# From https://github.com/facebookresearch/GeoRT/blob/main/geort/loss.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def chamfer_distance(input_points, target_points):
    """
    Args:
    - input_points: Point cloud array of shape [N, 3].
    - target_points: Point cloud array of shape [M, 3].

    Returns:
    - chamfer_dist: Scalar chamfer distance.
    """
    input_points = np.asarray(input_points, dtype=np.float32)[np.newaxis]  # [1, N, 3]
    target_points = np.asarray(target_points, dtype=np.float32)[np.newaxis]  # [1, M, 3]

    input_points.shape[1]
    target_points.shape[1]

    input_points = input_points.copy()
    target_points = target_points.copy()

    input_mean = np.mean(input_points, axis=1, keepdims=True)
    target_mean = np.mean(target_points, axis=1, keepdims=True)
    input_points -= input_mean
    target_points -= target_mean

    # Broadcasting: [1, N, 1, 3] - [1, 1, M, 3] => [1, N, M, 3]
    dist_matrix = np.sum(
        (input_points[:, :, np.newaxis, :] - target_points[:, np.newaxis, :, :]) ** 2,
        axis=-1,
    )  # [1, N, M]

    min_dist_a = np.min(dist_matrix, axis=2)  # [1, N]
    min_dist_b = np.min(dist_matrix, axis=1)  # [1, M]

    chamfer_dist = np.mean(min_dist_a, axis=1) + np.mean(min_dist_b, axis=1)

    return float(chamfer_dist.mean())
