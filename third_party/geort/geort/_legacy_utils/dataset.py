"""Helper functions related to datasets.

Author(s):
    - Robert Jomar Malate (robert.malate@mimicrobotics.com)
"""

# Standard
# Third-party
import numpy as np

# Custom
from geort._legacy_utils.common import (
    HandTracker,
    HandChirality,
)


def extract_positions_and_orientations_from_dataset(
    link_pose_dataset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, None]:
    """Separates link pose dataset and returns the link position and orientation

    Args:
        link_pose_dataset: np.ndarray of shape (T, 25, 7) where T is number of timesteps,
            25 is number of links, and 7 represents (x, y, z, qx, qy, qz, qw)

    Returns:
        - Tuple of (positions <T, 25, 3>, orientations <T, 25, 4>)
    """
    valid_dims = (3, 7)
    if link_pose_dataset.ndim != 3:
        error_msg = (
            "Input link_pose_dataset must be a 3D numpy array of shape (T, 25, 7)."
        )
        raise ValueError(error_msg)
    if link_pose_dataset.shape[-1] not in valid_dims:
        error_msg = "Input link_pose_dataset must have last dimension of size 3 (x, y, z) or 7 (x, y, z, qx, qy, qz, qw)."
        raise ValueError(error_msg)

    if link_pose_dataset.shape[-1] == 3:
        return link_pose_dataset, None
    else:
        return link_pose_dataset[:, :, :3], link_pose_dataset[:, :, 3:7]


def extract_hand_tracker_from_filename(
    dataset_filename: str,
) -> str | None:
    """Extracts the hand tracker type from the dataset filename.

    Args:
        dataset_filename: Name of the dataset file.
    Returns:
        Hand tracker type as a string or None if not found.
    """
    for hand_tracker in HandTracker:
        if hand_tracker.value in dataset_filename.lower():
            return hand_tracker.value

    return None


def extract_hand_chirality_from_filename(dataset_filename: str) -> str | None:
    """Extracts the hand chirality from the dataset filename.

    Args:
        dataset_filename: Name of the dataset file.
    Returns:
        Hand chirality as a string or None if not found.
    """
    for chirality in HandChirality:
        if str(chirality) in dataset_filename.lower():
            return str(chirality)

    return None


def extract_subject_id_from_filename(dataset_filename: str) -> str | None:
    """Extracts the subject ID from the dataset filename.

    Args:
        dataset_filename: Name of the dataset file.
    Returns:
        Subject ID as a string or None if not found.
    """
    parts = dataset_filename.split("_")
    for part in parts:
        if part.startswith("subject-"):
            return part.replace("subject-", "")

    return None


def extract_run_id_from_filename(dataset_filename: str) -> str | None:
    """Extracts the run ID from the dataset filename.

    Args:
        dataset_filename: Name of the dataset file.
    Returns:
        Run ID as a string or None if not found.
    """
    parts = dataset_filename.split("_")
    for part in parts:
        if part.startswith("run-"):
            return part.replace("run-", "").replace(".npy", "")

    return None
