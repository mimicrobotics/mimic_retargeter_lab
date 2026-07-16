"""Inspects the collected hand tracker data.

Author(s):
    - Robert Jomar Malate (robert.malate@mimicrobotics.com)
"""

# Standard
import argparse
import csv
import logging
from pathlib import Path

# Third-party
import numpy as np
from tabulate import tabulate

# Custom
from utils.common import HandTracker, FINGERTIP_INDICES
from utils.helpers import (
    initialize_logger,
    convert_m_to_cm,
    convert_m_to_mm,
    get_current_datetime,
)
from utils.dataset import (
    extract_hand_tracker_from_filename,
    extract_hand_chirality_from_filename,
    extract_subject_id_from_filename,
    extract_run_id_from_filename,
)
import utils.argparse_utils as argparse_utils


def parse_cli_args() -> argparse.ArgumentParser:
    """Initializes the argument parser for the script.

    Returns:
        An argparse.ArgumentParser instance.
    """
    parser = argparse_utils.create_argparser(
        description="Inspect recorded hand tracker data.",
    )
    argparse_utils.add_args(parser, "dataset_filename")
    parser.add_argument(
        "--output_csv",
        default=False,
        help="Boolean flag to save the statistics to a CSV file.",
        action="store_true",
    )
    return parser.parse_args()


def calculate_min_fingertip_distances(
    hand_pcloud_data: np.ndarray,
    fingertip_indices: dict[str, int],
) -> dict[str, float]:
    """Calculates the minimum distances between the thumb fingertip and other fingertips.

    Args:
        hand_pcloud_data: np.ndarray of shape (N, 3) or (N, 7) where N is number of points from mocap system.
    Returns:
        A dictionary with minimum distances between thumb and other fingertips.
    """
    fingertip_min_distances = {
        "thumb-index": np.inf,
        "thumb-middle": np.inf,
        "thumb-ring": np.inf,
        "thumb-pinky": np.inf,
    }

    # Slice only first 3 columns (x, y, z) for distance calculations
    positions = hand_pcloud_data[:, :, :3]

    for hand_pcloud_datum in positions:
        thumb_fingertip = hand_pcloud_datum[fingertip_indices["thumb"]]
        index_fingertip = hand_pcloud_datum[fingertip_indices["index"]]
        middle_fingertip = hand_pcloud_datum[fingertip_indices["middle"]]
        ring_fingertip = hand_pcloud_datum[fingertip_indices["ring"]]
        pinky_fingertip = hand_pcloud_datum[fingertip_indices["pinky"]]
        thumb_index_distance = np.linalg.norm(thumb_fingertip - index_fingertip)
        thumb_middle_distance = np.linalg.norm(thumb_fingertip - middle_fingertip)
        thumb_ring_distance = np.linalg.norm(thumb_fingertip - ring_fingertip)
        thumb_pinky_distance = np.linalg.norm(thumb_fingertip - pinky_fingertip)

        fingertip_min_distances["thumb-index"] = min(
            fingertip_min_distances["thumb-index"], thumb_index_distance
        )
        fingertip_min_distances["thumb-middle"] = min(
            fingertip_min_distances["thumb-middle"], thumb_middle_distance
        )
        fingertip_min_distances["thumb-ring"] = min(
            fingertip_min_distances["thumb-ring"], thumb_ring_distance
        )
        fingertip_min_distances["thumb-pinky"] = min(
            fingertip_min_distances["thumb-pinky"], thumb_pinky_distance
        )

    return fingertip_min_distances


def check_wrist_points_at_origin(
    hand_pcloud_data: np.ndarray, wrist_index: int = 0
) -> bool:
    """Checks if the wrist point is at the origin for all data points.

    Args:
        hand_pcloud_data: np.ndarray of shape (T, 25, 3) or (T, 25, 7) where T is number
                            of time points from mocap system.
        wrist_index: Index of the wrist point in the point cloud data.
    Returns:
        True if all wrist points are at the origin, False otherwise.
    """
    # Slice only first 3 columns (x, y, z)
    wrist_points = hand_pcloud_data[:, wrist_index, :3]
    return np.allclose(wrist_points, np.zeros(wrist_points.shape))


def calculate_pcloud_statistics(hand_pcloud_data: np.ndarray) -> dict[str, np.ndarray]:
    """Calculates statistics for the hand point cloud data.

    Args:
        hand_pcloud_data: np.ndarray of shape (N, 3) where N is number of points from mocap system.
    Returns:
        A dictionary containing mean and standard deviation of the point cloud data.
    """
    statistics = {
        "mean": np.mean(hand_pcloud_data, axis=0),
        "std_dev": np.std(hand_pcloud_data, axis=0),
    }
    return statistics


def save_statistics_to_csv(pcloud_statistics, filename="hand_stats.csv"):
    """Saves mean and std dev data to a CSV for Google Sheets."""
    means = pcloud_statistics["mean"]
    stds = pcloud_statistics["std_dev"]

    # Dynamically handle 3 or 7 columns
    num_features = means.shape[1]
    if num_features == 3:
        headers = [
            "Index",
            "Mean_X [m]",
            "Std_X [m]",
            "Mean_Y [m]",
            "Std_Y [m]",
            "Mean_Z [m]",
            "Std_Z [m]",
        ]
    elif num_features == 7:
        headers = [
            "Index",
            "Mean_X [m]",
            "Std_X [m]",
            "Mean_Y [m]",
            "Std_Y [m]",
            "Mean_Z [m]",
            "Std_Z [m]",
            "Mean_Qw",
            "Std_Qw",
            "Mean_Qx",
            "Std_Qx",
            "Mean_Qy",
            "Std_Qy",
            "Mean_Qz",
            "Std_Qz",
        ]
    else:
        error_msg = f"Unexpected number of features: {num_features}. Expected 3 or 7."
        raise ValueError(error_msg)

    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for i in range(len(means)):
            row = [i]
            for j in range(num_features):
                row.extend([means[i][j], stds[i][j]])
            writer.writerow(row)

    print(f"Data saved successfully to {filename}")


def load_hand_data(
    dataset_filename: str, logger: logging.Logger
) -> tuple[np.ndarray, dict]:
    """Handles file path resolution and loading.

    Args:
        dataset_filename: Name of the dataset file.
        logger: Logger instance for logging.
    Returns:
        A tuple containing the loaded data as a numpy array and metadata dictionary.
    """
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    data_path = (data_dir / dataset_filename).resolve()

    data = np.load(data_path, allow_pickle=True)

    metadata = {
        "hand_chirality": extract_hand_chirality_from_filename(dataset_filename),
        "hand_tracker": extract_hand_tracker_from_filename(dataset_filename),
        "subject_id": extract_subject_id_from_filename(dataset_filename),
        "run_id": extract_run_id_from_filename(dataset_filename),
        "datapath": data_path,
    }

    logger.info(f"Loaded data from: {data_path} (Shape: {data.shape})")
    return data, metadata


def log_results(
    pcloud_statistics: dict,
    fingertip_distances: dict,
    wrist_origin: bool,
    logger: logging.Logger,
):
    """Handles all terminal output and tabulate formatting."""
    logger.info("--- Inspection Results ---")

    # 1. Distances
    logger.info("Fingertip Minimum Distances:")
    for pair, dist in fingertip_distances.items():
        logger.info(
            f"  {pair:13}: {dist:.6f}m ({convert_m_to_cm(dist):.2f}cm | {convert_m_to_mm(dist):.2f}mm)"
        )

    logger.info(f"Wrist at origin: {wrist_origin}")

    # 2. Table Formatting (Handling 3 or 7 columns dynamically)
    means = pcloud_statistics["mean"]
    stds = pcloud_statistics["std_dev"]
    num_dims = means.shape[1]

    headers = [
        "Idx",
        "Mean X [m]",
        "Std X [m]",
        "Mean Y [m]",
        "Std Y [m]",
        "Mean Z [m]",
        "Std Z [m]",
    ]
    if num_dims == 7:
        headers += [
            "Mean Qw",
            "Std Qw",
            "Mean Qx",
            "Std Qx",
            "Mean Qy",
            "Std Qy",
            "Mean Qz",
            "Std Qz",
        ]

    table_data = []
    for i in range(len(means)):
        row = [i]
        for d in range(num_dims):
            row.extend([f"{means[i, d]:.5f}", f"{stds[i, d]:.5f}"])
        table_data.append(row)

    logger.info("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))


def main():
    # 1. Setup
    logger = initialize_logger(module_name=__name__, default_log_level=logging.INFO)
    args = parse_cli_args()

    # 2. Data Acquisition
    try:
        hand_data, meta = load_hand_data(args.dataset_filename, logger)
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        return

    if meta["hand_tracker"] is None:
        raise ValueError(
            f"Could not identify tracker type from {args.dataset_filename}"
        )

    # 3. Processing
    # Get indices based on the specific tracker used
    tracker_type = HandTracker(meta["hand_tracker"])
    indices = FINGERTIP_INDICES[tracker_type]

    distances = calculate_min_fingertip_distances(hand_data, fingertip_indices=indices)
    wrist_origin = check_wrist_points_at_origin(hand_data, wrist_index=0)
    stats = calculate_pcloud_statistics(hand_data)

    # 4. Output & Persistence
    log_results(stats, distances, wrist_origin, logger)

    csv_name = f"{get_current_datetime()}_stats_{meta['hand_tracker']}_{meta['hand_chirality']}_sub-{meta['subject_id']}.csv"
    if args.output_csv:
        save_statistics_to_csv(stats, filename=csv_name)
        logger.info(f"Saved statistics to {csv_name}")


if __name__ == "__main__":
    main()
