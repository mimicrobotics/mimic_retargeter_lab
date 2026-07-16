"""Transform point-cloud joint data collected from a hand tracker.

Applies necessary transformations to point cloud data to match
expected GeoRT conventions.

Author(s):
    - Robert Jomar Malate (robert.malate@mimicrobotics.com)
"""

# Standard
import argparse
from pathlib import Path
import logging

# Third-party
import numpy as np

# Custom
from utils.helpers import initialize_logger
from utils.argparse_utils import (
    create_argparser,
    add_dataset_filename_arg,
)


# Rotation matrix to convert from hand tracker data from mimic_robotics
# to GeoRT conventions. It is with respect to the GeoRT base frame.
ROTATION_MATRIX_MIMIC_TO_GEORT = np.array(
    [
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
WRIST_INDEX = 0


# --- TRANSFORM FUNCTIONS ---
def apply_rotation(data: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Applies rotation to point cloud data.

    Args:
        data: Point cloud data of shape (N, P, 3), where N is the number of frames,
              P is the number of points, and 3 corresponds to (x, y, z) coordinates.
        rotation_matrix: A 3x3 rotation matrix.
    Returns:
        Rotated point cloud data of the same shape as input.
    """
    return data @ rotation_matrix.T


def apply_translation(data: np.ndarray, translation_matrix: np.ndarray) -> np.ndarray:
    """Applies translation to point cloud data.

    Args:
        data: Point cloud data of shape (N, P, 3), where N is the number of frames,
              P is the number of points, and 3 corresponds to (x, y, z) coordinates.
        translation_vector: A 1D array of length 3 representing the translation in (x, y, z).
    Returns:
        Translated point cloud data of the same shape as input.
    """
    N, P, _ = data.shape
    translated_data = data - translation_matrix[:, None, :]
    return translated_data


# --- HELPER FUNCTIONS ---
def get_cli_args() -> argparse.Namespace:
    parser = create_argparser(
        description="Transform point-cloud joint data collected from a hand tracker."
    )
    parser = add_dataset_filename_arg(parser)
    return parser.parse_args()


def main():
    logger = initialize_logger(
        module_name=__name__,
        default_log_level=logging.INFO,
    )
    args = get_cli_args()

    # Resolve paths relative to this script
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    data_path = (data_dir / args.dataset_filename).resolve()
    if not data_path.exists():
        error_msg = f"Data file not found: {data_path}"
        raise FileNotFoundError(error_msg)
    logger.info(f"Loading data from: {data_path}")

    # Load data
    data = np.load(data_path)  # (N, 25, 3)
    logger.info(f"Original data shape: {data.shape}")

    # Apply transformations
    transformed_data = apply_rotation(data, ROTATION_MATRIX_MIMIC_TO_GEORT)
    logger.info(f"Transformed data shape: {transformed_data.shape}")

    # Save transformed data
    transformed_filename = args.dataset_filename.replace(".npy", "_transformed.npy")
    transformed_path = (data_dir / transformed_filename).resolve()
    np.save(transformed_path, transformed_data)
    logger.info(f"Saved transformed data to: {transformed_path}")


if __name__ == "__main__":
    main()
