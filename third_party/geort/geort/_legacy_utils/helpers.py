"""Helper functions for various tasks.

Author(s):
    - Robert Jomar Malate (robert.malate@mimicrobotics.com)
"""

# Standard
import logging
from pathlib import Path
import time

# Third-party
import numpy as np

# Custom
from geort._legacy_utils.common import (
    GC_ANGLE_OFFSETS,
)


def initialize_logger(
    module_name: str, default_log_level=logging.DEBUG
) -> logging.Logger:
    """Initializes a logger for the script.

    Returns:
        A logging.Logger instance.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(default_log_level)

    ch = logging.StreamHandler()
    ch.setLevel(default_log_level)

    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    return logger


def convert_m_to_mm(distance_m: float) -> float:
    """Converts distance from meters to millimeters.

    Args:
        distance_m: Distance in meters.
    Returns:
        Distance in millimeters.
    """
    return distance_m * 1000.0


def convert_m_to_cm(distance_m: float) -> float:
    """Converts distance from meters to centimeters.

    Args:
        distance_m: Distance in meters.
    Returns:
        Distance in centimeters.
    """
    return distance_m * 100.0


def convert_urdf_to_robot_joint_cmds(joint_cmd_ROS2_deg: np.ndarray) -> np.ndarray:
    """Converts URDF joint positions to robot joint commands.

    This method converts between joint positions used in the URDF model to
    the joint commands that are are sent to the robot over ROS2 commands.

    Note:
    - The URDF joint positions start from 0.0 and have units of radians. This
    was done to account for a calibration pose.
    - The robot joint commands have ranges that are more human readable
    and are in degrees. These are the values that are sent over ROS2 commands.

    Args:
        joint_cmd_urdf_rad: np.ndarray of shape (N,) representing joint positions for URDF.
    Returns:
        np.ndarray of shape (N,) representing joint commands for the robot.
    """
    return np.deg2rad(joint_cmd_ROS2_deg + GC_ANGLE_OFFSETS)


def convert_robot_to_urdf_joint_cmds(joint_cmd_urdf_rad: np.ndarray) -> np.ndarray:
    """Converts robot joint commands to URDF joint positions.

    This method converts between joint commands that are are sent to
    the robot over ROS2 commands to the joint positions used in the URDF model.

    Note:
    - The URDF joint positions start from 0.0 and have units of radians. This
    was done to account for a calibration pose.
    - The robot joint commands have ranges that are more human readable
    and are in degrees. These are the values that are sent over ROS2 commands.

    Args:
        joint_cmd_urdf_rad: np.ndarray of shape (N,) representing joint positions for URDF.
    Returns:
        np.ndarray of shape (N,) representing joint commands for the robot.
    """
    return np.rad2deg(joint_cmd_urdf_rad) - GC_ANGLE_OFFSETS


def find_project_root(anchor_name: str = "geort") -> Path:
    """Finds the root directory of the project by searching upwards.

    This function starts at the current file's location and walks up the
    directory tree until it finds a directory containing the 'anchor_name'.

    If the anchor is not found, it raises a FileNotFoundError.

    Args:
        anchor_name: The name of a folder or file that exists in the root
                     directory (e.g., 'geort', '.git', 'setup.py').
                     Defaults to 'geort' based on your project structure.

    Returns:
        Path: The absolute path to the project root directory.
    """
    # Start at the directory of this helper file
    current_path = Path(__file__).resolve()

    # Walk up the tree
    for parent in [current_path] + list(current_path.parents):
        # Check if the anchor exists inside this parent directory
        potential_anchor = parent / anchor_name

        # If the anchor itself is the directory we are looking for (e.g. 'geort')
        # OR if the anchor is a file/folder *inside* the current parent
        if potential_anchor.exists():
            # If the anchor we found IS the root folder (e.g. we found 'geort' folder),
            # return the parent of that folder to get the workspace root.
            # If 'geort' is a subfolder of the root, return parent.
            return parent

    raise FileNotFoundError(
        f"Could not find project root. Searched upwards for '{anchor_name}' "
        f"starting from {current_path}."
    )


def get_current_datetime() -> str:
    """Generate a current datetime string in the format 'YYYY-MM-DD_HH-MM-SS_<timezone>'."""
    return time.strftime("%Y-%m-%d_%H-%M-%S_%Z", time.localtime())
