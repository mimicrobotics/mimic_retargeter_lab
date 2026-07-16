"""Tests the forward kinematics of the robotic hand.

Author(s):
    - Robert Jomar Malate (robert.malate@mimicrobotics.com)
"""

# Standard
import argparse

# Third-party
import numpy as np

# Custom
from geort.env.hand_min import HandKinematicModel
from geort.utils.config_utils import get_config
from utils.helpers import initialize_logger
import utils.argparse_utils as argparse_utils
from utils.common import GRASP_PRIMITIVES, FingerName


# --- HELPER FUNCTIONS ---
def parse_cli_args() -> argparse.ArgumentParser:
    """Initializes the argument parser for the script.

    Returns:
        An argparse.ArgumentParser instance.
    """
    parser = argparse_utils.create_argparser(
        description="Test forward kinematics of the robotic hand.",
    )
    argparse_utils.add_args(parser, "hand")
    return parser.parse_args()


def compare_fingertip_positions(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    tolerance: float = 1e-3,
) -> bool:
    """Compares two sets of fingertip positions.
    Args:
        pos_a: First set of fingertip positions (N x 3).
        pos_b: Second set of fingertip positions (N x 3).
        tolerance: Tolerance for comparison.

    Returns:
        True if all corresponding fingertip positions are within the specified tolerance.
    """
    # Ensuring both inputs are numpy arrays with floats
    pos_a = np.array(pos_a, dtype=np.float64)
    pos_b = np.array(pos_b, dtype=np.float64)
    return np.allclose(pos_a, pos_b, atol=tolerance)


def main():
    logger = initialize_logger("test_forward_kinematics")
    args = parse_cli_args()

    # Load config and kinematic model
    config = get_config(args.hand)
    model = HandKinematicModel.build_from_config(config, render=True)
    logger.info(f"Loaded hand kinematic model for '{args.hand}'.")

    # Initializing kinematic model
    keypoint_links = [info["link"] for info in config["fingertip_link"]]
    keypoint_offsets = [info["center_offset"] for info in config["fingertip_link"]]
    model.initialize_keypoint(keypoint_links, keypoint_offsets)

    fingertip_grouping = [
        ("thumb_fingertip", FingerName.THUMB),
        ("index_fingertip", FingerName.INDEX),
        ("middle_fingertip", FingerName.MIDDLE),
        ("ring_fingertip", FingerName.RING),
        ("pinky_fingertip", FingerName.PINKY),
    ]

    # Testing FK for each grasp primitive
    error_tolerance = 1e-3  # Units vary but it's more about decimal places
    for grasp_primitive, data in GRASP_PRIMITIVES.items():
        grasp_primitive_qpos = data.joint_commands.urdf_rad
        output_fk_fingertip = model.keypoint_from_qpos(
            grasp_primitive_qpos, ret_orientation=True
        )

        output_fingertip_list = []
        expected_fingertip_list = []

        for link_name, finger_enum in fingertip_grouping:
            output_pos, output_quat = output_fk_fingertip[link_name]
            output_vec = np.concatenate((output_pos, output_quat))
            output_fingertip_list.append(output_vec)

            expected_pose = data.ground_truth_pose.fingers[finger_enum].fingertip_pose
            expected_pos = np.array(expected_pose.position, dtype=np.float64).flatten()
            expected_quat = np.array(
                expected_pose.orientation, dtype=np.float64
            ).flatten()
            expected_vec = np.concatenate((expected_pos, expected_quat))
            expected_fingertip_list.append(expected_vec)

        output_fingertip_positions = np.stack(output_fingertip_list)
        expected_fingertip_positions = np.stack(expected_fingertip_list)

        if compare_fingertip_positions(
            expected_fingertip_positions,
            output_fingertip_positions,
            tolerance=error_tolerance,
        ):
            logger.info(
                f"Fingertip poses for '{grasp_primitive}' match expected values within tolerance of {error_tolerance}."
            )
        else:
            logger.info(
                f"Fingertip poses for '{grasp_primitive}' do NOT match expected values or fall within tolerance of {error_tolerance}!"
            )


if __name__ == "__main__":
    main()
