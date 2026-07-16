"""Captures motion capture hand data and saves to disk.

Author(s):
    - Robert Jomar Malate (robert.malate@mimicrobotics.com)
"""

# Standard
import logging
import time

# ROS2
import rclpy
from rclpy.executors import SingleThreadedExecutor

# Third-party
import numpy as np
from tqdm import tqdm
import geort

# Custom
from utils.hand_tracker_utils.hand_tracker_data_recorder import HandChirality
from utils.hand_tracker_utils.manus_data_recorder_node import ManusDataRecorderNode
from utils.hand_tracker_utils.avp_data_recorder_node import AVPDataRecorderNode
from utils.common import HandTracker
from utils.helpers import initialize_logger, get_current_datetime
import utils.argparse_utils as argparse_utils


class MocapSystem:
    def __init__(
        self,
        node,
        executor: SingleThreadedExecutor,
        hand_chirality: HandChirality = HandChirality.RIGHT,
        timeout_sec: float = 0.05,
    ):
        """
        Args:
            node: An instance of ManusDataRecorderNode.
            executor: A rclpy executor that will call spin_once().
            hand: Which hand to read (RIGHT or LEFT).
            timeout_sec: spin_once timeout per get() call.
        """
        self._node = node
        self._executor = executor
        self._hand_chirality = hand_chirality
        self._timeout_sec = timeout_sec

    def get(self) -> np.ndarray | None:
        """
        Process one round of ROS2 callbacks and return latest hand data.

        Returns:
            np.ndarray of shape (N, 3) where N is number of points (21 for MANO), or
            None if no data is available.
        """
        self._executor.spin_once(timeout_sec=self._timeout_sec)

        hand_data = self._node.get_latest_hand_data(
            hand_chirality=self._hand_chirality,
            clear_after_read=False,
        )

        if hand_data is None:
            return None

        return hand_data


# --- Helper functions ---
def initialize_argparser():
    parser = argparse_utils.create_argparser(
        description="Capture motion capture hand data and save to disk."
    )

    # Identity arguments (for filename and metadata)
    parser.add_argument(
        "--subject_id",
        type=str,
        help="Subject identifier (ex. RJM, AAA)",
        required=True,
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="000",
        help="Run identifier (ex. 000, 001, 002)",
    )

    # Configuration arguments
    parser.add_argument(
        "--hand_chirality",
        type=str,
        choices=[str(HandChirality.RIGHT), str(HandChirality.LEFT)],
        help="Chirality of hand",
        required=True,
    )
    parser.add_argument(
        "--hand_tracker",
        type=str,
        help="Hand tracker system to use",
        choices=[HandTracker.MANUS, HandTracker.AVP],
        required=True,
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of samples to collect",
    )
    parser.add_argument(
        "--recording_frequency",
        type=int,
        default=50,
        help="Frequency to record data [Hz]",
    )

    return parser.parse_args()


def main():
    args = initialize_argparser()
    logger = initialize_logger(
        module_name="mocap_data_capture", default_log_level=logging.INFO
    )

    # Extracting args
    subject_id: str = args.subject_id
    run_id: str = args.run_id
    hand_chirality: HandChirality = HandChirality[args.hand_chirality.upper()]
    hand_tracker: HandTracker = HandTracker(args.hand_tracker)
    num_samples: int = args.num_samples
    recording_frequency: int = args.recording_frequency

    # Dataset information
    dataset_id = (
        f"{str(hand_chirality)}_{hand_tracker}_subject-{subject_id}_run-{run_id}"
    )
    sleep_between_samples_sec = 1.0 / recording_frequency

    dataset_metadata = {
        "dataset_id": dataset_id,
        "session_config": {
            "hand_tracker": hand_tracker,
            "hand_chirality": str(hand_chirality),
            "subject_id": subject_id,
            "run_id": run_id,
            "recording_frequency_hz": recording_frequency,
            "datetime_created": get_current_datetime(),
        },
        "ros2_information": {},
        "recording_metrics": {},
    }

    # ROS2 initialization
    rclpy.init()
    node = None
    if args.hand_tracker == HandTracker.MANUS:
        node = ManusDataRecorderNode()
    elif args.hand_tracker == HandTracker.AVP:
        node = AVPDataRecorderNode()
    if node is None:
        raise RuntimeError(
            "Failed to create data recorder node. Input a valid value for hand_tracker."
        )

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    if hasattr(node, "right_pcloud_topic_name") and hasattr(
        node, "left_pcloud_topic_name"
    ):
        dataset_metadata["ros2_information"]["topics"] = [
            node.right_pcloud_topic_name
            if hand_chirality == HandChirality.RIGHT
            else node.left_pcloud_topic_name
        ]

    # Mocap system initialization
    mocap = MocapSystem(
        node=node,
        executor=executor,
        hand_chirality=hand_chirality,
    )
    captured_data = []

    # Program initialization
    progress_bar = tqdm(
        total=num_samples,
        desc="Capturing hand data",
        unit="samples",
        initial=0,
    )

    try:
        logger.info("Starting data capture...")
        while len(captured_data) < num_samples and rclpy.ok():
            hand_data = mocap.get()

            if hand_data is not None:
                captured_data.append(hand_data)

                progress_bar.update(1)
                progress_bar.set_postfix(collected=len(captured_data))

            else:
                logger.debug("No hand data detected in this frame.")

            time.sleep(sleep_between_samples_sec)

        # Close progress bar since recording is done
        progress_bar.close()

        if len(captured_data) > 0:
            dataset_metadata["recording_metrics"] = {
                "total_samples_captured": len(captured_data),
                "data_shape": np.array(captured_data).shape,
                "duration_seconds": len(captured_data) / recording_frequency,
            }

            geort.save_data(
                human_data=captured_data,
                metadata=dataset_metadata,
                tag=dataset_id,
                logger=logger,
            )
            logger.info(
                f"Data capture complete. Saved {args.num_samples} samples to {dataset_id} directory."
            )

        else:
            info_msg = "Data capture ended before reaching target sample count."
            logger.info(info_msg)

    except KeyboardInterrupt:
        info_msg = "Data capture interrupted by user."
        logger.info(info_msg)
    finally:
        executor.remove_node(node)
        node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
