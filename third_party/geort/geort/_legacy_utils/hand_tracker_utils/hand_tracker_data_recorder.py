"""ROS2 Node to record hand tracker data from normalized point clouds.

Author(s):
    - Robert Jomar Malate (robert.malate@mimicrobotics.com)
"""

# Standard
import yaml

# ROS2
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# Third-party
import numpy as np

# Custom
from geort._legacy_utils.helpers import HandTracker, find_project_root
from geort._legacy_utils.common import HandChirality


class HandTrackerDataRecorderNode(Node):
    def __init__(self, hand_tracker: HandTracker):
        super().__init__("hand_tracker_data_recorder")
        self._hand_tracker: HandTracker = hand_tracker
        self._load_config_params(hand_tracker=self._hand_tracker)

        # Subscribers
        self.hand_tracker_right_normalized_pcloud_sub = self.create_subscription(
            Float32MultiArray,
            self._hand_tracker_right_normalized_pcloud_topic_name,
            self.hand_tracker_right_normalized_pcloud_callback,
            10,
        )

        self.hand_tracker_left_normalized_pcloud_sub = self.create_subscription(
            Float32MultiArray,
            self._hand_tracker_left_normalized_pcloud_topic_name,
            self.hand_tracker_left_normalized_pcloud_callback,
            10,
        )

        # Shape is (N, 3) where N is number of points
        self._latest_hand_right_data: np.ndarray | None = None
        self._latest_hand_left_data: np.ndarray | None = None

        self.get_logger().debug("HandTrackerDataRecorder node started.")

    def _load_config_params(self, hand_tracker: HandTracker):
        """Load configuration parameters from ROS2 parameter server.

        Must set the follwing parameters:
            self._hand_tracker_right_normalized_pcloud_topic_name
            self._hand_tracker_left_normalized_pcloud_topic_name
        """
        # Resolving path
        root_dir = find_project_root()
        config_filename = "hand_tracker_data_recorder.yaml"
        config_file_path = root_dir / "geort" / "config" / config_filename

        # Loading yaml
        config_data = {}
        try:
            with open(config_file_path, "r") as f:
                full_config = yaml.safe_load(f)
                config_data = full_config.get(hand_tracker, {})
            self.get_logger().info(
                f"Loaded hand tracker data recorder config from {config_file_path} for '{hand_tracker}'"
            )
        except Exception as e:
            error_msg = f"Failed to load config file at {config_file_path}: {e}"
            self.get_logger().error(error_msg)

        # Extract values
        self._hand_tracker_right_normalized_pcloud_topic_name: str = config_data.get(
            "hand_tracker_right_normalized_pcloud_topic_name_string", ""
        )
        self._hand_tracker_left_normalized_pcloud_topic_name: str = config_data.get(
            "hand_tracker_left_normalized_pcloud_topic_name_string", ""
        )

        if (
            not self._hand_tracker_right_normalized_pcloud_topic_name
            or not self._hand_tracker_left_normalized_pcloud_topic_name
        ):
            error_msg = (
                f"Missing topic names in config for hand tracker '{hand_tracker}'"
            )
            self.get_logger().error(error_msg)
            raise ValueError(error_msg)

        self.get_logger().info(
            f"Loaded parameters:\n"
            f"  hand_tracker_right_normalized_pcloud_topic_name: {self._hand_tracker_right_normalized_pcloud_topic_name}\n"
            f"  hand_tracker_left_normalized_pcloud_topic_name: {self._hand_tracker_left_normalized_pcloud_topic_name}"
        )

    def _msg_to_points(self, msg: Float32MultiArray) -> np.ndarray:
        """Convert Float32MultiArray message to Nx3 or Nx7 numpy array of points.

        Converts flat float array [x1, y1, z1, x2, y2, z2, ...] into shape (N, 3) if pcloud
        or [x1, y1, z1, qx1, qy1, qz1, qw1, x2, y2, z2, qx2, qy2, qz2, qw2, ...] into shape
        (N, 7) for link poses.
        """
        data = np.array(msg.data, dtype=np.float32)
        # breakpoint()
        if data.size % 3 != 0 and data.size % 7 != 0:
            warning_msg = f"Received pointcloud with {data.size} elements, not divisible by 3 or by 7."
            self.get_logger().warn(warning_msg)
            return data.reshape(-1)  # fallback

        if data.size % 3 == 0:
            points = data.reshape(-1, 3)
            # return points
        if data.size % 7 == 0:
            points = data.reshape(-1, 7)
            # return points
        # points = data.reshape(-1, 3)
        return points

    # --- Callbacks ---
    def hand_tracker_right_normalized_pcloud_callback(self, msg: Float32MultiArray):
        points = self._msg_to_points(msg)
        self._latest_hand_right_data = points
        self.get_logger().debug(f"Right hand data shape: {points.shape}")

    def hand_tracker_left_normalized_pcloud_callback(self, msg: Float32MultiArray):
        points = self._msg_to_points(msg)
        self._latest_hand_left_data = points
        self.get_logger().debug(f"Left hand data shape: {points.shape}")

    # --- Public Methods ---
    def get_latest_hand_data(
        self,
        hand_chirality: HandChirality,
        clear_after_read: bool = False,
    ) -> np.ndarray | None:
        """Get the latest hand data for the specified chirality.

        - If no message has been received yet: returns None.
        - Otherwise: returns a *copy* of the latest (N,3) array.

        Args:
            hand_chirality: HandChirality.RIGHT or HandChirality.LEFT
            clear_after_read: If True, sets internal buffer to None after returning.

        Returns:
            np.ndarray of shape (N, 3) or None
        """
        hand_data = None

        if hand_chirality == HandChirality.RIGHT:
            hand_data = self._latest_hand_right_data
            if hand_data is not None and clear_after_read:
                self._latest_hand_right_data = None

        elif hand_chirality == HandChirality.LEFT:
            hand_data = self._latest_hand_left_data
            if hand_data is not None and clear_after_read:
                self._latest_hand_left_data = None

        else:
            error_msg = f"Invalid hand chirality: {hand_chirality}"
            raise ValueError(error_msg)

        return None if hand_data is None else hand_data.copy()

    @property
    def right_pcloud_topic_name(self) -> str:
        return self._hand_tracker_right_normalized_pcloud_topic_name

    @property
    def left_pcloud_topic_name(self) -> str:
        return self._hand_tracker_left_normalized_pcloud_topic_name
