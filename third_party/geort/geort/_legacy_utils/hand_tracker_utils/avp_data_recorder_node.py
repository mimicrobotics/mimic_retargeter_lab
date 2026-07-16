"""ROS2 Node to record AVP hand tracker data from normalized point clouds.

Author(s):
    - Robert Jomar Malate (robert.malate@mimicrobotics.com)
"""

# ROS2
import rclpy

# Custom
from geort._legacy_utils.hand_tracker_utils.hand_tracker_data_recorder import (
    HandTrackerDataRecorderNode,
)
from geort._legacy_utils.common import HandTracker


class AVPDataRecorderNode(HandTrackerDataRecorderNode):
    def __init__(self):
        super().__init__(hand_tracker=HandTracker.AVP)
        self.get_logger().info("AVPDataRecorder node started.")


# Used for debugging
def main(args=None):
    rclpy.init(args=args)
    node = AVPDataRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down AVPDataRecorderNode...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
