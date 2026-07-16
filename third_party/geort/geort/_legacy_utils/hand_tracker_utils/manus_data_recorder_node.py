"""ROS2 Node to record MANUS hand tracker data from normalized point clouds.

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


class ManusDataRecorderNode(HandTrackerDataRecorderNode):
    def __init__(self):
        super().__init__(hand_tracker=HandTracker.MANUS)
        self.get_logger().info("ManusDataRecorder node started.")


# Used for debugging
def main(args=None):
    rclpy.init(args=args)
    node = ManusDataRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down ManusDataRecorderNode...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
