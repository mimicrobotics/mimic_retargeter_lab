"""Subscribes to point cloud data from ROS2 and visualizes inferred hand pose from GeoRT model.

- Subscribes to point clouds (Float32MultiArray), runs inference, and visualizes in MeshCat.
- Topic: /hand_tracker/right/pcloud_normalized (std_msgs/Float32MultiArray)
       Expected layout: Flat array of size 75 (25 points * 3 coords)

Author(s):
    - Robert Jomar Malate (robert.malate@mimicrobotics.com)
"""

# Standard
import argparse
import numpy as np
from pathlib import Path
import logging

# ROS 2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# GeoRT / Custom
import torch
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from geort import load_model, get_config
from geort.env.hand_min import HandKinematicModel
import utils.argparse_utils as argparse_utils
from utils.models import extract_hand_tracker_from_checkpoint_name


class HandVisualizer(HandKinematicModel):
    def __init__(
        self,
        urdf_path: str,
        joint_names: list[str],
        mesh_dir: str,
        logger: logging.Logger,
    ):
        super().__init__(urdf_path, joint_names)

        # Load Visuals
        _, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path, mesh_dir
        )

        # Setup MeshCat
        print("Initializing MeshCat Viewer...")
        self.viz = MeshcatVisualizer(
            self.model, self.collision_model, self.visual_model
        )
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel()
        print("MeshCat initialized.")

    def update_display(self, q_user):
        q_pin = self.convert_user_q_to_pinocchio_q(q_user)
        self.viz.display(q_pin)


class GeoRTInferenceNode(Node):
    def __init__(self, hand_config_name: str, ckpt_tag: str, topic_name: str):
        super().__init__("geort_inference_visualizer")
        self.logger = self.get_logger()

        # Load GeoRT Model
        self.model = load_model(ckpt_tag)
        info_msg = f"Loaded model: {ckpt_tag}"
        self.logger.info(f"{info_msg}")

        # Setup Hand Visualizer
        self.logger.info(f"Loading Hand Config '{hand_config_name}'...")
        hand_tracker = extract_hand_tracker_from_checkpoint_name(ckpt_tag)
        config = get_config(hand_config_name, hand_tracker)
        urdf_path = Path(config["urdf_path"])

        # Resolve mesh directory (assuming assets/p50/meshes structure)
        mesh_dir = str(urdf_path.resolve().parent)

        self.viz = HandVisualizer(
            urdf_path=str(urdf_path),
            joint_names=config["joint_order"],
            mesh_dir=mesh_dir,
            logger=self.logger,
        )

        # Setup Subscriber
        self.topic_name = topic_name

        self.subscription = self.create_subscription(
            Float32MultiArray,
            self.topic_name,
            self.listener_callback,
            10,  # QoS History depth
        )
        self.logger.info(f"Subscribed to: {self.topic_name} (Float32MultiArray)")

    def listener_callback(self, msg: Float32MultiArray):
        """
        Callback for new point cloud data.
        """
        # 1. Parse Data
        # Expected flat shape: 75 (25 keypoints * 3 xyz)
        data = np.array(msg.data, dtype=np.float32)

        if data.size != 75:
            self.logger.warn(
                f"Received array of size {data.size}, expected 75 (25*3). Skipping."
            )
            return

        # Reshape to (25, 3)
        pcloud = data.reshape(25, 3)

        # 2. Inference
        try:
            with torch.no_grad():
                # Model wrapper handles tensor conversion internally
                q_pred = self.model.forward(pcloud)

            # 3. Handle Output Type
            if isinstance(q_pred, torch.Tensor):
                q_user = q_pred.cpu().detach().numpy().flatten()
            else:
                q_user = np.array(q_pred).flatten()

            # 4. Update Visualization
            self.viz.update_display(q_user)

        except Exception as e:
            self.logger.error(f"Inference failed: {e}")


# --- HELPER FUNCTIONS ---
def parse_cli_args() -> argparse.ArgumentParser:
    parser = argparse_utils.create_argparser(
        description="ROS2 Node for GeoRT Inference Visualization.",
    )
    argparse_utils.add_args(parser, "hand")
    parser.add_argument(
        "--ckpt_tag", type=str, required=True, help="GeoRT Checkpoint Tag"
    )
    parser.add_argument(
        "--topic_name",
        type=str,
        help="ROS2 topic name for point cloud data",
        required=True,
    )

    parsed_args, _ = parser.parse_known_args()
    return parsed_args


def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)

    # Parse CLI Args (using argparse separately to avoid conflict with ROS args)
    parsed_args = parse_cli_args()
    node = GeoRTInferenceNode(
        hand_config_name=parsed_args.hand,
        ckpt_tag=parsed_args.ckpt_tag,
        topic_name=parsed_args.topic_name,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
