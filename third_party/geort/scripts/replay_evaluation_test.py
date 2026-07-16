"""Feeds recorded point cloud data to a trained GeoRT model and visualizes the predicted hand pose.

- Visualizes the output of a trained GeoRT model on pre-recorded
hand tracker point cloud data.
- Visualization is done using MeshCat since it integrates well with
Pinocchio kinematic models and is lightweight for quick inspection.
- This is useful for quickly evaluating the model performance without
needing to deploy it.

Author(s):
    - Robert Jomar Malate (robert.malate@mimicrobotics.com)
"""

# Standard
import time
import argparse
import logging

# Third-party
import numpy as np
import pinocchio as pin
import torch
from pathlib import Path
from pinocchio.visualize import MeshcatVisualizer

# Custom
from geort import load_model, get_config
from geort.env.hand_min import HandKinematicModel
from utils.helpers import initialize_logger
import utils.argparse_utils as argparse_utils
from utils.dataset import extract_positions_and_orientations_from_dataset


class HandVisualizer(HandKinematicModel):
    def __init__(
        self,
        urdf_path: str,
        joint_names: list[str],
        mesh_dir: str,
        logger: logging.Logger,
    ):
        # 1. Initialize the parent (Loads kinematics, sets up couplings/IDs)
        super().__init__(urdf_path, joint_names)
        self.logger = logger

        # 2. Load Visual Geometry
        # We call buildModelsFromUrdf to get the visual_model.
        # We ignore the first return (model) because parent already loaded self.model
        _, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path, mesh_dir
        )

        # 3. Setup MeshCat
        # We use the parent's self.model so the kinematics match perfectly
        self.logger.info("Initializing MeshCat Viewer...")
        self.viz = MeshcatVisualizer(
            self.model, self.collision_model, self.visual_model
        )
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel()
        self.logger.info("MeshCat initialized. Check your browser.")

    def update_display(self, q_user):
        """
        Takes user joint angles (radians), converts to Pinocchio q, and updates viewer.
        """
        # Use the parent's conversion logic to handle coupling/reordering
        q_pin = self.convert_user_q_to_pinocchio_q(q_user)
        self.viz.display(q_pin)


# --- HELPER FUNCTIONS ---
def parse_cli_args() -> argparse.ArgumentParser:
    """Initializes the argument parser for the script.

    Returns:
        An argparse.ArgumentParser instance.
    """
    parser = argparse_utils.create_argparser(
        description="Replay GeoRT inference on recorded point cloud data.",
    )
    argparse_utils.add_args(parser, "dataset_filename", "hand")
    parser.add_argument("--ckpt_tag", type=str, help="GeoRT Checkpoint Tag")
    parser.add_argument("--fps", type=int, default=30, help="Playback speed")
    return parser.parse_args()


def main():
    logger = initialize_logger("replay_evaluation_test")
    args = parse_cli_args()

    # Loading data
    dataset_filename = args.dataset_filename
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    data_path = (data_dir / dataset_filename).resolve()
    if not data_path.exists():
        error_msg = f"Data file not found: {data_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Loading data from: {data_path}")
    dataset = np.load(data_path)
    pcloud_data, _ = extract_positions_and_orientations_from_dataset(dataset)
    if (
        len(pcloud_data.shape) != 3
        or pcloud_data.shape[1] != 25
        or pcloud_data.shape[2] != 3
    ):
        error_msg = f"Data shape mismatch. Expected (N, 25, 3), got {pcloud_data.shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Load GeoRT model
    model_ckpt_tag = args.ckpt_tag
    logger.info(f"Loading GeoRT model '{model_ckpt_tag}'...")
    model = load_model(model_ckpt_tag)

    # Setup visualizer
    config = get_config(args.hand)
    urdf_path = Path(config["urdf_path"])
    mesh_dir = str(urdf_path.resolve().parent)
    hand_viz = HandVisualizer(
        urdf_path=str(urdf_path),
        joint_names=config["joint_order"],
        mesh_dir=mesh_dir,
        logger=logger,
    )
    logger.info(f"Loaded URDF from: {urdf_path}")

    # Playback loop
    logger.info(f"Starting playback of {len(pcloud_data)} frames...")
    dt = 1.0 / args.fps

    try:
        for i, pcloud_data_i in enumerate(pcloud_data):
            loop_start = time.time()

            # Inference
            with torch.no_grad():
                q_pred = model.forward(pcloud_data_i)

            # Move Robot
            q_user = q_pred.copy()
            hand_viz.update_display(q_user)

            # Timing
            elapsed = time.time() - loop_start
            time.sleep(max(0, dt - elapsed))

            if i % 30 == 0:
                logger.info(f"Frame {i}/{len(pcloud_data)}")

    except KeyboardInterrupt:
        logger.info("Stopped by user.")


if __name__ == "__main__":
    main()
