"""Visualizes the workspace of a robotic hand.

Author(s):
    - Robert Jomar Malate (robert.malate@mimicrobotics.com)
"""

# Standard
import argparse
import os

# Third-party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as meshcat_geometry

# Custom
from geort.env.hand_min import HandKinematicModel
from geort.utils.config_utils import get_config
from utils.helpers import initialize_logger
import utils.argparse_utils as argparse_utils


# --- CONSTANTS ---
FINGER_COLORS = {
    "thumb": [1.0, 0.0, 0.0],  # Red
    "index": [0.0, 1.0, 0.0],  # Green
    "middle": [0.0, 0.0, 1.0],  # Blue
    "ring": [0.5, 0.0, 0.5],  # Purple
    "pinky": [1.0, 0.5, 0.0],  # Orange
    "default": [0.5, 0.5, 0.5],  # Gray
}


# --- HELPER FUNCTIONS ---
def parse_cli_args() -> argparse.ArgumentParser:
    parser = argparse_utils.create_argparser(
        description="Visualize the workspace of a robotic hand by sweeping its joint space."
    )
    argparse_utils.add_args(parser, "hand")
    parser.add_argument(
        "--export", action="store_true", help="Save the point cloud data to disk."
    )
    parser.add_argument(
        "--num_samples", type=int, default=5000, help="Number of sweep samples."
    )
    parser.add_argument(
        "--viz_3d", action="store_true", help="Launch MeshCat URDF visualization."
    )

    return parser.parse_args()


def setup_kinematic_model(hand_name: str) -> tuple[HandKinematicModel, list[str]]:
    """Loads the model and initializes fingertip keypoints."""
    config = get_config(hand_name)
    model = HandKinematicModel.build_from_config(config, render=False)

    keypoint_links = [info["link"] for info in config["fingertip_link"]]
    keypoint_offsets = [info["center_offset"] for info in config["fingertip_link"]]
    model.initialize_keypoint(keypoint_links, keypoint_offsets)

    return model, keypoint_links


def generate_workspace_cloud(
    model: HandKinematicModel, keypoint_links: list[str], num_samples: int = 5000
) -> dict[str, np.ndarray]:
    """Sweeps joint space and returns a dictionary of point arrays for each finger."""
    q_min, q_max = model.get_joint_limit()
    fingertip_data = {link: [] for link in keypoint_links}

    for _ in range(num_samples):
        # Sample independent user joints (16 DOF)
        q_random = np.random.uniform(q_min, q_max)

        # FK handles joint coupling internally
        fk_results = model.keypoint_from_qpos(q_random, ret_orientation=False)

        for link_name in keypoint_links:
            fingertip_data[link_name].append(fk_results[link_name])

    # Convert lists to numpy arrays (N x 3)
    return {link: np.array(pts) for link, pts in fingertip_data.items()}


def get_color_for_link(link_name: str) -> list[float]:
    """Retrieves color based on finger naming conventions."""
    for finger, color in FINGER_COLORS.items():
        if finger in link_name.lower():
            return color
    return FINGER_COLORS["default"]


def create_matplotlib_plot(hand_name: str, cloud_data: dict[str, np.ndarray]):
    """Creates interactive Matplotlib plot with toggles and reset."""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(left=0.3, right=0.95, bottom=0.1, top=0.9)

    scatters = []
    labels = list(cloud_data.keys())

    for link_name, pts in cloud_data.items():
        color = get_color_for_link(link_name)
        scatter = ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2], s=4, c=[color], label=link_name, alpha=0.5
        )
        scatters.append(scatter)

    ax.set_title(
        f"{hand_name} Workspace (Matplotlib)", pad=20, fontsize=16, fontweight="bold"
    )
    fig.legend(
        loc="upper left",
        bbox_to_anchor=(0.05, 0.95),
        title="Fingertip Legend",
        frameon=True,
        shadow=True,
    )

    # Toggle Buttons
    rax = plt.axes([0.05, 0.45, 0.2, 0.25], frameon=False)
    check = CheckButtons(rax, labels, [True] * len(labels))

    def toggle_visibility(label):
        index = labels.index(label)
        scatters[index].set_visible(not scatters[index].get_visible())
        plt.draw()

    check.on_clicked(toggle_visibility)

    # Reset Button
    reset_ax = plt.axes([0.08, 0.35, 0.1, 0.05])
    btn_reset = Button(reset_ax, "Reset View", color="lightgray", hovercolor="skyblue")

    def reset_view(event):
        ax.view_init(elev=30, azim=-60)
        for i, status in enumerate(check.get_status()):
            if not status:
                check.set_active(i)
        plt.draw()

    btn_reset.on_clicked(reset_view)

    ax._persistent_widgets = [check, btn_reset]
    plt.show()


def visualize_in_meshcat(hand_name: str, cloud_data: dict[str, np.ndarray], logger):
    """Overlays point clouds onto the hand URDF in MeshCat."""
    config = get_config(hand_name)
    urdf_path = config["urdf_path"]
    mesh_dir = os.path.dirname(urdf_path)

    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        urdf_path, package_dirs=[mesh_dir]
    )
    viz = MeshcatVisualizer(model, collision_model, visual_model)

    try:
        viz.initViewer(open=True)
        viz.loadViewerModel("pinocchio")
        viz.display(pin.neutral(model))
    except Exception as e:
        logger.error(f"MeshCat initialization failed: {e}")
        return

    for link_name, points in cloud_data.items():
        color = get_color_for_link(link_name)
        num_points = points.shape[0]

        # Color formatting for MeshCat
        points_3xn = np.ascontiguousarray(points.T, dtype=np.float32)
        colors_3xn = np.tile(
            np.array(color, dtype=np.float32).reshape(3, 1), (1, num_points)
        )

        viz.viewer[f"workspace/{link_name}"].set_object(
            meshcat_geometry.PointCloud(
                position=points_3xn, color=colors_3xn, size=0.0015
            )
        )

    logger.info("MeshCat visualization active. Use browser to view.")


def main():
    logger = initialize_logger("workspace_sweep")
    args = parse_cli_args()

    # 1. Initialization
    logger.info(f"Initializing model for '{args.hand}'...")
    model, keypoint_links = setup_kinematic_model(args.hand)

    # 2. Sweep
    logger.info(f"Generating workspace cloud with {args.num_samples} samples...")
    cloud_data = generate_workspace_cloud(model, keypoint_links, args.num_samples)

    # 3. Export (Optional)
    if args.export:
        output_path = f"data/workspaces/{args.hand}_workspace.npz"
        os.makedirs("data/workspaces", exist_ok=True)
        np.savez_compressed(output_path, **cloud_data)
        logger.info(f"Data exported to {output_path}")

    # 4. Visualization
    if args.viz_3d:
        visualize_in_meshcat(args.hand, cloud_data, logger)

    # Always show Matplotlib as the primary analysis tool
    create_matplotlib_plot(args.hand, cloud_data)


if __name__ == "__main__":
    main()
