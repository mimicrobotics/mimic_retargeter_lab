"""Visualize point-cloud joint data collected from a hand tracker.

Visualizes data collected from a hand tracker. It automatically detects
the hand tracker type, chirality, and metadata based on the dataset filename.

Note: 
- Most of this code has been generated using GenAI tools like ChatGPT or Gemini. 
Please review and test thoroughly before using in production.

Usage Examples
--------------
# 1. Show a single frame
python visualize_hand_tracker_data.py \
    --mode frame \
    --frame_idx 0 \
    --dataset_filename dataset_manus-test-001_subject-RJM.npy

# 2. Interactive viewer
python visualize_hand_tracker_data.py \
    --mode interactive \
    --dataset_filename dataset_manus-test-001_subject-RJM.npy

# 3. Export animation to MP4
python visualize_hand_tracker_data.py \
    --mode animate \
    --out media/hand_motion.mp4 \
    --dataset_filename dataset_manus-test-001_subject-RJM.npy

Authors
    - Robert Jomar Malate (robert.malate@mimicrobotics.com)
"""

# Standard
import argparse
from pathlib import Path
import logging

# Third-party
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# Custom
from utils.common import HandTracker, FINGER_GROUPS
from utils.helpers import (
    initialize_logger,
)
from utils.dataset import (
    extract_hand_tracker_from_filename,
    extract_hand_chirality_from_filename,
    extract_subject_id_from_filename,
    extract_run_id_from_filename,
)
import utils.argparse_utils as argparse_utils


class HandTrackerVisualizer:
    """
    Visualize MANUS/MANO-style 3D joint data.

    - Data shape: [T, 25, 3] (Positions only) OR [T, 25, 7] (Pos + Quat)
    """

    def __init__(self, data_path: Path, hand_tracker: HandTracker, joint_names=None):
        self.data_path = data_path
        self.raw_data = np.load(self.data_path)
        filename_str = self.data_path.name

        # -----------------------------------------------------
        # 1. Extract Metadata
        # -----------------------------------------------------
        self.tracker_name = extract_hand_tracker_from_filename(filename_str)
        self.subject_id = extract_subject_id_from_filename(filename_str) or "N/A"
        self.run_id = extract_run_id_from_filename(filename_str) or "N/A"

        # Extract Chirality for Camera Setup
        chirality_str = extract_hand_chirality_from_filename(filename_str)

        # -----------------------------------------------------
        # 2. Configure View based on Chirality
        # -----------------------------------------------------
        if chirality_str and "left" in chirality_str.lower():
            self.chirality = "left"
            # Mirror view for left hand
            self.default_view = {"elev": 30, "azim": 60}
        else:
            self.chirality = "right"  # Default to right if unknown
            # Standard view for right hand
            self.default_view = {"elev": 30, "azim": -60}

        # -----------------------------------------------------
        # 3. Data Validation & Slicing
        # -----------------------------------------------------
        if self.raw_data.ndim != 3 or self.raw_data.shape[1] != 25:
            raise ValueError(
                f"Expected data of shape [T, 25, D], got {self.raw_data.shape}"
            )

        dims = self.raw_data.shape[2]
        if dims not in [3, 7]:
            raise ValueError(
                f"Expected last dimension to be 3 (pos) or 7 (pos+quat), got {dims}"
            )

        self.pos_data = self.raw_data[:, :, :3]
        self.num_frames, self.num_joints, _ = self.pos_data.shape

        # Handle Orientations (last 4 columns)
        self.has_orientation = dims == 7
        self.quat_data = self.raw_data[:, :, 3:] if self.has_orientation else None

        # -----------------------------------------------------
        # 4. Skeleton Setup
        # -----------------------------------------------------
        if joint_names is None:
            self.joint_names = [str(i) for i in range(self.num_joints)]
        else:
            self.joint_names = joint_names

        self.finger_groups = FINGER_GROUPS[hand_tracker]
        self.finger_colors = {
            "wrist": "black",
            "thumb": "tab:red",
            "index": "tab:green",
            "middle": "tab:blue",
            "ring": "tab:purple",
            "pinky": "tab:orange",
        }
        self.edges = self._build_edges()

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------
    def _build_edges(self):
        """Build skeleton edges from finger groups."""
        edges = []
        wrist_idx = self.finger_groups["wrist"][0]
        for finger_name, indices in self.finger_groups.items():
            if finger_name == "wrist" or not indices:
                continue
            edges.append((wrist_idx, indices[0]))
            for i in range(len(indices) - 1):
                edges.append((indices[i], indices[i + 1]))
        return edges

    @staticmethod
    def _set_equal_axes(ax, pts):
        """Make x/y/z axes have equal scale."""
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        centers = (mins + maxs) / 2.0
        span = (maxs - mins).max() / 2.0
        ax.set_xlim(centers[0] - span, centers[0] + span)
        ax.set_ylim(centers[1] - span, centers[1] + span)
        ax.set_zlim(centers[2] - span, centers[2] + span)

    def _draw_coordinate_frame(self, ax, origin=(0, 0, 0), length=0.05):
        ox, oy, oz = origin
        ax.plot([ox, ox + length], [oy, oy], [oz, oz], c="r", lw=2, label="_nolegend_")
        ax.plot([ox, ox], [oy, oy + length], [oz, oz], c="g", lw=2, label="_nolegend_")
        ax.plot([ox, ox], [oy, oy], [oz, oz + length], c="b", lw=2, label="_nolegend_")

    def _color_for_joint(self, joint_idx):
        for finger_name, indices in self.finger_groups.items():
            if joint_idx in indices:
                return self.finger_colors.get(finger_name, "gray")
        return "gray"

    def _draw_joint_orientations(self, ax, positions, quaternions, scale=0.015):
        """Renders local XYZ basis vectors for each joint."""
        for i in range(self.num_joints):
            origin = positions[i]
            q = quaternions[i]
            if np.all(q == 0):
                continue

            rot = R.from_quat(q)
            matrix = rot.as_matrix()

            # X(Red), Y(Green), Z(Blue)
            ax.quiver(*origin, *(matrix[:, 0] * scale), color="r", lw=1.5, alpha=0.8)
            ax.quiver(*origin, *(matrix[:, 1] * scale), color="g", lw=1.5, alpha=0.8)
            ax.quiver(*origin, *(matrix[:, 2] * scale), color="b", lw=1.5, alpha=0.8)

    # ---------------------------------------------------------
    # Core Plotting Logic (The "Engine")
    # ---------------------------------------------------------
    def _draw_scene(self, ax, pos_frame, quat_frame=None, draw_labels=False):
        """Draws the joints, edges, and arrows on the provided axes."""
        self._draw_coordinate_frame(ax)

        # 1. Joints
        for finger_name, indices in self.finger_groups.items():
            color = self.finger_colors.get(finger_name, "gray")
            pts = pos_frame[indices]
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2], label=finger_name, s=20, color=color
            )

        # 2. Edges
        for i, j in self.edges:
            color = self._color_for_joint(i)
            ax.plot(
                [pos_frame[i, 0], pos_frame[j, 0]],
                [pos_frame[i, 1], pos_frame[j, 1]],
                [pos_frame[i, 2], pos_frame[j, 2]],
                color=color,
            )

        # 3. Orientations
        if self.has_orientation and quat_frame is not None:
            self._draw_joint_orientations(ax, pos_frame, quat_frame)

        # 4. Labels
        if draw_labels:
            # Manual offsets for cleaner text
            offset_map = {
                0: (-0.015, -0.01, -0.01),
                1: (0.01, 0.01, 0.01),
                21: (0.015, 0.005, 0.008),
                6: (0, 0, 0),
                16: (0.007, 0.004, 0.006),
                11: (0.005, 0.003, 0.007),
            }
            for idx, (x, y, z) in enumerate(pos_frame):
                dx, dy, dz = offset_map.get(idx, (0, 0, 0))
                ax.text(
                    x + dx,
                    y + dy,
                    z + dz,
                    self.joint_names[idx],
                    fontsize=9,
                    color="black",
                )

    def _setup_axes_labels_and_view(self, ax, pos_frame, frame_idx):
        """Sets the title with metadata and orients the 3D camera."""
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Build Title with Metadata
        title_text = (
            f"Hand Tracker: {self.tracker_name} | {self.chirality.title()} | "
            f"Subject ID: {self.subject_id} | Run ID: {self.run_id}\n"
            f"Frame: {frame_idx} / {self.num_frames - 1}"
        )
        ax.set_title(title_text, fontsize=10)

        self._set_equal_axes(ax, pos_frame)

        # Apply Camera View (Dynamic based on Chirality)
        ax.view_init(elev=self.default_view["elev"], azim=self.default_view["azim"])

        ax.legend(loc="upper right", fontsize=8)

    # ---------------------------------------------------------
    # Public interfaces
    # ---------------------------------------------------------
    def show_single_frame(self, frame_idx: int = 0):
        frame_idx = max(0, min(frame_idx, self.num_frames - 1))
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Get data
        pos = self.pos_data[frame_idx]
        quat = self.quat_data[frame_idx] if self.has_orientation else None

        # Draw & Configure
        self._draw_scene(ax, pos, quat, draw_labels=True)
        self._setup_axes_labels_and_view(ax, pos, frame_idx)

        plt.tight_layout()
        plt.show()

    def interactive_view(self):
        idx = 0
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        def update_plot():
            ax.clear()
            pos = self.pos_data[idx]
            quat = self.quat_data[idx] if self.has_orientation else None

            self._draw_scene(ax, pos, quat, draw_labels=True)
            self._setup_axes_labels_and_view(ax, pos, idx)
            fig.canvas.draw_idle()

        # Initial plot
        update_plot()

        def on_key(event):
            nonlocal idx
            if event.key in ["right", "d"]:
                idx = (idx + 1) % self.num_frames
            elif event.key in ["left", "a"]:
                idx = (idx - 1) % self.num_frames
            elif event.key == "q":
                plt.close(fig)
                return
            update_plot()

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.tight_layout()
        plt.show()

    def animate(
        self, out_path: Path, fps: int = 20, step: int = 1, max_frames: int | None = 300
    ):
        # Downsample data
        pos_frames = self.pos_data[::step]
        quat_frames = self.quat_data[::step] if self.has_orientation else None

        if max_frames is not None:
            pos_frames = pos_frames[:max_frames]
            if quat_frames is not None:
                quat_frames = quat_frames[:max_frames]

        # Use all points to keep axis scale consistent
        all_pts = pos_frames.reshape(-1, 3)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        def init():
            ax.clear()
            self._setup_axes_labels_and_view(ax, all_pts, 0)
            return []

        def update(i):
            ax.clear()
            p = pos_frames[i]
            q = quat_frames[i] if quat_frames is not None else None

            # Calculate REAL frame index for the title
            real_frame_idx = i * step

            # Draw scene (no labels for speed)
            self._draw_scene(ax, p, q, draw_labels=False)

            # Update title and camera view
            self._setup_axes_labels_and_view(ax, all_pts, real_frame_idx)
            return []

        init()
        out_path = out_path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        Writer = (
            FFMpegWriter
            if out_path.suffix.lower() != ".gif"
            else __import__(
                "matplotlib.animation", fromlist=["PillowWriter"]
            ).PillowWriter
        )
        writer = Writer(fps=fps)

        print(f"[Saving animation → {out_path}]")
        with writer.saving(fig, str(out_path), dpi=100):
            for i in tqdm(range(len(pos_frames)), desc="Rendering", unit="frame"):
                update(i)
                writer.grab_frame()
        plt.close(fig)
        print(f"[✓] Animation saved to: {out_path}")


# -------------------------------------------------------------
# HELPER FUNCTIONS & CLI
# -------------------------------------------------------------
def parse_cli_args() -> argparse.Namespace:
    parser = argparse_utils.create_argparser(
        description="Visualize hand tracker joint data."
    )
    argparse_utils.add_args(parser, "dataset_filename")
    parser.add_argument(
        "--mode", choices=["frame", "interactive", "animate"], default="interactive"
    )
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--output_filename", type=str, default="hand_motion.mp4")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=500)
    return parser.parse_args()


def main():
    logger = initialize_logger(module_name=__name__, default_log_level=logging.INFO)
    args = parse_cli_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    data_path = (data_dir / args.dataset_filename).resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"Loading data from: {data_path}")
    hand_tracker = HandTracker(
        extract_hand_tracker_from_filename(args.dataset_filename)
    )

    visualizer = HandTrackerVisualizer(data_path, hand_tracker)

    if args.mode == "frame":
        visualizer.show_single_frame(args.frame_idx)
    elif args.mode == "interactive":
        visualizer.interactive_view()
    elif args.mode == "animate":
        media_dir = (script_dir.parent / "media").resolve()
        media_dir.mkdir(parents=True, exist_ok=True)
        visualizer.animate(
            media_dir / Path(args.output_filename).name,
            args.fps,
            args.step,
            args.max_frames,
        )


if __name__ == "__main__":
    main()
