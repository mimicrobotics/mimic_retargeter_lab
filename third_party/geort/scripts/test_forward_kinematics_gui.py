"""
Tests the forward kinematics of the robotic hand.

Features:
    - Loads GeoRT's HandKinematicModel for a given hand (e.g., p50 or allegro)
    - Opens a SAPIEN viewer showing the hand
    - Opens a Tkinter GUI with one slider per active joint (user-order)
    - Moving sliders updates the hand pose in real time
    - Fingertips are visualized as red spheres (from HandViewerEnv)

Author(s):
    - Robert Jomar Malate (robert.malate@mimicrobotics.com)
"""

import tkinter as tk
from typing import List

import numpy as np

from geort.env.hand import HandKinematicModel
from geort.utils.config_utils import get_config


PINCH_INDEX_QPOS_ANGLES = np.array(
    [
        52,
        -19,
        -17,
        -45,  # Thumb
        -8,
        70,
        25,  # Index
        0,
        0,
        0,  # Middle
        0,
        0,
        0,  # Ring
        0,
        0,
        0,  # Pinky
    ]
)

GC_ANGLE_OFFSETS = np.array(
    [
        30.0,  # thumb_cmc
        -20.0,  # thumb_mcp
        50.0,  # thumb_pp
        50.0,  # thumb_dp
        8.5,  # index_mcp_abd
        15.0,  # index_mcp_flex
        15.0,  # index_pip
        12.5,  # middle_mcp_abd
        15.0,  # middle_mcp_flex
        15.0,  # middle_pip
        17.5,  # ring_mcp_abd
        15.0,  # ring_mcp_flex
        15.0,  # ring_pip
        22.5,  # pinky_mcp_abd
        15.0,  # pinky_mcp_flex
        15.0,  # pinky_pip
    ]
)

PINCH_INDEX_QPOS_ANGLES_RAW = PINCH_INDEX_QPOS_ANGLES  # - GC_ANGLE_OFFSETS
PINCH_INDEX_ANGLES_RAD = np.deg2rad(PINCH_INDEX_QPOS_ANGLES_RAW)
print("PINCH_INDEX_QPOS_ANGLES:", PINCH_INDEX_QPOS_ANGLES)
print("PINCH_INDEX_QPOS_ANGLES_RAW:", PINCH_INDEX_QPOS_ANGLES_RAW)
print("PINCH_INDEX_ANGLES_RAD:", PINCH_INDEX_ANGLES_RAD)


def build_gui(
    root: tk.Tk,
    joint_names: List[str],
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    initial_qpos: np.ndarray,
):
    """
    Build a scrollable Tk GUI with one slider per joint.
    Returns the list of Scale widgets.
    """
    root.title("Hand Joint Controls")

    canvas = tk.Canvas(root)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    frame = tk.Frame(canvas)

    frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    scales = []

    for i, name in enumerate(joint_names):
        lo = float(joint_lower[i])
        hi = float(joint_upper[i])

        label = tk.Label(frame, text=f"{i}: {name}")
        label.pack(anchor="w", padx=5, pady=(5, 0))

        scale = tk.Scale(
            frame,
            from_=hi,
            to=lo,
            resolution=0.01,
            orient="horizontal",
            length=300,
        )
        scale.set(float(initial_qpos[i]))
        scale.pack(anchor="w", padx=10, pady=(0, 5))

        scales.append(scale)

    return scales


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hand",
        type=str,
        default="p50",
        help="Hand name as defined in GeoRT configs (e.g., 'p50' or 'allegro')",
    )
    args = parser.parse_args()

    # Load config and kinematic model
    config = get_config(args.hand)
    model = HandKinematicModel.build_from_config(config, render=True)
    viewer_env = model.get_viewer_env()

    # Joint info in *user* order
    n_dof = model.get_n_dof()
    joint_lower, joint_upper = model.get_joint_limit()
    joint_names = model.joint_names

    # Start at joint lower limits (or mid-range if you prefer)
    qpos_user = np.abs(PINCH_INDEX_ANGLES_RAD)
    print("Starting at abs val PINCH_INDEX_ANGLES_RAD pose:", qpos_user)
    # qpos_user = (joint_lower + joint_upper) / 2.0  # alternative

    print(f"Loaded hand '{args.hand}' with {n_dof} DOFs.")
    print("Use the GUI sliders to move each joint. Close the Tk window to exit.")

    for i, joint in enumerate(joint_names):
        print(
            f"  {i}: {joint} [joint_lower: {joint_lower[i]:.6f}, joint_upper: {joint_upper[i]:.6f}]"
        )

    # --- Build GUI in main thread ---
    root = tk.Tk()
    scales = build_gui(root, joint_names, joint_lower, joint_upper, qpos_user)

    # --- Update loop driven by Tk's timer ---
    def update():
        # Read slider values
        for i in range(n_dof):
            qpos_user[i] = float(scales[i].get())

        # Update hand pose
        model.set_qpos_target(qpos_user)
        # print(f"qpos_user: {qpos_user}")
        # print(f"qpos_sapien: {model.hand.get_qpos()}")

        # Step and render SAPIEN
        viewer_env.update()

        # Schedule next update (~100 FPS)
        root.after(10, update)

    # Start periodic updates and enter Tk mainloop
    update()
    root.mainloop()


if __name__ == "__main__":
    main()
