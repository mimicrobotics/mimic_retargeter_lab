import numpy as np
from scipy.spatial.transform import Rotation as R

LOCAL_JOINT_DOFS = [
    "wrist.x",
    "wrist.y",
    "wrist.z",
    # thumb
    # "th_mcp.x",
    # "th_mcp.z",
    "th_proximal.x",
    "th_proximal.y",
    "th_proximal.z",
    "th_distal.x",
    "th_distal.z",
    "th_tip.x",
    "th_tip.z",
    # first/index finger
    # "ff_mcp.x",
    # "ff_mcp.z",
    "ff_proximal.x",
    "ff_proximal.z",
    "ff_distal.x",
    "ff_distal.z",
    "ff_tip.x",
    # middle finger
    # "mf_mcp.x",
    # "mf_mcp.z",
    "mf_proximal.x",
    "mf_proximal.z",
    "mf_distal.x",
    "mf_tip.x",
    # ring finger
    # "rf_mcp.x",
    # "rf_mcp.z",
    "rf_proximal.x",
    "rf_proximal.z",
    "rf_distal.x",
    "rf_tip.x",
    # little finger
    # "lf_mcp.x",
    # "lf_mcp.z",
    "lf_proximal.x",
    "lf_proximal.z",
    "lf_distal.x",
    "lf_tip.x",
]


# joint orientation: x facing to the right, y is facing forward, z is facing upward
def extract_joint_angles(kin_tree) -> dict[str, dict[str, np.ndarray]]:
    joint_angles = {}
    for frame_name, transform in kin_tree.items():
        rot = transform[:, :3, :3]
        euler_rot = R.from_matrix(rot).as_euler("xyz", degrees=False)
        x_component = euler_rot[:, 0]
        y_component = euler_rot[:, 1]
        z_component = euler_rot[:, 2]
        rots = {
            "x": x_component,
            "y": y_component,
            "z": z_component,
        }
        for dof_char in ["x", "y", "z"]:
            if (local_dof := f"{frame_name}.{dof_char}") in LOCAL_JOINT_DOFS:
                joint_angles[local_dof] = rots[dof_char]

    return joint_angles
