"""Utility functions for MuJoCo."""

# Standard
from pathlib import Path
from os import PathLike

# Third-party
import numpy as np
import mujoco


_MJ_CONTEXT_CACHE: dict[str, tuple[mujoco.MjModel, mujoco.MjData]] = {}


def get_mj_context(
    mjcf_path: str | Path | PathLike,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Get (model, data) context for an MJCF path, cached by path."""
    key = str(mjcf_path)
    ctx = _MJ_CONTEXT_CACHE.get(key)
    if ctx is not None:
        return ctx
    model = mujoco.MjModel.from_xml_path(key)
    data = mujoco.MjData(model)
    _MJ_CONTEXT_CACHE[key] = (model, data)
    return model, data


def get_link_origin_from_mjcf_xyz_cached(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    name: str,
    obj_type: str = "joint",  # "joint" | "body" | "site"
    qpos: np.ndarray | None = None,
) -> np.ndarray:
    """Return link origin from a cached MuJoCo (model, data) context."""
    if qpos is None:
        data.qpos[:] = model.qpos0
    else:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.shape[0] != model.nq:
            raise ValueError(f"qpos must have length {model.nq}, got {qpos.shape[0]}")
        data.qpos[:] = qpos

    mujoco.mj_forward(model, data)

    if obj_type == "joint":
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint '{name}' not found in model")
        return data.xanchor[jid].copy()

    if obj_type == "body":
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' not found in model")
        return data.xpos[bid].copy()

    if obj_type == "site":
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise ValueError(f"Site '{name}' not found in model")
        return data.site_xpos[sid].copy()

    raise ValueError(f"Unsupported link type '{obj_type}'")


def get_actuated_joints_indices(model):
    # Get the names of all joints
    [model.joint(i).name for i in range(model.njnt)]

    # Get the names of the actuated joints
    actuated_joint_names = [
        model.joint(model.actuator(i).trnid[0]).name for i in range(model.nu)
    ]

    # Get the indices of the actuated joints
    return [model.joint(name).id for name in actuated_joint_names]


def get_actuated_joints_limits(model):
    # Get the joint limits for the actuated joints
    return model.jnt_range[get_actuated_joints_indices(model)]


def get_link_origin_from_mjcf_xyz(
    mjcf_path: str | Path | PathLike,
    name: str,
    link_type: str = "joint",  # "joint" | "body" | "site"
    qpos: np.ndarray | None = None,
) -> np.ndarray:
    """
    Returns the origin of a link in the MJCF model.

    Args:
        mjcf_path: The path to the MJCF model.
        name: The name of the link.
        link_type: The type of the link. Can be "joint", "body", or "site".
        qpos: The qpos of the model.

    Returns:
        The origin of the link in the MJCF model.

    Raises:
        ValueError: If the link is not found in the MJCF model.
    """
    model, data = get_mj_context(mjcf_path)
    return get_link_origin_from_mjcf_xyz_cached(model, data, name, link_type, qpos)
