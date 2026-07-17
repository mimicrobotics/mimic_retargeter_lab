"""MJCF-backed kinematic model for GeoRT training.

Duck-types `geort.env.hand_min.HandKinematicModel` against mimic_retargeter_lab's
`RobotHandModel` + MuJoCo MJCF, so GeoRT's trainer can train MJCF-defined
hands without any URDF in the loop. The trainer touches exactly four
methods on this class (mirrored signatures):

    build_from_config(config)   — construct from a config dict
    get_joint_limit()           — (lower, upper) in joint_order
    initialize_keypoint(...)    — register fingertip queries
    keypoint_from_qpos(...)     — FK to fingertip positions

`joint_order` is interpreted as a list of *actuated* joint names. The
adapter expands user qpos through `to_model.joint_map` so any MJCF
coupling (tendons, equality constraints) is honored exactly.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from mimic_retargeter_lab.hand_models import RobotHandModel
from mimic_retargeter_lab.utils.mj_utils import get_mj_context


class MjcfHandKinematicModel:
    def __init__(
        self,
        to_model: RobotHandModel,
        joint_order: list[str],
    ):
        self.to_model = to_model
        self.joint_order = list(joint_order)

        # Accept both the actuated name (e.g. "A_thumb_base2cmc") and its
        # unprefixed qpos equivalent ("thumb_base2cmc"). GeoRT's existing
        # configs and our yamls list the unprefixed form; the MJCF on this
        # hand uses the prefixed form for actuators.
        actuated_names = list(to_model.get_actuated_joint_names())
        name_to_actuated_idx: dict[str, int] = {}
        for i, a_name in enumerate(actuated_names):
            name_to_actuated_idx[a_name] = i
            qpos_equiv = to_model._joint_name_from_actuated_name(a_name)
            if qpos_equiv != a_name:
                name_to_actuated_idx[qpos_equiv] = i

        unknown = [n for n in self.joint_order if n not in name_to_actuated_idx]
        if unknown:
            raise ValueError(
                f"joint_order names not in to_model.get_actuated_joint_names() "
                f"(or their unprefixed qpos equivalents): {unknown}. "
                f"Known names: {sorted(name_to_actuated_idx)}"
            )
        self._actuated_reorder = np.array(
            [name_to_actuated_idx[n] for n in self.joint_order], dtype=np.int32
        )

        # Joint limits: look up by actuated name (the dict keys are actuated).
        actuated_limits = to_model.get_actuated_joint_limits()
        actuated_for_order = [actuated_names[i] for i in self._actuated_reorder]
        self._joint_lower = np.array(
            [actuated_limits[n][0] for n in actuated_for_order], dtype=np.float32
        )
        self._joint_upper = np.array(
            [actuated_limits[n][1] for n in actuated_for_order], dtype=np.float32
        )

        self._mj_model, self._mj_data = get_mj_context(to_model.get_model_path())

        # Cache qpos addresses for the hand's qpos joints once.
        _, self._qpos_adrs = RobotHandModel._mj_qpos_address_per_joint(
            self._mj_model, list(to_model.get_qpos_joint_names())
        )

        self._joint_map = np.asarray(to_model.joint_map, dtype=np.float32)

        self._keypoint_link_names: list[str] = []
        self._keypoint_offsets: list[np.ndarray] = []
        self._keypoint_resolved: list[tuple[str, int]] = []

    @classmethod
    def build_from_config(cls, config: dict, **kwargs) -> "MjcfHandKinematicModel":
        """Instantiate from a GeoRT-shaped config dict.

        `to_model` must be passed as a kwarg (or, for legacy callers, embedded
        in `config`). It is kept out of `config` by default so the trainer's
        `save_json(self.config, ...)` step doesn't trip over a non-serializable
        `RobotHandModel`.

        Required:
          - to_model: RobotHandModel instance — kwarg or `config["to_model"]`.
          - config["joint_order"]: list of actuated joint names.
        """
        to_model = kwargs.get("to_model") or config.get("to_model")
        if to_model is None:
            raise ValueError(
                "MjcfHandKinematicModel.build_from_config requires `to_model` to be "
                "passed as a kwarg or set in `config`."
            )
        return cls(to_model=to_model, joint_order=config["joint_order"])

    def get_joint_limit(self) -> tuple[np.ndarray, np.ndarray]:
        return self._joint_lower.copy(), self._joint_upper.copy()

    def initialize_keypoint(
        self, keypoint_link_names: list[str], keypoint_offsets: list[Any]
    ) -> None:
        resolved: list[tuple[str, int]] = []
        for name in keypoint_link_names:
            body_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                resolved.append(("body", int(body_id)))
                continue
            site_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id >= 0:
                resolved.append(("site", int(site_id)))
                continue
            raise ValueError(
                f"Keypoint link {name!r} not found as body or site in MJCF "
                f"({self.to_model.get_model_path()})."
            )
        self._keypoint_link_names = list(keypoint_link_names)
        self._keypoint_offsets = [
            np.asarray(o, dtype=np.float32) for o in keypoint_offsets
        ]
        self._keypoint_resolved = resolved

    def keypoint_from_qpos(
        self,
        qpos_user: np.ndarray,
        ret_vec: bool = False,
        ret_orientation: bool = False,
    ):
        qpos_user = np.ascontiguousarray(qpos_user, dtype=np.float32)
        if qpos_user.shape != (len(self.joint_order),):
            raise ValueError(
                f"qpos_user shape {qpos_user.shape} != ({len(self.joint_order)},)"
            )

        ctrl = np.zeros(self.to_model.num_actuated_dofs, dtype=np.float32)
        ctrl[self._actuated_reorder] = qpos_user
        full_qpos = ctrl @ self._joint_map.T

        self._mj_data.qpos[self._qpos_adrs] = full_qpos
        mujoco.mj_forward(self._mj_model, self._mj_data)

        result: dict[str, Any] = {}
        vec_result: list[np.ndarray] = []
        for name, offset, (kind, oid) in zip(
            self._keypoint_link_names, self._keypoint_offsets, self._keypoint_resolved
        ):
            if kind == "body":
                pos = np.asarray(self._mj_data.xpos[oid], dtype=np.float32)
                rot = np.asarray(self._mj_data.xmat[oid], dtype=np.float32).reshape(
                    3, 3
                )
            else:
                pos = np.asarray(self._mj_data.site_xpos[oid], dtype=np.float32)
                rot = np.asarray(
                    self._mj_data.site_xmat[oid], dtype=np.float32
                ).reshape(3, 3)
            pos_world = pos + rot @ offset
            vec_result.append(pos_world)

            if ret_orientation:
                quat = _rot_matrix_to_quat_xyzw(rot)
                result[name] = (pos_world, quat)
                vec_result.append(quat)
            else:
                result[name] = pos_world

        if ret_vec:
            return np.array(vec_result, dtype=np.float32)
        return result


def _rot_matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert a (3, 3) rotation matrix to (qx, qy, qz, qw) — scipy convention."""
    R = np.asarray(R, dtype=np.float32)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz, qw], dtype=np.float32)
