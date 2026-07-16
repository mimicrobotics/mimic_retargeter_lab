import types
from typing import Any, Dict, List

import numpy as np

from dexworld.hand_models import HumanHandModel, RobotHandModel

from .base_online_retargeter import BaseOnlineRetargeter


class JointAngleRetargeter(BaseOnlineRetargeter):
    """
    Minimal joint-angle retargeter.

    Expectations:
      - `joint_mapping` is a LIST of mapping entries. Each entry is a dict
        with keys:
          - `name` (str): human readable entry name (recommended)
          - `tgt_key` (str): target (robot) qpos joint name
          - `src_key` (str): source (human) joint name (as returned by
                             from_model.to_joint_angles())
          - `coef` (float, optional): multiplicative coefficient (default 1.0)
          - `offset` (float, optional): additive offset (default 0.0)

    This implementation keeps logic minimal and assumes the config is well-formed.
    """

    def __init__(
        self,
        from_model: HumanHandModel,
        to_model: RobotHandModel,
        joint_mapping: List[Dict[str, Any]],
        wrist_mapping: List[Dict[str, Any]],
        constant_joints: Dict[str, float] | None = None,
        device: str = "cpu",
    ):
        super().__init__(from_model, to_model, wrist_mapping)
        self.constant_joints = constant_joints or {}

        # Pure-numpy retargeter — runs on CPU regardless. `_device` is exposed
        # so the Latency metric can label timings (utils/retarget_utils.py:184).
        self._device = types.SimpleNamespace(platform=device)

        self._mapping: Dict[str, Dict[str, Any]] = {}
        for entry in joint_mapping:
            tgt = entry["tgt_key"]
            src = entry["src_key"]
            self._mapping[tgt] = {
                "source": src,
                "coef": float(entry.get("coef", 1.0)),
                "offset": float(entry.get("offset", 0.0)),
                "name": entry.get("name"),
            }

        self.inverse_joint_map = np.linalg.pinv(
            np.asarray(self.to_model.joint_map, dtype=np.float32)
        )

    def retarget(
        self, joints: np.ndarray, wrist_transform: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Retarget input `joints` (batch) from from_model into target model actuated joints.

        Args:
          joints: array with shape (batch_size, ...). Passed to
                  from_model.from_joints(...).
          wrist_transform: transform from the wrist frame of the source hand to the world frame.

        Returns:
          np.ndarray of shape (batch_size, num_actuated)
        """
        joints = np.asarray(joints, dtype=np.float32)

        # to_joint_angles handles (J, 3) directly -- no batch needed
        joint_angles = self.from_model.to_joint_angles(joints)

        out_qpos = np.zeros(self.to_model.num_qpos_dofs, dtype=np.float32)
        qpos_joint_names = self.to_model.get_qpos_joint_names()

        for tgt_name, cfg in self._mapping.items():
            val = np.asarray(joint_angles[cfg["source"]], dtype=np.float32).squeeze()
            out_qpos[qpos_joint_names.index(tgt_name)] = (
                cfg["coef"] * val + cfg["offset"]
            )

        for tgt_name, const_val in self.constant_joints.items():
            out_qpos[qpos_joint_names.index(tgt_name)] = float(const_val)

        out_actuated = out_qpos @ self.inverse_joint_map.T

        tgt_wrist_transform = None
        if wrist_transform is not None:
            tgt_wrist_transform = self.wrist_retargeter.retarget(wrist_transform)

        return out_actuated[None, :], tgt_wrist_transform
