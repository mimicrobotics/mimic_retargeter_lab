"""Unit tests for JointAngleRetargeter with fake hand models."""

from __future__ import annotations

import numpy as np
import pytest

from dexworld.retargeting.online import base_online_retargeter as bor_mod
from dexworld.retargeting.online.joint_angle_retargeter import JointAngleRetargeter


class FakeFromModel:
    """Fake from_model that returns canned joint angles."""

    def __init__(self, angles: dict[str, np.ndarray]):
        self._angles = angles

    def to_joint_angles(self, joints_3d: np.ndarray) -> dict[str, np.ndarray]:
        return self._angles


class FakeToModel:
    """Fake to_model that returns a fixed joint angles."""

    def __init__(self) -> None:
        self.num_qpos_dofs = 3
        self.num_actuated_dofs = 3
        self.joint_map = np.eye(3, dtype=np.float32)

    def get_qpos_joint_names(self) -> list[str]:
        return ["joint_a", "joint_b", "joint_c"]

    def get_actuated_joint_names(self) -> list[str]:
        return ["actuator_a", "actuator_b", "actuator_c"]

    def get_actuated_joint_limits(self) -> dict[str, tuple[float, float]]:
        return {
            "actuator_a": (-1.0, 1.0),
            "actuator_b": (-1.0, 1.0),
            "actuator_c": (-1.0, 1.0),
        }

    def get_neutral_ctrl_pose(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)


class FakeWristRetargeter:
    """Fake wrist retargeter that returns a fixed wrist transform."""

    def retarget(self, wrist_transform: np.ndarray) -> np.ndarray:
        return wrist_transform + 1.0


@pytest.fixture
def wrist_mapping() -> dict[str, object]:
    return {
        "tgt_key": "thumb_cmc",
        "root_key": "thumb_cmc",
        "transform": np.eye(4, dtype=np.float32),
    }


def test_mapping_coef_offset_and_constant_joints(monkeypatch, wrist_mapping) -> None:
    """Test that the retargeter applies the mapping coef, offset, and constant joints."""
    monkeypatch.setattr(
        bor_mod, "WristRetargeter", lambda *_a, **_k: FakeWristRetargeter()
    )

    angles = {
        "src_a": np.float32(1.0),
        "src_b": np.float32(2.0),
    }
    r = JointAngleRetargeter(
        from_model=FakeFromModel(angles),
        to_model=FakeToModel(),
        joint_mapping=[
            {"tgt_key": "joint_a", "src_key": "src_a", "coef": 2.0, "offset": 0.5},
            {"tgt_key": "joint_b", "src_key": "src_b", "coef": -1.0, "offset": 0.0},
        ],
        wrist_mapping=wrist_mapping,
        constant_joints={"joint_c": 7.0},
    )

    joints = np.zeros((21, 3), dtype=np.float32)
    out, wrist = r.retarget(joints)

    assert out.shape == (1, 3)
    np.testing.assert_allclose(out[0, 0], 2.5, atol=1e-6)
    np.testing.assert_allclose(out[0, 1], -2.0, atol=1e-6)
    np.testing.assert_allclose(out[0, 2], 7.0, atol=1e-6)
    assert wrist is None


def test_projection_with_non_identity_joint_map(monkeypatch, wrist_mapping) -> None:
    """Test that the retargeter projects the joint angles correctly."""
    monkeypatch.setattr(
        bor_mod, "WristRetargeter", lambda *_a, **_k: FakeWristRetargeter()
    )

    to_model = FakeToModel()
    to_model.joint_map = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]],
        dtype=np.float32,
    )
    angles = {"s": np.float32(1.0)}
    r = JointAngleRetargeter(
        from_model=FakeFromModel(angles),
        to_model=to_model,
        joint_mapping=[
            {"tgt_key": "joint_c", "src_key": "s", "coef": 1.0, "offset": 0.0}
        ],
        wrist_mapping=wrist_mapping,
        constant_joints={},
    )
    out, _ = r.retarget(np.zeros((21, 3), dtype=np.float32))
    expected_act = np.array([[0.0, 0.0, 0.5]], dtype=np.float32)
    np.testing.assert_allclose(out, expected_act, atol=1e-5)


def test_wrist_transform_branch(monkeypatch, wrist_mapping) -> None:
    """Test that the retargeter applies the wrist transform correctly."""
    monkeypatch.setattr(
        bor_mod, "WristRetargeter", lambda *_a, **_k: FakeWristRetargeter()
    )

    r = JointAngleRetargeter(
        from_model=FakeFromModel({}),
        to_model=FakeToModel(),
        joint_mapping=[],
        wrist_mapping=wrist_mapping,
        constant_joints={},
    )
    wrist_in = np.zeros((4, 4), dtype=np.float32)
    _, wrist_out = r.retarget(
        np.zeros((21, 3), dtype=np.float32), wrist_transform=wrist_in.copy()
    )
    assert wrist_out is not None
    np.testing.assert_allclose(wrist_out, wrist_in + 1.0, atol=1e-6)
