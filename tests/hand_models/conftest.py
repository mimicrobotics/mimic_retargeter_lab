import numpy as np
import pytest
from pathlib import Path
from dexworld.types import Chirality

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES_PATH = REPO_ROOT / "tests" / "fixtures"


class GoldenData:
    """Accessor for golden .npz fixture data, keyed by pose label."""

    def __init__(self, npz_path: Path):
        self._npz = np.load(npz_path)
        labels = list(self._npz["pose_labels"])
        self._map = {label: i for i, label in enumerate(labels)}

    def idx(self, label: str) -> int:
        return self._map[label]

    def ctrl(self, label: str) -> np.ndarray:
        return self._npz["ctrl_poses"][self._map[label]]

    def qpos(self, label: str) -> np.ndarray:
        return self._npz["qpos_poses"][self._map[label]]

    def fingertips(self, label: str) -> np.ndarray:
        return self._npz["fingertips"][self._map[label]]

    def keyvectors(self, label: str) -> np.ndarray:
        return self._npz["keyvectors"][self._map[label]]

    def frame_poses(self, label: str) -> np.ndarray:
        return self._npz["frame_poses"][self._map[label]]

    def fingertip_jacobians(self, label: str) -> np.ndarray:
        return self._npz["fingertip_jacobians"][self._map[label]]

    @property
    def joint_map(self) -> np.ndarray:
        return self._npz["joint_map"]

    @property
    def qpos_joint_names(self) -> list[str]:
        if "qpos_joint_names" not in self._npz:
            return []
        return list(self._npz["qpos_joint_names"])

    @property
    def actuated_joint_names(self) -> list[str]:
        if "actuated_joint_names" not in self._npz:
            return []
        return list(self._npz["actuated_joint_names"])

    @property
    def fingertip_landmarks(self) -> list[str]:
        if "fingertip_landmarks" not in self._npz:
            return []
        return list(self._npz["fingertip_landmarks"])

    @property
    def labels(self) -> list[str]:
        return list(self._map.keys())

    @property
    def random_labels(self) -> list[str]:
        return [label for label in self._map if label.startswith("random_")]


@pytest.fixture
def load_golden():
    """Factory fixture: returns a callable that loads a golden .npz by filename.

    Usage in test files:
        @pytest.fixture
        def golden(load_golden):
            return load_golden("shadow_hand_right_robot_hand_data_golden.npz")
    """

    def _load(filename: str) -> GoldenData:
        return GoldenData(FIXTURES_PATH / filename)

    return _load


@pytest.fixture
def assets_path():
    return REPO_ROOT / "assets" / "mjcf"


@pytest.fixture
def shadow_hand_path(assets_path):
    return assets_path / "shadow_hand"


@pytest.fixture
def mimic_hand_path(assets_path):
    return assets_path / "mimic_p050_hand"


@pytest.fixture
def allegro_hand_path(assets_path):
    return assets_path / "wonik_allegro_hand"


@pytest.fixture
def shadow_dexee_hand_path(assets_path):
    return assets_path / "shadow_dexee_hand"


@pytest.fixture
def leap_hand_path(assets_path):
    return assets_path / "leap_hand"


@pytest.fixture
def wuji_hand_path(assets_path):
    return assets_path / "wuji_hand"

@pytest.fixture
def orca_v2_hand_path(assets_path):
    return assets_path / "orca_v2_hand"


@pytest.fixture
def right():
    return Chirality.RIGHT


@pytest.fixture
def left():
    return Chirality.LEFT
