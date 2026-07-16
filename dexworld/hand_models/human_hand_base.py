from abc import abstractmethod

import numpy as np

from dexworld.types import Chirality
from .base_hand import BaseHandModel


class HumanHandModel(BaseHandModel):
    """Base class for human hand tracking models."""

    def __init__(self, chirality: Chirality = Chirality.RIGHT):
        super().__init__(chirality)

    @abstractmethod
    def to_joint_angles(self, joints_3d: np.ndarray) -> dict[str, np.ndarray]:
        """Extract internal joint angles from 3D keypoints."""
        pass

    @abstractmethod
    def to_kinematic_tree(
        self, joints_3d: np.ndarray, return_frame_dict: bool = False
    ) -> tuple[np.ndarray | dict, list]:
        """Compute the kinematic tree frames and links from 3D keypoints."""
        pass

    @abstractmethod
    def get_qpos_joint_names(self) -> list[str]:
        pass

    @abstractmethod
    def get_landmark_transforms(self, joints_3d: np.ndarray) -> dict:
        """Return 4x4 transforms for all configured landmarks."""
        pass
