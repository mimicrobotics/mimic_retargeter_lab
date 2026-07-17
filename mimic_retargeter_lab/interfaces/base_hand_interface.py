from abc import ABC, abstractmethod

import numpy as np


class BaseHandInterface(ABC):
    def __init__(self, initial_joint_angles_command):
        self.joint_angles = initial_joint_angles_command
        self.commanded_joint_angles = initial_joint_angles_command

    @abstractmethod
    def get_joint_angles(self):
        pass

    @abstractmethod
    def set_joint_angles(self, joint_angles):
        pass

    def set_hand_transform(
        self, hand_transform: np.ndarray, tgt_key: str = "rh_forearm"
    ):
        """Set the hand base (wrist/forearm) transform. No-op by default."""
        pass
