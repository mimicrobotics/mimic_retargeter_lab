from abc import ABC, abstractmethod


class BaseWristDataSource(ABC):
    """Base class for devices that produce SE3 wrist pose commands (e.g., SpaceMouse)."""

    @abstractmethod
    def get_iter(self):
        """Yield dicts with:
        - 'wrist_pose': np.ndarray (4, 4) -- current absolute SE3 wrist pose
        - 'twist': np.ndarray (6,) -- raw linear (x,y,z) + angular (rx,ry,rz) velocities
        - 'buttons': dict -- button states
        """
        pass

    @abstractmethod
    def reset_pose(self, pose=None):
        """Reset the integrated wrist pose to identity or a given SE3 matrix."""
        pass
