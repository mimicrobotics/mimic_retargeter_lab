from abc import ABC
from pathlib import Path


class BaseObject(ABC):
    def __init__(self, name: str, id: int, mesh_path: Path):
        self.name = name
        self.id = id
        self.mesh_path = mesh_path
        self.initial_pose = None

    def set_initial_pose(self, pose):
        self.initial_pose = pose
