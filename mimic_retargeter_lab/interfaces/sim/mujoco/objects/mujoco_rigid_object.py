from pathlib import Path

from mimic_retargeter_lab.interfaces.objects.base_object import BaseObject

from .mujoco_base_object import MujocoBaseObject


class MujocoRigidObject(MujocoBaseObject):
    """
    Simple representation of a rigid object that can produce a small MJCF/XML
    snippet describing the object's mesh asset and a mocap/body entry for
    visualization in MuJoCo.

    Notes:
      - The returned XML is a fragment that contains an <asset> entry (mesh)
        and a <body> entry that references the mesh by name. It is intentionally
        small so it can be inserted into a larger MJCF string or parsed
        individually by code which collects assets and worldbody entries.
      - The geom is marked contype="0" conaffinity="0" so it does not participate
        in collisions by default (useful for purely visual replay).
    """

    @classmethod
    def from_base_object(cls, base_object: BaseObject):
        return cls(
            base_object.name,
            base_object.id,
            base_object.mesh_path,
            base_object.initial_pose,
        )

    def __init__(self, name: str, id: int, mesh_path: Path, initial_pose):
        super().__init__()
        self.name = name
        self.id = id
        self.mesh_path = mesh_path
        self.initial_pose = initial_pose

    def get_body_name(self) -> str:
        """Return a deterministic body name for this object."""
        # safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in self.name)
        # return f"obj_{self.id}_{safe_name}"
        return self.name

    def get_mesh_asset_name(self) -> str:
        """Return a deterministic mesh asset name for this object."""
        return f"{self.get_body_name()}_mesh"

    def get_xml(self) -> str:
        """
        Return a small MJCF/XML fragment that declares the mesh asset and a mocap
        body containing a visual geom referencing that mesh.

        The fragment contains:
          - <asset><mesh .../></asset>
          - <body name="..." mocap="true"> with a mesh geom referencing the asset

        The mesh path is emitted as an absolute/relative path exactly as provided
        in `mesh_path.as_posix()`. Consumers should ensure the path is valid from
        MuJoCo's working directory or adjust the path beforehand.

        Returns:
            str: MJCF/XML fragment
        """
        mesh_file = Path(self.mesh_path).as_posix()
        body_name = self.get_body_name()
        mesh_name = self.get_mesh_asset_name()

        # A small, self-contained fragment. Consumers can stitch multiple such
        # fragments (assets/body entries) into a full <mujoco> model.
        xml = f"""
        <asset>
          <mesh name="{mesh_name}" file="{mesh_file}"/>
        </asset>

        <!-- Visual body for object '{self.name}' (id={self.id}) -->
        <body name="{body_name}" mocap="true" pos="0 0 0" quat="0 0 0 1">
          <!-- visual mesh geom: non-colliding by default to serve as visualization -->
          <geom type="mesh" mesh="{mesh_name}" contype="0" conaffinity="0" rgba="0.8 0.8 0.8 1"/>
        </body>
        """
        return xml
