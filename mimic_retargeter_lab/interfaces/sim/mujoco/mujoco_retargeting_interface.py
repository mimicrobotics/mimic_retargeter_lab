import numpy as np
import os
import xml.etree.ElementTree as ET
from mimic_retargeter_lab.hand_models import RobotHandModel
from mimic_retargeter_lab.interfaces import BaseHandInterface
from mimic_retargeter_lab.utils.mj_utils import get_actuated_joints_indices


class MujocoRetargetingInterface(BaseHandInterface):
    """
    Implements the BaseHandInterface for the MuJoCo simulator.

    This class loads a robot hand model (from BaseHandModel) and sets up a
    MuJoCo simulation. It provides methods to:
    - Set the robot's joint angles (using position actuators via 'ctrl').
    - Get the robot's current joint angles.
    - Step the simulation.
    - Visualize fingertip frames (using mocap spheres).
    - Visualize a human hand skeleton (using mocap spheres and capsules)
      for debug and retargeting visualization.
    """

    # --- Default constants for human hand visualization (MANO 21-point) ---
    _DEFAULT_NUM_JOINTS = 21
    _DEFAULT_HAND_LINKS = [
        # Palm
        (0, 1),
        (0, 5),
        (0, 9),
        (0, 13),
        (0, 17),
        (5, 9),
        (9, 13),
        (13, 17),
        # Thumb
        (1, 2),
        (2, 3),
        (3, 4),
        # Index
        (5, 6),
        (6, 7),
        (7, 8),
        # Middle
        (9, 10),
        (10, 11),
        (11, 12),
        # Ring
        (13, 14),
        (14, 15),
        (15, 16),
        # Pinky
        (17, 18),
        (18, 19),
        (19, 20),
    ]
    _LINK_RADIUS = 0.005
    _NUM_KABSCH_LANDMARKS = 3
    _KABSCH_LINK_RADIUS = 0.0035
    _NUM_KABSCH_KEYVEC_LINES = 15
    # ----------------------------------------------------------------------------

    def __init__(
        self,
        hand_model: RobotHandModel,
        num_hand_joints: int | None = None,
        hand_links: list[tuple[int, int]] | None = None,
    ):
        import mujoco
        import mujoco.viewer

        self.mujoco = mujoco
        self.hand_model = hand_model

        # Configurable joint count and link topology for human hand viz.
        self._NUM_JOINTS = num_hand_joints or self._DEFAULT_NUM_JOINTS
        self._HAND_LINKS = hand_links or self._DEFAULT_HAND_LINKS
        self._NUM_LINKS = len(self._HAND_LINKS)
        self._NUM_KABSCH_POINTS = self._NUM_JOINTS

        # Create enhanced model with mocap bodies
        curr_dir = os.getcwd()
        os.chdir(os.path.dirname(hand_model.get_model_path()))
        enhanced_xml = self._create_enhanced_xml(
            hand_model.get_model_path(), num_fingertips=hand_model.get_num_fingertips()
        )
        self.mj_model = mujoco.MjModel.from_xml_string(enhanced_xml)
        os.chdir(curr_dir)
        self.mj_model.opt.gravity[:] = [0, 0, 0]
        self.mj_data = mujoco.MjData(self.mj_model)

        # Add overhead lighting for better visualization
        self._setup_lighting()

        # Launch viewer
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        # Configure viewer settings for better visualization
        self._configure_viewer()

        # --- Hand model setup ---
        joint_names = self.hand_model.get_qpos_joint_names()

        # Get DoF indices for reading joint positions
        self.joint_ids = [
            self.mujoco.mj_name2id(self.mj_model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in joint_names
        ]
        self.joint_ids = [
            jid for jid in self.joint_ids if jid != -1
        ]  # Filter not-found
        self.dof_indices = [self.mj_model.jnt_qposadr[jid] for jid in self.joint_ids]

        # Get actuator indices for setting joint targets
        # (Assumes actuators are named identically to joints, which is common)
        self.actuator_indices = get_actuated_joints_indices(self.mj_model)
        # self.actuator_indices = [
        #     self.mujoco.mj_name2id(
        #         self.mj_model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, name
        #     )
        #     for name in joint_names
        # ]
        # self.actuator_indices = [idx for idx in self.actuator_indices if idx != -1]

        if len(self.dof_indices) != len(joint_names) or len(
            self.actuator_indices
        ) != len(joint_names):
            # print(f"Warning: Mismatch in expected joints ({len(joint_names)}).")
            print(
                f"Found {len(self.dof_indices)} DoFs and {len(self.actuator_indices)} actuators."
            )
            if len(self.actuator_indices) == 0:
                print(
                    "ERROR: No actuators found. The MuJoCo XML file must define actuators (e.g., <position>) for the joints."
                )

        # ----------------------------------------------------------------

        # --- Visualization setup ---
        # For robot fingertips
        self.fingertip_frames = None
        self.mocap_sphere_ids = []
        self._setup_mocap_spheres()  # Finds 'fingertip_sphere_...'

        # For human hand debug visualization
        self.hand_frames = None
        self.links = None  # Kept for API consistency
        self._setup_human_hand_mocap()  # Finds 'human_joint_...' etc.
        self._setup_kabsch_debug_mocap()
        self._quat_buffer = np.zeros(4)  # For link orientation calculation

        # Initialize simulation
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.viewer.sync()

    def step(self):
        """Advances the simulation by one step and syncs the viewer."""
        if self.viewer.is_running():
            try:
                # Step the physics simulation (runs PD controllers)
                self.mujoco.mj_step(self.mj_model, self.mj_data)
                # Synchronize viewer
                self.viewer.sync()
            except self.mujoco.FatalError as e:
                print(f"MuJoCo simulation error: {e}")
                self.viewer.close()

    def update_fingertip_frames(
        self, fingertip_frames: np.ndarray, transl: np.ndarray = np.array([1, 0, 0])
    ):
        """
        Update the visualization of robot fingertip frames using mocap spheres.

        Args:
            fingertip_frames: Array of shape (N, 4, 4) or (4, 4) containing
                              transformation matrices for N fingertip frames.
        """
        self.fingertip_frames = fingertip_frames
        transl_transform = np.eye(4)
        transl_transform[:3, 3] = transl
        fingertip_frames = fingertip_frames @ transl_transform
        if fingertip_frames.ndim == 2:
            fingertip_frames = fingertip_frames[np.newaxis, ...]

        for i, frame in enumerate(fingertip_frames):
            if i >= len(self.mocap_sphere_ids):
                break

            pos = frame[:3, 3]
            mocap_id = self.mocap_sphere_ids[i]
            self.mj_data.mocap_pos[mocap_id] = pos

            mj_quat = np.zeros(4)
            self.mujoco.mju_mat2Quat(mj_quat, frame[:3, :3].flatten())
            self.mj_data.mocap_quat[mocap_id] = mj_quat

        for i in range(len(fingertip_frames), len(self.mocap_sphere_ids)):
            mocap_id = self.mocap_sphere_ids[i]
            self.mj_data.mocap_pos[mocap_id] = np.array([0, 0, -1])

    def update_hand_frames(self, hand_frames: np.ndarray, links: list = None):
        """
        Updates the human hand visualization (joints and links) from 4x4
        transformation matrices.
        """
        if hand_frames.shape != (self._NUM_JOINTS, 4, 4):
            if hand_frames.ndim == 3 and hand_frames.shape[0] != self._NUM_JOINTS:
                print(
                    f"Warning: update_hand_frames expected shape ({self._NUM_JOINTS}, 4, 4), but got {hand_frames.shape}. Taking first element."
                )
                hand_frames = hand_frames[0]
            elif hand_frames.ndim != 3:
                raise ValueError(
                    f"Expected hand_frames of shape ({self._NUM_JOINTS}, 4, 4), but got {hand_frames.shape}"
                )

        self.hand_frames = hand_frames

        joint_pos = hand_frames[:, :3, 3]
        joint_rots = hand_frames[:, :3, :3]

        # --- 1. Update Joint Spheres ---
        for i in range(self._NUM_JOINTS):
            if i >= len(self._joint_mocap_ids):
                break
            mocap_id = self._joint_mocap_ids[i]
            self.mj_data.mocap_pos[mocap_id] = joint_pos[i]
            mj_quat = np.zeros(4)
            self.mujoco.mju_mat2Quat(mj_quat, joint_rots[i].flatten())
            self.mj_data.mocap_quat[mocap_id] = mj_quat

        # --- 2. Update Link Capsules ---
        all_joint_positions = self.mj_data.mocap_pos[self._joint_mocap_ids]

        for i in range(self._NUM_LINKS):
            if i >= len(self._link_mocap_ids):
                break

            j1, j2 = self._HAND_LINKS[i]
            if j1 >= len(all_joint_positions) or j2 >= len(all_joint_positions):
                continue

            mocap_id = self._link_mocap_ids[i]
            geom_id = self._link_geom_ids[i]

            pos1 = all_joint_positions[j1]
            pos2 = all_joint_positions[j2]

            mid_pos = (pos1 + pos2) / 2
            vec = pos2 - pos1
            length = np.linalg.norm(vec)

            if length > 1e-6:
                self.mujoco.mju_quatZ2Vec(self._quat_buffer, vec)
            else:
                self._quat_buffer = [1, 0, 0, 0]

            self.mj_data.mocap_pos[mocap_id] = mid_pos
            self.mj_data.mocap_quat[mocap_id] = self._quat_buffer

            # --- FIXED: Changed self.model to self.mj_model ---
            self.mj_model.geom_size[geom_id, 1] = length / 2

    def update_kabsch_debug_landmarks(
        self,
        tgt_landmarks: np.ndarray | None,
        src_raw_points: np.ndarray | None,
        src_aligned_points: np.ndarray | None,
        src_keyvectors: np.ndarray | None,
        tgt_keyvectors: np.ndarray | None,
    ) -> None:
        """Update Kabsch debug landmark markers in the MuJoCo viewer.

        Args:
            tgt_landmarks: Points to visualize as target landmarks used for alignment.
            src_raw_points: Points to visualize as source raw points (pre-alignment).
            src_aligned_points: Points to visualize as source aligned points (post-alignment).
            src_keyvectors: Line segments (N, 2, 3) of source keyvectors —
                each row is a (start, end) pair drawn as a capsule.
            tgt_keyvectors: Line segments (N, 2, 3) of target keyvectors —
                each row is a (start, end) pair drawn as a capsule.
        """

        def _update_group(points: np.ndarray | None, mocap_ids: list[int]) -> None:
            if points is None:
                self.mj_data.mocap_pos[mocap_ids] = np.tile(
                    [0, 0, -1], (len(mocap_ids), 1)
                )
                return

            pts = np.asarray(points, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[1] != 3:
                raise ValueError(f"Expected points of shape (N, 3), got {pts.shape}")

            limit = min(len(mocap_ids), pts.shape[0])
            for i in range(limit):
                self.mj_data.mocap_pos[mocap_ids[i]] = pts[i]

            for i in range(limit, len(mocap_ids)):
                self.mj_data.mocap_pos[mocap_ids[i]] = np.array([0, 0, -1])

        _update_group(tgt_landmarks, self._kabsch_tgt_mocap_ids)
        _update_group(src_raw_points, self._kabsch_src_raw_cloud_mocap_ids)
        _update_group(src_aligned_points, self._kabsch_src_aligned_cloud_mocap_ids)

        def _update_keyvector_segments(
            segments: np.ndarray | None,
            link_mocap_ids: list[int],
            link_geom_ids: list[int],
        ) -> None:
            """Render an arbitrary set of line segments (start, end) as capsules.

            ``segments`` has shape (N, 2, 3): row i draws a capsule between
            ``segments[i, 0]`` and ``segments[i, 1]``. Unused mocap slots are
            parked off-screen with zero length.
            """
            if segments is None:
                for i in range(len(link_mocap_ids)):
                    self.mj_data.mocap_pos[link_mocap_ids[i]] = np.array([0, 0, -1])
                    self.mj_model.geom_size[link_geom_ids[i], 1] = 1e-6
                return

            seg = np.asarray(segments, dtype=np.float64)
            if seg.ndim != 3 or seg.shape[1] != 2 or seg.shape[2] != 3:
                raise ValueError(
                    f"Expected segments with shape (N, 2, 3), got {seg.shape}"
                )

            limit = min(len(link_mocap_ids), seg.shape[0])
            for i in range(limit):
                start = seg[i, 0]
                end = seg[i, 1]
                vec = end - start
                length = np.linalg.norm(vec)
                if length > 1e-6:
                    mid = (start + end) / 2.0
                    self.mujoco.mju_quatZ2Vec(self._quat_buffer, vec)
                    self.mj_data.mocap_pos[link_mocap_ids[i]] = mid
                    self.mj_data.mocap_quat[link_mocap_ids[i]] = self._quat_buffer
                    self.mj_model.geom_size[link_geom_ids[i], 1] = length / 2.0
                else:
                    self.mj_data.mocap_pos[link_mocap_ids[i]] = np.array([0, 0, -1])
                    self.mj_model.geom_size[link_geom_ids[i], 1] = 1e-6

            for i in range(limit, len(link_mocap_ids)):
                self.mj_data.mocap_pos[link_mocap_ids[i]] = np.array([0, 0, -1])
                self.mj_model.geom_size[link_geom_ids[i], 1] = 1e-6

        _update_keyvector_segments(
            src_keyvectors,
            self._kabsch_src_keyvec_link_mocap_ids,
            self._kabsch_src_keyvec_link_geom_ids,
        )
        _update_keyvector_segments(
            tgt_keyvectors,
            self._kabsch_tgt_keyvec_link_mocap_ids,
            self._kabsch_tgt_keyvec_link_geom_ids,
        )

    def set_hand_transform(
        self, hand_transform: np.ndarray, tgt_key: str = "rh_forearm"
    ):
        """Set the hand base (wrist/forearm) SE3 transform.

        Uses mocap if the target body is a mocap body, otherwise sets
        body_pos / body_quat directly on the model (works for regular
        kinematic bodies like the hand root).
        """
        bid = self.mujoco.mj_name2id(
            self.mj_model, self.mujoco.mjtObj.mjOBJ_BODY, tgt_key
        )
        if bid == -1:
            raise ValueError(f"Body '{tgt_key}' not found")

        trans = hand_transform[:3, 3]
        rot = np.ascontiguousarray(hand_transform[:3, :3].flatten(), dtype=np.float64)
        quat = np.zeros(4)
        self.mujoco.mju_mat2Quat(quat, rot)

        mocap_id = self.mj_model.body(bid).mocapid[0]
        if mocap_id >= 0:
            self.mj_data.mocap_pos[mocap_id] = trans
            self.mj_data.mocap_quat[mocap_id] = quat
        else:
            self.mj_model.body_pos[bid] = trans
            self.mj_model.body_quat[bid] = quat

    def get_joint_angles(self) -> np.ndarray:
        """Get current joint angles from the simulation."""
        if not self.dof_indices:
            return np.array([])
        return self.mj_data.qpos[self.dof_indices]

    def set_joint_angles(self, joint_angles: np.ndarray):
        """Set target joint angles for the hand's position actuators."""
        if len(joint_angles) != len(self.actuator_indices):
            # Don't throw an error if no actuators are found, just warn.
            if len(self.actuator_indices) == 0:
                return  # Warning was already printed in __init__
            raise ValueError(
                f"Input joint_angles has length {len(joint_angles)}, but model has {len(self.actuator_indices)} actuators."
            )
        # self.mj_data.ctrl[self.actuator_indices] = joint_angles
        # self.mj_data.ctrl = joint_angles
        self.mj_data.qpos[self.dof_indices] = joint_angles @ self.hand_model.joint_map.T

    # --- Helper Methods ---

    def _setup_lighting(self):
        self.mj_model.vis.global_.fovy = 45
        self.mj_model.vis.quality.shadowsize = 4096
        self.mj_model.vis.headlight.ambient = [0.3, 0.3, 0.3]
        self.mj_model.vis.headlight.diffuse = [0.7, 0.7, 0.7]
        self.mj_model.vis.headlight.specular = [0.1, 0.1, 0.1]

    def _configure_viewer(self):
        if hasattr(self.viewer, "cam"):
            self.viewer.cam.distance = 0.5
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -30
        if hasattr(self.viewer, "opt"):
            self.viewer.opt.frame = 1
            self.viewer.opt.label = 0

    def _create_enhanced_xml(self, original_model_path, num_fingertips) -> str:
        try:
            tree = ET.parse(original_model_path)
            root = tree.getroot()
        except ET.ParseError as e:
            print(f"Error parsing XML file: {original_model_path}")
            raise e

        worldbody = root.find("worldbody")
        if worldbody is None:
            worldbody = ET.SubElement(root, "worldbody")

        if worldbody.find("geom[@type='plane']") is None:
            ET.SubElement(
                worldbody,
                "geom",
                type="plane",
                pos="0 0 -0.3",
                size="1 1 0.1",
                rgba=".9 .9 .9 1",
            )

        # 0. World frame axes (RGB = XYZ)
        axis_length = "0.15"
        axis_radius = "0.003"
        for axis_name, fromto, rgba in [
            ("world_axis_x", f"0 0 0 {axis_length} 0 0", "1 0 0 0.8"),
            ("world_axis_y", f"0 0 0 0 {axis_length} 0", "0 1 0 0.8"),
            ("world_axis_z", f"0 0 0 0 0 {axis_length}", "0 0 1 0.8"),
        ]:
            ET.SubElement(
                worldbody,
                "geom",
                name=axis_name,
                type="capsule",
                size=axis_radius,
                fromto=fromto,
                rgba=rgba,
                contype="0",
                conaffinity="0",
            )

        # 1. Add mocap spheres for ROBOT fingertips
        for i in range(num_fingertips):
            sphere_body = ET.SubElement(
                worldbody,
                "body",
                name=f"fingertip_sphere_{i}",
                mocap="true",
                pos="0 0 -1",
            )
            ET.SubElement(
                sphere_body,
                "geom",
                name=f"fingertip_sphere_geom_{i}",
                type="sphere",
                size="0.01",
                rgba=f"{1.0 - i * 0.1} 0 0 0.5",
                contype="0",
                conaffinity="0",
            )

        # 2. Mocap bodies for HUMAN joints
        for i in range(self._NUM_JOINTS):
            joint_body = ET.SubElement(
                worldbody, "body", name=f"human_joint_{i}", mocap="true", pos="0 0 -1"
            )
            ET.SubElement(
                joint_body,
                "geom",
                type="sphere",
                size="0.008",
                rgba="0.2 0.2 0.8 0.7",
                contype="0",
                conaffinity="0",
            )

        # 3. Mocap bodies for HUMAN links
        for i in range(self._NUM_LINKS):
            link_body = ET.SubElement(
                worldbody, "body", name=f"human_link_{i}", mocap="true", pos="0 0 -1"
            )
            ET.SubElement(
                link_body,
                "geom",
                name=f"human_link_geom_{i}",
                type="capsule",
                size=f"{self._LINK_RADIUS} 0.001",
                rgba="0.7 0.7 0.7 0.6",
                contype="0",
                conaffinity="0",
            )

        # 4. Kabsch debug landmarks (source raw, source aligned, target)
        kabsch_groups = [
            ("kabsch_src_raw", "0.95 0.15 0.15 0.95"),
            ("kabsch_src_aligned", "1.0 0.55 0.15 0.95"),
            ("kabsch_tgt", "0.15 0.35 1.0 0.95"),
        ]
        for group_name, rgba in kabsch_groups:
            for i in range(self._NUM_KABSCH_LANDMARKS):
                marker_body = ET.SubElement(
                    worldbody,
                    "body",
                    name=f"{group_name}_{i}",
                    mocap="true",
                    pos="0 0 -1",
                )
                ET.SubElement(
                    marker_body,
                    "geom",
                    name=f"{group_name}_geom_{i}",
                    type="sphere",
                    size="0.012",
                    rgba=rgba,
                    contype="0",
                    conaffinity="0",
                )

        # 5. Kabsch full source clouds (before and after alignment)
        cloud_groups = [
            ("kabsch_src_raw_cloud", "0.95 0.15 0.15 0.45"),
            ("kabsch_src_aligned_cloud", "1.0 0.55 0.15 0.55"),
        ]
        for group_name, rgba in cloud_groups:
            for i in range(self._NUM_KABSCH_POINTS):
                marker_body = ET.SubElement(
                    worldbody,
                    "body",
                    name=f"{group_name}_{i}",
                    mocap="true",
                    pos="0 0 -1",
                )
                ET.SubElement(
                    marker_body,
                    "geom",
                    name=f"{group_name}_geom_{i}",
                    type="sphere",
                    size="0.0075",
                    rgba=rgba,
                    contype="0",
                    conaffinity="0",
                )

        # 6. Kabsch keyvector lines within each model (wrist -> fingertips)
        for i in range(self._NUM_KABSCH_KEYVEC_LINES):
            link_body = ET.SubElement(
                worldbody,
                "body",
                name=f"kabsch_src_keyvec_link_{i}",
                mocap="true",
                pos="0 0 -1",
            )
            ET.SubElement(
                link_body,
                "geom",
                name=f"kabsch_src_keyvec_link_geom_{i}",
                type="capsule",
                size=f"{self._KABSCH_LINK_RADIUS} 0.001",
                rgba="1.0 0.15 0.15 0.95",
                contype="0",
                conaffinity="0",
            )
        for i in range(self._NUM_KABSCH_KEYVEC_LINES):
            link_body = ET.SubElement(
                worldbody,
                "body",
                name=f"kabsch_tgt_keyvec_link_{i}",
                mocap="true",
                pos="0 0 -1",
            )
            ET.SubElement(
                link_body,
                "geom",
                name=f"kabsch_tgt_keyvec_link_geom_{i}",
                type="capsule",
                size=f"{self._KABSCH_LINK_RADIUS} 0.001",
                rgba="0.15 0.35 1.0 0.95",
                contype="0",
                conaffinity="0",
            )

        return ET.tostring(root, encoding="unicode")

    def _setup_mocap_spheres(self):
        """Find and store mocap IDs for robot fingertip visualization."""
        self.mocap_sphere_ids = []
        for i in range(self.mj_model.nbody):
            if self.mj_model.body_mocapid[i] >= 0:
                body_name = self.mujoco.mj_id2name(
                    self.mj_model, self.mujoco.mjtObj.mjOBJ_BODY, i
                )
                if body_name and body_name.startswith("fingertip_sphere_"):
                    mocap_id = self.mj_model.body_mocapid[i]
                    try:
                        index = int(body_name.split("_")[-1])
                        self.mocap_sphere_ids.append((index, mocap_id))
                    except (ValueError, IndexError):
                        print(
                            f"Warning: Could not parse index from mocap body name: {body_name}"
                        )

        self.mocap_sphere_ids.sort()
        self.mocap_sphere_ids = [mocap_id for index, mocap_id in self.mocap_sphere_ids]

        if self.mocap_sphere_ids:
            self.mj_data.mocap_pos[self.mocap_sphere_ids] = np.tile(
                [0, 0, -1], (len(self.mocap_sphere_ids), 1)
            )

    def _setup_human_hand_mocap(self):
        """Find and store mocap/geom IDs for human hand visualization."""
        try:
            self._joint_mocap_ids = [
                self.mj_model.body(f"human_joint_{i}").mocapid[0]
                for i in range(self._NUM_JOINTS)
            ]
            self._link_mocap_ids = [
                self.mj_model.body(f"human_link_{i}").mocapid[0]
                for i in range(self._NUM_LINKS)
            ]
            self._link_geom_ids = [
                self.mj_model.geom(f"human_link_geom_{i}").id
                for i in range(self._NUM_LINKS)
            ]

            all_mocap_ids = self._joint_mocap_ids + self._link_mocap_ids
            if all_mocap_ids:
                self.mj_data.mocap_pos[all_mocap_ids] = np.tile(
                    [0, 0, -1], (len(all_mocap_ids), 1)
                )

        except KeyError as e:
            print(
                f"Error finding human hand mocap bodies. Was the XML enhanced correctly? {e}"
            )
            self._joint_mocap_ids = []
            self._link_mocap_ids = []
            self._link_geom_ids = []
        except Exception as e:
            print(f"An unexpected error occurred in _setup_human_hand_mocap: {e}")
            self._joint_mocap_ids = []
            self._link_mocap_ids = []
            self._link_geom_ids = []

    def _setup_kabsch_debug_mocap(self):
        """Find and store mocap IDs for Kabsch debug landmark visualization."""
        try:
            self._kabsch_src_raw_mocap_ids = [
                self.mj_model.body(f"kabsch_src_raw_{i}").mocapid[0]
                for i in range(self._NUM_KABSCH_LANDMARKS)
            ]
            self._kabsch_src_aligned_mocap_ids = [
                self.mj_model.body(f"kabsch_src_aligned_{i}").mocapid[0]
                for i in range(self._NUM_KABSCH_LANDMARKS)
            ]
            self._kabsch_tgt_mocap_ids = [
                self.mj_model.body(f"kabsch_tgt_{i}").mocapid[0]
                for i in range(self._NUM_KABSCH_LANDMARKS)
            ]
            self._kabsch_src_raw_cloud_mocap_ids = [
                self.mj_model.body(f"kabsch_src_raw_cloud_{i}").mocapid[0]
                for i in range(self._NUM_KABSCH_POINTS)
            ]
            self._kabsch_src_aligned_cloud_mocap_ids = [
                self.mj_model.body(f"kabsch_src_aligned_cloud_{i}").mocapid[0]
                for i in range(self._NUM_KABSCH_POINTS)
            ]
            self._kabsch_src_keyvec_link_mocap_ids = [
                self.mj_model.body(f"kabsch_src_keyvec_link_{i}").mocapid[0]
                for i in range(self._NUM_KABSCH_KEYVEC_LINES)
            ]
            self._kabsch_src_keyvec_link_geom_ids = [
                self.mj_model.geom(f"kabsch_src_keyvec_link_geom_{i}").id
                for i in range(self._NUM_KABSCH_KEYVEC_LINES)
            ]
            self._kabsch_tgt_keyvec_link_mocap_ids = [
                self.mj_model.body(f"kabsch_tgt_keyvec_link_{i}").mocapid[0]
                for i in range(self._NUM_KABSCH_KEYVEC_LINES)
            ]
            self._kabsch_tgt_keyvec_link_geom_ids = [
                self.mj_model.geom(f"kabsch_tgt_keyvec_link_geom_{i}").id
                for i in range(self._NUM_KABSCH_KEYVEC_LINES)
            ]

            all_mocap_ids = (
                self._kabsch_src_raw_mocap_ids
                + self._kabsch_src_aligned_mocap_ids
                + self._kabsch_tgt_mocap_ids
                + self._kabsch_src_raw_cloud_mocap_ids
                + self._kabsch_src_aligned_cloud_mocap_ids
                + self._kabsch_src_keyvec_link_mocap_ids
                + self._kabsch_tgt_keyvec_link_mocap_ids
            )
            if all_mocap_ids:
                self.mj_data.mocap_pos[all_mocap_ids] = np.tile(
                    [0, 0, -1], (len(all_mocap_ids), 1)
                )
        except Exception as e:
            print(f"An unexpected error occurred in _setup_kabsch_debug_mocap: {e}")
            self._kabsch_src_raw_mocap_ids = []
            self._kabsch_src_aligned_mocap_ids = []
            self._kabsch_tgt_mocap_ids = []
            self._kabsch_src_raw_cloud_mocap_ids = []
            self._kabsch_src_aligned_cloud_mocap_ids = []
            self._kabsch_src_keyvec_link_mocap_ids = []
            self._kabsch_src_keyvec_link_geom_ids = []
            self._kabsch_tgt_keyvec_link_mocap_ids = []
            self._kabsch_tgt_keyvec_link_geom_ids = []
