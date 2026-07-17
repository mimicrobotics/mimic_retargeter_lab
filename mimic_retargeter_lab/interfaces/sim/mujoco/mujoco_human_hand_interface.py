import numpy as np


class MujocoHumanHandInterface:
    """
    Visualizes a human hand in MuJoCo by drawing a sphere for each joint
    and manually creating, placing, and sizing capsule geoms for each link.
    """

    _NUM_JOINTS = 21  # Standard number of joints for a hand model

    # --- ADDED THIS ---
    # Define the 20 links connecting the 21 joints (based on standard
    # MediaPipe/MANO ordering: 0=wrist, 1-4=thumb, 5-8=index, etc.)
    _HAND_LINKS = [
        # Palm
        (0, 1),
        (0, 5),
        (0, 9),
        (0, 13),
        (0, 17),
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
    _NUM_LINKS = len(_HAND_LINKS)
    _LINK_RADIUS = 0.005  # Define a constant for the bone radius
    # ------------------

    def __init__(self):
        """
        Initializes the MuJoCo scene, model, and viewer.

        Args:
            human_hand_data_source: A data source that provides hand joint data.
        """
        import mujoco
        import mujoco.viewer

        self.mujoco = mujoco

        xml = self._create_mujoco_xml()
        self.model = self.mujoco.MjModel.from_xml_string(xml)
        self.data = self.mujoco.MjData(self.model)

        # --- MODIFIED THIS SECTION ---
        # Store mocap IDs for the joints for quick access
        self._joint_mocap_ids = [
            self.model.body(f"joint_{i}").mocapid[0] for i in range(self._NUM_JOINTS)
        ]

        # Store mocap IDs for the links (to set pos/quat)
        self._link_mocap_ids = [
            self.model.body(f"link_{i}").mocapid[0] for i in range(self._NUM_LINKS)
        ]

        # Store geom IDs for the links (to set size)
        self._link_geom_ids = [
            self.model.geom(f"link_geom_{i}").id for i in range(self._NUM_LINKS)
        ]

        # A buffer for quaternion calculations
        self._quat_buffer = np.zeros(4)
        # ---------------------------

        self.viewer = self.mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.opt.frame = 1

    def _create_mujoco_xml(self) -> str:
        """Generates the MJCF model string for the joint visualization."""

        # --- MODIFIED THIS SECTION ---

        # 1. Mocap bodies for joints
        joint_bodies_xml = "".join(
            [
                f'<body name="joint_{i}" mocap="true">'
                f'<geom type="sphere" size="0.008" rgba="0.8 0.2 0.2 0.7" contype="0" conaffinity="0"/>'
                f"</body>"
                for i in range(self._NUM_JOINTS)
            ]
        )

        # 2. Mocap bodies for links
        # We give them a tiny default half-height (0.001)
        link_bodies_xml = "".join(
            [
                f'<body name="link_{i}" mocap="true">'
                f'<geom name="link_geom_{i}" type="capsule" size="{self._LINK_RADIUS} 0.001" rgba="0.7 0.7 0.7 0.8" contype="0" conaffinity="0"/>'
                f"</body>"
                for i in range(self._NUM_LINKS)
            ]
        )

        # 3. Combine into the full XML
        xml_str = f"""
        <mujoco>
          <option>
            <flag gravity="disable"/>
          </option>
          <worldbody>
            <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
            <geom type="plane" pos="0 0 -0.1" size="1 1 0.1" rgba=".9 .9 .9 1"/>

            {joint_bodies_xml}

            {link_bodies_xml}
          </worldbody>
        </mujoco>
        """
        # ---------------------------
        return xml_str

    def update_hand_frames(self, hand_frames: np.ndarray, links: list = None):
        """
        Updates the joint positions from the 4x4 transformation matrices.

        Args:
            hand_frames (np.ndarray): Array of shape (21, 4, 4) for joint transforms.
            links (list, optional): This argument is ignored. Defaults to None.
        """
        if hand_frames.shape != (self._NUM_JOINTS, 4, 4):
            raise ValueError(
                f"Expected hand_frames of shape ({self._NUM_JOINTS}, 4, 4), but got {hand_frames.shape}"
            )

        # Extract the translation part (x, y, z) from each 4x4 matrix
        joint_pos = hand_frames[:, :3, 3]
        joint_rots = hand_frames[:, :3, :3]

        for rot, mocap_id in zip(joint_rots, self._joint_mocap_ids):
            mj_quat = np.zeros(4)
            self.mujoco.mju_mat2Quat(mj_quat, rot.flatten())
            self.data.mocap_quat[mocap_id] = mj_quat
        # Set the position of the mocap bodies
        self.data.mocap_pos[self._joint_mocap_ids] = joint_pos

        # --- ADDED THIS SECTION (MANUAL LINK UPDATE) ---

        # Get the full array of joint positions from mjData
        all_joint_positions = self.data.mocap_pos[self._joint_mocap_ids]

        for i in range(self._NUM_LINKS):
            # 1. Get endpoint joint indices and mocap/geom IDs
            j1, j2 = self._HAND_LINKS[i]
            mocap_id = self._link_mocap_ids[i]
            geom_id = self._link_geom_ids[i]

            # 2. Get 3D positions
            pos1 = all_joint_positions[j1]
            pos2 = all_joint_positions[j2]

            # 3. Calculate midpoint, vector, and length
            mid_pos = (pos1 + pos2) / 2
            vec = pos2 - pos1
            length = np.linalg.norm(vec)

            # 4. Calculate orientation quaternion (aligning Z-axis with vec)
            self.mujoco.mju_quatZ2Vec(self._quat_buffer, vec)

            # 5. Set mocap body position and orientation
            self.data.mocap_pos[mocap_id] = mid_pos
            self.data.mocap_quat[mocap_id] = self._quat_buffer

            # 6. Set geom size (radius, half-height)
            # This modifies the model itself, which is reflected in the next render.
            self.model.geom_size[geom_id, 1] = length / 2
        # ---------------------------------------------

    def step(self):
        """Advances the simulation by one step and syncs the viewer."""
        # mj_step() is called to ensure the mocap body positions are updated in the simulation state.
        self.mujoco.mj_step(self.model, self.data)
        self.viewer.sync()
