import numpy as np
from scipy.spatial.transform import Rotation as R
from mimic_retargeter_lab.hand_models import RobotHandModel
from mimic_retargeter_lab.interfaces import BaseHandInterface


class MujocoHandInterface(BaseHandInterface):
    def __init__(self, hand_model: RobotHandModel):
        import mujoco
        import mujoco.viewer

        self.mujoco = mujoco

        # Create enhanced model with mocap spheres for fingertip visualization
        enhanced_xml = self._create_enhanced_xml_with_spheres(
            hand_model.get_model_path(), max_spheres=hand_model.get_num_fingertips()
        )
        self.mj_model = mujoco.MjModel.from_xml_string(enhanced_xml)
        self.mj_model.opt.gravity[:] = [0, 0, 0]
        self.mj_data = mujoco.MjData(self.mj_model)

        # Add overhead lighting for better visualization
        self._setup_lighting()

        # Launch viewer
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        # Configure viewer settings for better visualization
        self._configure_viewer()

        # Initialize simulation
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.viewer.sync()

        # Frame visualization variables
        self.fingertip_frames = None
        self.mocap_sphere_ids = []
        self._setup_mocap_spheres()

    def _setup_lighting(self):
        """Setup overhead lighting for better visualization."""
        # Enable frame visualization in the model
        self.mj_model.vis.global_.fovy = 45
        self.mj_model.vis.quality.shadowsize = 4096

        # Set up ambient lighting
        self.mj_model.vis.headlight.ambient = [0.3, 0.3, 0.3]
        self.mj_model.vis.headlight.diffuse = [0.7, 0.7, 0.7]
        self.mj_model.vis.headlight.specular = [0.1, 0.1, 0.1]

    def _configure_viewer(self):
        """Configure viewer settings for optimal visualization."""
        if hasattr(self.viewer, "cam"):
            # Set camera position for better viewing angle
            self.viewer.cam.distance = 0.5
            self.viewer.cam.azimuth = 45
            self.viewer.cam.elevation = -20

        # Enable frame visualization in viewer
        if hasattr(self.viewer, "opt"):
            self.viewer.opt.frame = 1  # Show coordinate frames
            self.viewer.opt.label = 1  # Disable labels to reduce clutter

    def _create_enhanced_xml_with_spheres(self, original_model_path, max_spheres=10):
        """Create enhanced XML with mocap spheres for fingertip visualization."""
        import xml.etree.ElementTree as ET

        # Read the original XML
        tree = ET.parse(original_model_path)
        root = tree.getroot()

        # Find or create worldbody
        worldbody = root.find("worldbody")
        if worldbody is None:
            worldbody = ET.SubElement(root, "worldbody")

        # Add mocap spheres for fingertip visualization
        for i in range(max_spheres):
            sphere_body = ET.SubElement(worldbody, "body")
            sphere_body.set("name", f"fingertip_sphere_{i}")
            sphere_body.set("mocap", "true")
            sphere_body.set("pos", "0 0 -1")  # Start hidden underground
            sphere_body.set("quat", "0 0 0 1")  # Start with identity orientation

            sphere_geom = ET.SubElement(sphere_body, "geom")
            sphere_geom.set("name", f"fingertip_sphere_geom_{i}")
            sphere_geom.set("type", "sphere")
            sphere_geom.set("size", "0.015")
            sphere_geom.set(
                "rgba", f"{1.0 - i * 0.05} 0 0 0.0"
            )  # Different shades of red
            sphere_geom.set("contype", "0")
            sphere_geom.set("conaffinity", "0")

        # Convert to string
        xml_string = ET.tostring(root, encoding="unicode", xml_declaration=True)
        return xml_string

    def _setup_mocap_spheres(self):
        """Setup mocap sphere IDs for fingertip visualization."""
        self.mocap_sphere_ids = []

        # Find mocap bodies for spheres
        for i in range(self.mj_model.nbody):
            if self.mj_model.body_mocapid[i] >= 0:
                body_name = self.mujoco.mj_id2name(
                    self.mj_model, self.mujoco.mjtObj.mjOBJ_BODY, i
                )
                if body_name and "fingertip_sphere_" in body_name:
                    self.mocap_sphere_ids.append(self.mj_model.body_mocapid[i])

        print(f"Found {len(self.mocap_sphere_ids)} mocap spheres for visualization")

        # Hide all spheres initially
        for mocap_id in self.mocap_sphere_ids:
            self.mj_data.mocap_pos[mocap_id] = np.array([0, 0, -1])  # Underground

    def update_fingertip_frames(self, fingertip_frames):
        """
        Update the visualization of fingertip frames using mocap spheres.

        Args:
            fingertip_frames: Array of shape (N, B, 4, 4) containing transformation matrices
                            for N fingertip frames, or (4, 4) for a single frame.
        """
        # Store the current frames
        self.fingertip_frames = fingertip_frames.squeeze(1)

        # Ensure fingertip_frames is a 3D array
        # if fingertip_frames.ndim == 2:
        #     fingertip_frames = fingertip_frames[np.newaxis, ...]

        # print(f"fingertip frames shape: {fingertip_frames.shape}")

        # Update mocap sphere positions
        for i, frame in enumerate(self.fingertip_frames):
            # print(f"frame shape: {frame.shape}")
            pos = frame[:3, 3]
            mocap_id = self.mocap_sphere_ids[i]
            self.mj_data.mocap_pos[mocap_id] = pos
            mj_quat = np.zeros(4)
            self.mujoco.mju_mat2Quat(mj_quat, frame[:3, :3].flatten())
            self.mj_data.mocap_quat[mocap_id] = mj_quat

        # Hide unused spheres
        for i in range(len(fingertip_frames), len(self.mocap_sphere_ids)):
            mocap_id = self.mocap_sphere_ids[i]
            self.mj_data.mocap_pos[mocap_id] = np.array([0, 0, -1])  # Hide underground

    def get_joint_angles(self):
        """Get current joint angles from the simulation."""
        self.joint_angles = self.mj_data.qpos
        return self.joint_angles

    def set_joint_angles(self, joint_angles):
        """Set target joint angles for the hand."""
        self.commanded_joint_angles = joint_angles
        self.mj_data.ctrl[:] = joint_angles

    def step(self):
        """Step the simulation forward."""
        if self.viewer.is_running():
            # Step the physics simulation
            self.mujoco.mj_step(self.mj_model, self.mj_data)

            # Synchronize viewer
            self.viewer.sync()

    def toggle_joint_frames(self, show=True):
        """Toggle visualization of joint coordinate frames."""
        if hasattr(self.viewer, "opt"):
            self.viewer.opt.frame = 1 if show else 0

    def toggle_labels(self, show=True):
        """Toggle visualization of labels."""
        if hasattr(self.viewer, "opt"):
            self.viewer.opt.label = 1 if show else 0

    def show_frame_info(self):
        """Display current frame information."""
        if self.fingertip_frames is not None:
            print("=== Current Fingertip Frames ===")
            frames = self.fingertip_frames
            if frames.ndim == 2:
                frames = frames[np.newaxis, ...]
            for i, frame in enumerate(frames):
                pos = frame[:3, 3]
                rot_matrix = frame[:3, :3]
                rotation = R.from_matrix(rot_matrix)
                euler = rotation.as_euler("xyz", degrees=True)
                print(
                    f"  Frame {i}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], "
                    f"euler=[{euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°]"
                )
        else:
            print("No fingertip frames currently set")
