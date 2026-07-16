import threading
from pathlib import Path

import numpy as np

from dexworld.data_sources import BaseHandDataSource, BaseWristDataSource
from dexworld.hand_models import create_robot_hand
from dexworld.retargeting.online import create_retargeter
from dexworld.types import Chirality, HandLandmark, RobotHandType, Retargeter, Simulator
from dexworld.utils import DataLogger


class _WristReaderThread:
    """Reads from a BaseWristDataSource in a daemon thread, keeping the latest wrist pose."""

    def __init__(self, wrist_data_source: BaseWristDataSource):
        self._source = wrist_data_source
        self._lock = threading.Lock()
        self._latest_wrist_pose = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        for frame in self._source.get_iter():
            with self._lock:
                self._latest_wrist_pose = frame["wrist_pose"]

    def get_latest_wrist_pose(self) -> np.ndarray | None:
        with self._lock:
            return self._latest_wrist_pose


class KinematicRetargetingScene:
    def __init__(
        self,
        hand_type: RobotHandType,
        robot_hand_base_path: Path,
        chirality: Chirality,
        hand_data_source: BaseHandDataSource,
        simulator: Simulator,
        retargeter: Retargeter,
        retargeter_cfg={},
        num_episodes: int = 1,
        wrist_data_source: BaseWristDataSource | None = None,
    ):
        self.human_hand_data_source = hand_data_source

        self.human_hand_frame_iter = self.human_hand_data_source.get_iter()
        self.human_hand_model = self.human_hand_data_source.hand_model
        self.data_logger = DataLogger()
        self.idx = 0

        self.robot_hand_model = create_robot_hand(
            hand_type, robot_hand_base_path, chirality
        )

        # Resolve wrist body name from the robot hand model's landmark config.
        arm_attachment = self.robot_hand_model._landmark_config.get(
            HandLandmark.ARM_ATTACHMENT
        )
        self._wrist_body_name = arm_attachment.name

        # Determine human hand viz parameters (default 21-point MANO, or
        # 25-point MANUS if the data source uses ManusHandModel).
        viz_kwargs = {}
        try:
            from dexworld.hand_models.manus_hand import ManusHandModel

            if isinstance(self.human_hand_model, ManusHandModel):
                viz_kwargs = {
                    "num_hand_joints": ManusHandModel.NUM_NODES,
                    "hand_links": ManusHandModel.HAND_LINKS_25,
                }
        except ImportError:
            pass

        match simulator:
            case Simulator.MUJOCO:
                from dexworld.interfaces import MujocoRetargetingInterface

                self.hand_interface = MujocoRetargetingInterface(
                    self.robot_hand_model, **viz_kwargs
                )

        # Initialize wrist controller to the hand's rest pose from the MJCF,
        # so it doesn't snap to identity on the first frame.
        self._wrist_reader = None
        if wrist_data_source is not None:
            rest_pose = self._get_body_rest_pose(self._wrist_body_name)
            wrist_data_source.reset_pose(rest_pose)
            self._wrist_reader = _WristReaderThread(wrist_data_source)

        self.retargeter = create_retargeter(
            retargeter,
            from_model=self.human_hand_model,
            to_model=self.robot_hand_model,
            **retargeter_cfg,
        )

    def _get_body_rest_pose(self, body_name: str) -> np.ndarray:
        """Read a body's pos/quat from the loaded MuJoCo model and return as 4x4."""
        import mujoco

        mj_model = self.hand_interface.mj_model
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid == -1:
            return np.eye(4, dtype=np.float64)

        pos = mj_model.body_pos[bid].copy()
        quat = mj_model.body_quat[bid].copy()  # MuJoCo (w, x, y, z)

        rot = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot, quat)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot.reshape(3, 3)
        T[:3, 3] = pos
        return T

    def step(
        self, update_data: bool = True, repeating: bool = True, sleep_time: float = 0.1
    ):
        if update_data:
            try:
                human_hand_dict = next(self.human_hand_frame_iter)
                self.hand_interface.update_hand_frames(
                    human_hand_dict["transforms"], human_hand_dict["links"]
                )
            except StopIteration:
                if not repeating:
                    return True
                self.human_hand_frame_iter = self.human_hand_data_source.get_iter()
                self.idx = 0
                human_hand_dict = next(self.human_hand_frame_iter)
                self.hand_interface.update_hand_frames(
                    human_hand_dict["transforms"], human_hand_dict["links"]
                )

            # retargeting
            src_pcloud = np.asarray(human_hand_dict["joints"], dtype=np.float32)
            robot_joints, _ = self.retargeter.retarget(src_pcloud)

            # Optional MuJoCo Kabsch landmark debug overlay.
            # Applies only to retargeters that use keyvector-based retargeting, and only if debug_mode is enabled in the retargeter config.
            # TODO: Make it more general with other retargeter types and/or debug visualizations
            if (
                getattr(self.retargeter, "debug_mode", False)
                and hasattr(self.retargeter, "debug_visualization_data")
                and hasattr(self.hand_interface, "update_kabsch_debug_landmarks")
            ):
                debug_viz_data = self.retargeter.debug_visualization_data
                if debug_viz_data is not None:
                    self.hand_interface.update_kabsch_debug_landmarks(
                        debug_viz_data.get("tgt_landmarks"),
                        debug_viz_data.get("src_raw_points"),
                        debug_viz_data.get("src_aligned_points"),
                        debug_viz_data.get("src_keyvectors"),
                        debug_viz_data.get("tgt_keyvectors"),
                    )
                else:
                    self.hand_interface.update_kabsch_debug_landmarks(
                        None, None, None, None, None
                    )

            robot_joints = robot_joints.squeeze(0)
            # update hand_interface
            self.hand_interface.set_joint_angles(robot_joints)

        # time.sleep(sleep_time)

        # Apply SpaceMouse wrist pose if available.
        if self._wrist_reader is not None:
            wrist_pose = self._wrist_reader.get_latest_wrist_pose()
            if wrist_pose is not None:
                self.hand_interface.set_hand_transform(
                    wrist_pose, tgt_key=self._wrist_body_name
                )

        self.idx += 1
        self.hand_interface.step()
        return False
