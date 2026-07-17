from pathlib import Path
import numpy as np
from mimic_retargeter_lab.hand_models import create_robot_hand
from mimic_retargeter_lab.types import Chirality, Simulator, RobotHandType, Scene


class FingertipViewScene:
    def __init__(
        self,
        hand_type: RobotHandType,
        robot_hand_base_path: Path,
        chirality: Chirality,
        simulator_type: Simulator,
        scene: Scene,
    ):
        self.hand_model = create_robot_hand(hand_type, robot_hand_base_path, chirality)

        match simulator_type:
            case Simulator.MUJOCO:
                from mimic_retargeter_lab.interfaces import MujocoHandInterface

                self.hand_interface = MujocoHandInterface(self.hand_model)
            case _:
                raise ValueError(f"Unsupported simulator type: {simulator_type}")

    def get_joint_angles(self) -> np.ndarray:
        return self.hand_interface.get_joint_angles()

    def set_joint_angles(self, joint_angles: np.ndarray):
        self.hand_interface.set_joint_angles(joint_angles)

    def step(self):
        joint_angles = self.hand_interface.get_joint_angles()
        fingertip_frames = self.hand_model.from_qpos_joint_angles(
            joint_angles
        ).to_fingertips()
        self.hand_interface.update_fingertip_frames(fingertip_frames)

        self.hand_interface.step()
