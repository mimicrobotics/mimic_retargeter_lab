from mimic_retargeter_lab.types import Chirality, Simulator, RobotHandType, Scene
from mimic_retargeter_lab.scenes import FingertipViewScene
from pathlib import Path
import os
import tyro


def main(hand_type: str, chirality: str, simulator: str, scene: str):
    hand_path = Path(__file__).parent.parent / "assets" / "mjcf" / hand_type
    os.chdir(hand_path)
    model = FingertipViewScene(
        hand_type=RobotHandType(hand_type),
        robot_hand_base_path=hand_path,
        chirality=Chirality(chirality),
        simulator_type=Simulator(simulator),
        scene=Scene(scene),
    )

    for _ in range(100000):
        model.step()


if __name__ == "__main__":
    tyro.cli(main)
