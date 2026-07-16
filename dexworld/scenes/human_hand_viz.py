from pathlib import Path
import time

from dexworld.types import Simulator, HandDataset
from dexworld.data_sources import HandDatasetReader
from dexworld.interfaces import MujocoHumanHandInterface


class HumanHandViewScene:
    def __init__(
        self,
        base_path: Path,
        hand_dataset: HandDataset,
        simulator: Simulator,
        num_episodes: int = 1,
        episode_id: str | None = None,
    ):
        self.human_hand_data_source = HandDatasetReader(
            base_path, hand_dataset, num_episodes=num_episodes, episode_id=episode_id
        )
        self.human_hand_frame_iter = self.human_hand_data_source.get_iter()
        self.idx = 0

        match simulator:
            case Simulator.MUJOCO:
                self.human_hand_interface = MujocoHumanHandInterface()
            case _:
                raise ValueError(f"Unsupported simulator: {simulator}")

    def step(self):
        try:
            hand_data_dict = next(self.human_hand_frame_iter)
            self.human_hand_interface.update_hand_frames(
                hand_data_dict["transforms"], hand_data_dict["links"]
            )
            print(f"updating frames {self.idx}")
        except StopIteration:
            self.human_hand_frame_iter = self.human_hand_data_source.get_iter()
            self.idx = 0
            hand_data_dict = next(self.human_hand_frame_iter)
            self.human_hand_interface.update_hand_frames(
                hand_data_dict["transforms"], hand_data_dict["links"]
            )

        time.sleep(0.1)

        self.idx += 1
        self.human_hand_interface.step()
