from dexworld.scenes import HumanHandViewScene
from dexworld.types import HandDataset, Simulator
from pathlib import Path
import tyro


def main(
    base_path: str,
    hand_dataset: str,
    simulator: str,
    num_episodes: int = 1,
    episode_id: str | None = None,
):
    base_path = Path(base_path)
    hand_dataset = HandDataset(hand_dataset)
    simulator = Simulator(simulator)
    scene = HumanHandViewScene(
        base_path, hand_dataset, simulator, num_episodes, episode_id
    )

    for _ in range(10000):
        scene.step()


if __name__ == "__main__":
    tyro.cli(main)
