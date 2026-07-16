import pickle
import numpy as np
from pathlib import Path
import tyro

from dexworld.utils import ManoPreprocessor
from dexworld.utils import HumanDataWriter

"""

Extract data from WiLoR outputs saved as pkl.
Assumes only one right hand is visible per frame.

Structure:
- data is a list of lists for all timesteps:
    - Each list represents a frame, and contains a list of output dicts
    - Each dictionary contains the following keys:
        - 'hand_bbox', 4 image-space coordinates
        - is_right: True or False
        - 'wilor_preds': dict
            - 'global_orient'
            - 'hand_pose', (1, 15, 3)
            - 'betas', (1, 10)
            - 'pred_cam'
            - 'pred_keypoints_3d', (1, 21, 3)
            - 'pred_keypoints_2d'
            - 'pred_vertices'
            - 'pred_cam_t_full'
            - 'scaled_focal_length'
"""


def main(data_path: str, out_dir: str, out_dataset_name: str):
    preprocessor = ManoPreprocessor()
    writer = HumanDataWriter(Path(out_dir), out_dataset_name)
    with open(data_path, "rb") as f:
        data = pickle.load(f)
        print(f"Found sequence with {len(data)} elements")
        for episode_name, episode_data in data.items():
            print(f"Processing episode: {episode_name}")
            joints = []
            for frame in episode_data:
                for pred in frame:
                    pred_joints = pred["wilor_preds"]["pred_keypoints_3d"]
                    if not pred["is_right"]:
                        continue
                    joints.append(pred_joints)
            processed = preprocessor.convert_from_joints(
                np.array(joints).squeeze(1), add_normalization=True
            )
            print(f"Writing episode with keys {processed.keys()}")
            writer.write_episode(str(episode_name).replace("/", "_"), processed)


if __name__ == "__main__":
    tyro.cli(main)
