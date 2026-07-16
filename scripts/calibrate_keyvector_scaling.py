from dexworld.data_sources import HandDatasetReader
from dexworld.retargeting.online import KeyvectorRetargeter
from dexworld.types import HandDataset, Chirality
from dexworld.hand_models import ShadowHandModel

import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
from pathlib import Path


@hydra.main(
    config_path="../config", config_name="offline_retargeting", version_base="1.2"
)
def main(cfg):
    hand_path = Path(__file__).parent.parent / "assets" / "mjcf" / cfg.hand_type
    data_base_path = Path(hydra.utils.to_absolute_path(cfg.data_base_path)).resolve()
    # Load shadow hand model
    shadow_hand_model = ShadowHandModel(hand_path, Chirality.RIGHT)

    # Load human hand data
    human_hand_dataset = HandDatasetReader(
        data_base_path,
        HandDataset.WILOR_TEST_LONG,
        num_episodes=1,
        # episode_id="mf_mcp.jpg",
    )
    human_hand_model = human_hand_dataset.hand_model
    data = next(human_hand_dataset.get_episode_iter())

    # Retarget human hand data to shadow hand model
    retargeter_cfg = OmegaConf.to_container(cfg.retargeter.config, resolve=True)
    retargeter = KeyvectorRetargeter(
        human_hand_dataset.hand_model,
        shadow_hand_model,
        **{
            **retargeter_cfg,
            "debug_mode": cfg.debug_mode,
            "debug_every_n_frames": cfg.debug_every_n_frames,
        },
    )
    shadow_hand_data = retargeter.retarget(data["joints"])

    human_keyvectors = human_hand_model.from_joints(data["joints"]).to_keyvectors()
    human_keyvector_lengths = {
        name: np.linalg.norm(np.asarray(keyvector), axis=1)
        for name, keyvector in human_keyvectors.items()
    }

    robot_keyvectors = shadow_hand_model.compute_keyvectors(
        np.asarray(shadow_hand_data), joint_space="ctrl"
    )
    robot_keyvector_lengths = {
        name: np.linalg.norm(np.asarray(keyvector), axis=1)
        for name, keyvector in robot_keyvectors.items()
    }

    keyvector_ratios = {}
    keyvectors_cfg = cfg.retargeter.config.keyvectors_cfg
    for keyvector_cfg in keyvectors_cfg:
        name = keyvector_cfg["name"]
        src_key = keyvector_cfg["src_key"]
        tgt_key = keyvector_cfg["tgt_key"]
        keyvector_ratios[name] = (
            human_keyvector_lengths[src_key] / robot_keyvector_lengths[tgt_key]
        )

    # Plot selected ratios
    interesting_ratios = [
        "palm_to_thumb",
        "palm_to_index",
        "palm_to_middle",
        "palm_to_ring",
        "palm_to_pinky",
    ]

    # Plot joint angles
    fig, axs = plt.subplots(len(interesting_ratios), figsize=(15, 6))
    for i, ratio_name in enumerate(interesting_ratios):
        axs[i].plot(keyvector_ratios[ratio_name])
        axs[i].set_title(ratio_name)
        axs[i].set_xlabel("Frame")
        axs[i].set_ylabel("Ratio")
        axs[i].grid(True)
        # axs[i].set_ylim(0.5, 1.5)
        axs[i].set_xlim(0, len(keyvector_ratios[ratio_name]))
        axs[i].set_xticks(range(0, len(keyvector_ratios[ratio_name]), 100))
        axs[i].set_xticklabels(range(0, len(keyvector_ratios[ratio_name]), 100))

    plt.show()

    # idx =     # plt.show()
    # plt.close()
    # fig.tight_layout()
    # fig.savefig("joint_angles.pdf")
    # plt.close()


if __name__ == "__main__":
    main()
