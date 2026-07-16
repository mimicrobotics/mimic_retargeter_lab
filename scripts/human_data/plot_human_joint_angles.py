from dexworld.data_sources import HandDatasetReader
from dexworld.types import HandDataset, Chirality
from dexworld.retargeting import JointAngleRetargeter
from dexworld.hand_models import ShadowHandModel

import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
from pathlib import Path


@hydra.main(
    config_path="../../config", config_name="offline_retargeting", version_base="1.2"
)
def main(cfg):
    hand_path = Path(__file__).parent.parent.parent / "assets" / "mjcf" / cfg.hand_type
    data_base_path = Path(hydra.utils.to_absolute_path(cfg.data_base_path)).resolve()
    # Load shadow hand model
    shadow_hand_model = ShadowHandModel(hand_path, Chirality.RIGHT)

    # Load human hand data
    human_hand_dataset = HandDatasetReader(
        data_base_path,
        HandDataset.WILOR_TEST_INDEX_MCP,
        num_episodes=1,
        # episode_id="mf_mcp.jpg",
    )
    human_hand_model = human_hand_dataset.hand_model
    human_hand_dataset.get_iter()
    # aggregate data
    data = next(human_hand_dataset.get_episode_iter())
    # for frame in tqdm(human_hand_frame_iter):
    #     for k, v in frame.items():
    #         if k not in data:
    #             if isinstance(v, np.ndarray):
    #                 data[k] = []
    #             elif isinstance(v, dict):
    #                 data[k] = {}
    #             else:
    #                 raise ValueError(f"Unsupported data type: {type(v)}")
    #         if isinstance(v, np.ndarray):
    #             data[k].append(v)
    #         elif isinstance(v, dict):
    #             for kk, vv in v.items():
    #                 if kk not in data[k]:
    #                     data[k][kk] = []
    #                 data[k][kk].append(vv)

    # for k, v in data.items():
    #     if isinstance(v, dict):
    #         for kk, vv in v.items():
    #             if kk not in data:
    #                 data[kk] = []
    #             data[kk].append(vv)
    #     else:
    #         data[k] = np.array(v)

    # for k, v in data.items():
    #     if isinstance(v, dict):
    #         for kk, vv in v.items():
    #             if len(vv.shape) > 1 and vv.shape[1] == 1:
    #                 data[k][kk] = vv.squeeze(1)
    #             print(f"{k}.{kk}: {data[k][kk]}")
    #     else:
    #         if len(v.shape) > 1 and v.shape[1] == 1:
    #             data[k] = v.squeeze(1)
    #         print(f"{k}: {data[k].shape}")

    # Retarget human hand data to shadow hand model
    retargeter = JointAngleRetargeter(
        human_hand_dataset.hand_model,
        shadow_hand_model,
        **OmegaConf.to_container(cfg.retargeter.config),
    )
    shadow_hand_data = retargeter.retarget(data["joints"])
    print(f"shadow hand data shape: {shadow_hand_data.shape}")

    # joint_mapping = retargeter.joint_mapping

    # Plot joint angles
    interesting_joint_angles = ["ff_proximal.x", "ff_distal.x", "ff_tip.x"]
    interesting_shadow_joint_angles = ["rh_A_FFJ3", "rh_A_FFJ0"]
    fig, axs = plt.subplots(2, 3, figsize=(15, 6))
    idx = 0
    for i, joint in enumerate(human_hand_model.get_qpos_joint_names()):
        if joint not in interesting_joint_angles:
            continue
        axs[0, idx].plot(data["joint_angles"][joint])
        axs[0, idx].set_title(f"Human {joint}")
        idx += 1

    idx = 0
    for i, joint in enumerate(shadow_hand_model.get_actuated_joint_names()):
        # plot_idx = i + len(human_hand_model.get_qpos_joint_names())
        if joint not in interesting_shadow_joint_angles:
            continue
        axs[1, idx].plot(shadow_hand_data[:, i])
        axs[1, idx].set_title(f"Shadow {joint}")
        idx += 1
    plt.show()
    plt.close()
    # fig.tight_layout()
    # fig.savefig("joint_angles.pdf")
    # plt.close()


if __name__ == "__main__":
    main()
