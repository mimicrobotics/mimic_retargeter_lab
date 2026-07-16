"""One-off smoke test for MjcfHandKinematicModel.

Run from repo root:
    JAX_PLATFORMS=cpu python scripts/smoke_test_mjcf_geort.py
"""

import dexworld  # noqa: F401  pins JAX_PLATFORMS

from pathlib import Path

import numpy as np

from dexworld.hand_models import create_robot_hand
from dexworld.retargeting.online.geort import MjcfHandKinematicModel
from dexworld.types import Chirality, RobotHandType


REPO_ROOT = Path(__file__).resolve().parent.parent
HAND_PATH = REPO_ROOT / "assets" / "mjcf" / "mimic_p050_hand"


def main():
    to_model = create_robot_hand(
        RobotHandType.MIMIC_P050_HAND, HAND_PATH, Chirality.RIGHT
    )

    joint_order = list(to_model.get_actuated_joint_names())
    print(f"actuated joints ({len(joint_order)}): {joint_order}")

    hand = MjcfHandKinematicModel(to_model=to_model, joint_order=joint_order)

    lo, hi = hand.get_joint_limit()
    print(
        f"limits: lower {lo.shape}, upper {hi.shape}, range examples: "
        f"{lo[0]:.3f}..{hi[0]:.3f}"
    )

    fingertip_links = [
        "thumb_fingertip",
        "index_fingertip",
        "middle_fingertip",
        "ring_fingertip",
        "pinky_fingertip",
    ]
    hand.initialize_keypoint(fingertip_links, [[0.0, 0.0, 0.0]] * 5)

    rng = np.random.default_rng(0)
    qpos = rng.uniform(lo, hi).astype(np.float32)
    print(f"random qpos: {qpos}")

    fk = hand.keypoint_from_qpos(qpos)
    for name, pos in fk.items():
        print(f"  {name}: {pos}")

    fk_neutral = hand.keypoint_from_qpos(np.zeros_like(qpos))
    for name, pos in fk_neutral.items():
        print(f"  {name} (neutral): {pos}")


if __name__ == "__main__":
    main()
