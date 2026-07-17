"""Inspects kinematic tree of a MuJoCo model.

This script inspects the kinematic tree of a MuJoCo model
and launches a viewer to visualize the model along with
the link frames.

Usage:
    python -m tests.inspect_mjcf_kinematics --robot_hand shadow_hand --chirality right
    python -m tests.inspect_mjcf_kinematics --robot_hand mimic_p050_hand --chirality right
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from mimic_retargeter_lab.types import Chirality, RobotHandType
from mimic_retargeter_lab.utils import configure_logging, get_logger

MJCF_BASE_DIR = Path("assets/mjcf")


class MjcfKinematicsInspector:
    def __init__(self, xml_model_path: Path):
        self._logger = get_logger(self.__class__.__name__)
        resolved = xml_model_path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"MJCF file not found: {resolved}")
        self._logger.info(f"Loading MJCF model: {resolved}")
        self._model = mujoco.MjModel.from_xml_path(str(resolved))
        self._data = mujoco.MjData(self._model)

    def print_kinematic_tree(self, model: mujoco.MjModel) -> None:
        """Prints the body hierarchy to the terminal."""
        self._logger.info("\n=== Kinematic Tree Structure ===")
        for i in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            parent_id = model.body_parentid[i]
            parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)

            depth = 0
            curr = i
            while curr != 0:
                curr = model.body_parentid[curr]
                depth += 1

            indent = "  " * depth
            self._logger.info(f"{indent}└── {body_name} (Parent: {parent_name})")
        self._logger.info("================================\n")

    def launch_viewer(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            with viewer.lock():
                # 1. Make the physical meshes transparent so frames aren't hidden inside them
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1

                # 2. Render joint axes (shows up as colored cylinders/arrows)
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1

                # 3. Ensure sites (like fingertips/grasp points) are visible as 3D markers
                viewer.opt.sitegroup[:] = 1

                # 4. CRITICAL: Lock the coordinate frames to Bodies (the actual kinematic links)
                viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY

                # 5. Lock the text labels to Bodies so you can identify the tree
                viewer.opt.label = mujoco.mjtLabel.mjLABEL_BODY

            self._logger.info("Viewer launched with full kinematic tree visualization.")

            while viewer.is_running():
                step_start = time.time()
                mujoco.mj_step(model, data)
                viewer.sync()

                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data


def resolve_mjcf_path(robot_hand: RobotHandType, chirality: Chirality) -> Path:
    """Resolve the MJCF XML path from hand type and chirality.

    Prefers chirality-specific ``scene_{chirality}.xml``; falls back to
    ``scene.xml`` for symmetric hands that ship a single scene file.
    """
    hand_dir = MJCF_BASE_DIR / robot_hand.value
    per_side = hand_dir / f"scene_{chirality.value}.xml"
    if per_side.exists():
        return per_side
    symmetric = hand_dir / "scene.xml"
    if symmetric.exists():
        return symmetric
    return per_side


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MJCF kinematic tree.")
    parser.add_argument(
        "--robot_hand",
        type=str,
        required=True,
        choices=[ht.value for ht in RobotHandType],
        help="Hand model type.",
    )
    parser.add_argument(
        "--chirality",
        type=str,
        default="right",
        choices=[c.value for c in Chirality],
        help="Hand chirality (default: right).",
    )
    return parser.parse_args()


def main():
    configure_logging()
    args = parse_args()
    robot_hand = RobotHandType(args.robot_hand)
    chirality = Chirality(args.chirality)
    mjcf_path = resolve_mjcf_path(robot_hand, chirality)

    inspector = MjcfKinematicsInspector(xml_model_path=mjcf_path)
    inspector.print_kinematic_tree(inspector.model)
    inspector.launch_viewer(inspector.model, inspector.data)


if __name__ == "__main__":
    main()
