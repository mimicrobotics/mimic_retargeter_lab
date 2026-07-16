# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sapien.core as sapien
from sapien.utils import Viewer
from geort.utils.config_utils import get_config
from geort.utils.hand_utils import (
    get_entity_by_name,
    get_active_joints,
    get_active_joint_indices,
)


class HandKinematicModel:
    def __init__(
        self,
        scene=None,
        render=False,
        hand=None,
        hand_urdf="",
        n_hand_dof=16,
        base_link="base_link",
        joint_names=[],
        # Ideally, these two guys (PD controller args) shouldn't be here.
        # -- There should be a controller class. I leave them here for code simplicity (maybe truth: or because I am lazy).
        # If you see your hand model doing something weird (in the simulation viewer below), tune them.
        kp=400.0,
        kd=10,
    ):

        self.engine = None
        if scene is None:
            engine = sapien.Engine()

            if render:
                renderer = sapien.VulkanRenderer()
                engine.set_renderer(renderer)
                print("Enable Render Mode.")
            else:
                renderer = None
            scene_config = sapien.SceneConfig()
            scene_config.default_dynamic_friction = 1.0
            scene_config.default_static_friction = 1.0
            scene_config.default_restitution = 0.00
            scene_config.contact_offset = 0.02
            scene_config.enable_pcm = False
            scene_config.solver_iterations = 25
            scene_config.solver_velocity_iterations = 1
            scene = engine.create_scene(scene_config)
            self.engine = engine

        self.scene = scene
        self.renderer = renderer

        if hand is not None:
            self.hand = hand

        else:
            loader = scene.create_urdf_loader()
            self.hand = loader.load(hand_urdf)
            init_position = [0, 0, 0.5]  # xyz, allegro [0.0, 0.0, 0.35]
            init_orientation = [1.0, 0, 0, 0]  # wxyz, allegro [0.695, 0, -0.718, 0]
            self.hand.set_root_pose(sapien.Pose(init_position, init_orientation))

        self.pmodel = self.hand.create_pinocchio_model()

        # Setup hand base link.
        self.base_link = get_entity_by_name(self.hand.get_links(), base_link)
        self.base_link_idx = self.hand.get_links().index(self.base_link)

        # Setup hand dofs.
        self.all_joints = get_active_joints(self.hand, joint_names)
        all_limits = [joint.get_limits() for joint in self.all_joints]

        self.joint_names = joint_names

        # Create user to sim index mapping
        self.user_idx_to_sim_idx = get_active_joint_indices(self.hand, joint_names)
        print("User-to-Sim Joint", self.user_idx_to_sim_idx)

        # Create sim to user index mapping (for active joints)
        self.n_sim_dof = len(self.hand.get_active_joints())
        self.sim_idx_to_user_idx = [-1] * self.n_sim_dof
        for user_idx, sim_idx in enumerate(self.user_idx_to_sim_idx):
            self.sim_idx_to_user_idx[sim_idx] = user_idx
        # self.sim_idx_to_user_idx = [self.user_idx_to_sim_idx.index(i) for i in range(len(self.user_idx_to_sim_idx))]
        print("Sim-to-User Joint", self.sim_idx_to_user_idx)

        # Joint couplings (mapping in the sim indexes)
        self._coupling_joint_map = {}
        self._add_joint_coupling("index_mp2dp", "index_pp2mp", 1.0, 0.0)
        self._add_joint_coupling("middle_mp2dp", "middle_pp2mp", 1.0, 0.0)
        self._add_joint_coupling("ring_mp2dp", "ring_pp2mp", 1.0, 0.0)
        self._add_joint_coupling("pinky_mp2dp", "pinky_pp2mp", 1.0, 0.0)

        self.joint_lower_limit = np.array(
            [limit[0][0] for limit in all_limits]
        )  # this is in user specified "joint_name" order
        self.joint_upper_limit = np.array(
            [limit[0][1] for limit in all_limits]
        )  # this is in user specified "joint_name" order
        print(self.joint_lower_limit, self.joint_upper_limit)

        init_qpos = self.convert_user_order_to_sim_order(
            (self.joint_lower_limit + self.joint_upper_limit) / 2
        )
        self.hand.set_qpos(init_qpos)
        self.hand.set_qvel(0.0 * init_qpos)
        self.qpos_target = init_qpos

        for i, joint in enumerate(self.all_joints):
            print(
                i,
                self.joint_names[i],
                joint,
                self.joint_lower_limit[i],
                self.joint_upper_limit[i],
            )
            joint.set_drive_property(kp, kd, force_limit=10)

    def __del__(self):
        del self.engine
        del self.scene

    def get_n_dof(self):
        """
        number of dof.
        """
        return len(self.joint_lower_limit)

    def get_joint_limit(self):
        """
        Get the hand joint limit.
        """
        return self.joint_lower_limit, self.joint_upper_limit

    def initialize_keypoint(self, keypoint_link_names, keypoint_offsets):
        """
        Setup keypoints to track.
        """
        keypoint_links = [
            get_entity_by_name(self.hand.get_links(), link)
            for link in keypoint_link_names
        ]
        print(keypoint_links)

        keypoint_links_id_dict = {
            link_name: (self.hand.get_links().index(keypoint_links[i]), i)
            for i, link_name in enumerate(keypoint_link_names)
        }
        self.keypoint_links = keypoint_links
        self.keypoint_links_id_dict = keypoint_links_id_dict
        self.keypoint_offsets = np.array(keypoint_offsets)

    def convert_user_order_to_sim_order(self, qpos_user):
        """
        Map user-ordered qpos (len = len(self.user_idx_to_sim_idx))
        into sim-ordered qpos (len = self.n_sim_dof), which is what SAPIEN uses.
        Coupled joints are filled based on their parent joints.
        """
        assert qpos_user.shape[0] == len(self.user_idx_to_sim_idx), (
            f"Expected {len(self.user_idx_to_sim_idx)}-dof qpos, got {qpos_user.shape[0]}"
        )

        # Initialize sim qpos to zeros
        qpos_sim = np.zeros(self.n_sim_dof)

        # Fill primary (user-controlled) joints
        for user_i, sim_i in enumerate(self.user_idx_to_sim_idx):
            qpos_sim[sim_i] = qpos_user[user_i]

        # Apply couplings
        for child_sim_idx, (
            parent_sim_idx,
            multiplier,
            offset,
        ) in self._coupling_joint_map.items():
            qpos_sim[child_sim_idx] = qpos_sim[parent_sim_idx] * multiplier + offset

        return qpos_sim

    def keypoint_from_qpos(self, qpos, ret_vec=False):
        """
        Get keypoints from hand qpos. qpos is specified using the user order.
        """
        qpos = self.convert_user_order_to_sim_order(qpos)
        self.pmodel.compute_forward_kinematics(qpos)
        base_pose = self.pmodel.get_link_pose(self.base_link_idx)

        result = {}
        vec_result = []

        for m, (link_idx, i) in self.keypoint_links_id_dict.items():
            pose = self.pmodel.get_link_pose(link_idx)
            new_pose = sapien.Pose(
                p=pose.p
                + (
                    pose.to_transformation_matrix()[:3, :3]
                    @ self.keypoint_offsets[i].reshape(3, 1)
                ).reshape(-1),
                q=pose.q,
            )

            x = (base_pose.inv() * new_pose).p  # convert to hand base frame.
            vec_result.append(x)
            result[m] = x

        if ret_vec:
            return np.array(vec_result)
        return result

    @staticmethod
    def build_from_config(config, **kwargs):
        """
        Build a kinematic model from user config.
        """
        render = kwargs.get("render", False)
        urdf_path = config["urdf_path"]
        n_hand_dof = len(config["joint_order"])
        base_link = config["base_link"]
        joint_order = config["joint_order"]

        model = HandKinematicModel(
            hand_urdf=urdf_path,
            render=render,
            n_hand_dof=n_hand_dof,
            base_link=base_link,
            joint_names=joint_order,
        )
        return model

    def get_viewer_env(self):
        return HandViewerEnv(self)

    def get_scene(self):
        return self.scene

    def get_renderer(self):
        return self.renderer

    def set_qpos_target(self, qpos_user: np.ndarray) -> None:
        """
        This function is only used during visualization.
        qpos_user is in user indexing order.
        We directly set the articulation qpos in sim order.
        """
        qpos_user = np.clip(
            qpos_user,
            self.joint_lower_limit + 1e-3,
            self.joint_upper_limit - 1e-3,
        )

        self.qpos_target = qpos_user
        qpos_sim = self.convert_user_order_to_sim_order(qpos_user)
        self.hand.set_qpos(qpos_sim)

        # Avoids residual velocities
        self.hand.set_qvel(np.zeros_like(qpos_sim))

    def _add_joint_coupling(
        self,
        child_joint_name: str,
        parent_joint_name: str,
        multiplier: float,
        offset: float,
    ):
        """
        Add a coupling relationship between two joints in the hand model. This is for the sim
        joint indices.
        """
        child_joint_idx_sim = get_active_joint_indices(self.hand, [child_joint_name])[0]
        parent_joint_idx_sim = get_active_joint_indices(self.hand, [parent_joint_name])[
            0
        ]
        self._coupling_joint_map[child_joint_idx_sim] = (
            parent_joint_idx_sim,
            multiplier,
            offset,
        )


class HandViewerEnv:
    def __init__(self, model):
        scene = model.get_scene()
        scene.set_timestep(1 / 50.0)
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        scene.add_ground(altitude=0)

        viewer = Viewer(model.get_renderer())
        viewer.set_scene(scene)
        viewer.window.set_camera_position([0.1550926, -0.1623763, 0.85])
        viewer.window.set_camera_rotation([0.8716827, 0.3260138, 0.12817779, 0.3427167])
        viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

        self.model = model
        self.scene = scene
        self.viewer = viewer

        self.render_scene = scene.get_renderer_scene()
        self.debug_links = [
            "thumb_fingertip",
            "index_fingertip",
            "middle_fingertip",
            "ring_fingertip",
            "pinky_fingertip",
        ]

        # Create a small marker actor for each fingertip
        self.tip_markers = {}
        for name in self.debug_links:
            builder = scene.create_actor_builder()
            builder.add_sphere_visual(radius=0.005, color=[1, 0, 0])  # just visual
            marker = builder.build_static(name=f"{name}_marker")
            self.tip_markers[name] = marker

    def update(self):
        self.scene.step()
        self.scene.update_render()

        # For the debug markers on fingertips
        links = self.model.hand.get_links()
        for name, marker in self.tip_markers.items():
            link = get_entity_by_name(links, name)
            pose = link.get_pose()
            marker.set_pose(pose)

        self.viewer.render()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", type=str, default="allegro")

    args = parser.parse_args()

    # Load Hand Model
    config = get_config(args.hand)
    model = HandKinematicModel.build_from_config(config, render=True)
    viewer_env = model.get_viewer_env()

    # Control Loop
    n_dof = model.get_n_dof()
    dof_lower, dof_upper = model.get_joint_limit()

    finger_to_sweep = "pinky"
    joint_to_sweep_indx = model.joint_names.index(f"{finger_to_sweep}_pp2mp")

    # Building the sweep
    num_points = 150
    sweep = np.linspace(
        dof_lower[joint_to_sweep_indx], dof_upper[joint_to_sweep_indx], num_points
    )
    sweep = np.concatenate([sweep, sweep[::-1]])

    qpos_zero = dof_lower

    step = 0
    while True:
        viewer_env.update()

        target_qpos = qpos_zero.copy()
        target_qpos[joint_to_sweep_indx] = sweep[step % len(sweep)]
        model.set_qpos_target(target_qpos)

        q_sim = model.hand.get_qpos()
        step += 1
