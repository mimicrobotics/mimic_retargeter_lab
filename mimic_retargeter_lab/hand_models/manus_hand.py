"""Hand model for the MANUS Metagloves Pro.

Maps the 25 MANUS skeleton nodes directly to HandLandmark entries.
Unlike ManoKeypointHandModel (which takes a 21-point MANO cloud and runs
inverse/forward kinematics), this model works with the raw 25-point
skeleton from the gloves — no MANO indirection needed since the gloves
provide real joint positions and orientations.

MANUS 25-node topology (confirmed via live visualization):
    0        = wrist
    1-4      = thumb  (1=base, 2=mid, 3=dp, 4=tip)
    5-9      = index  (5=MCP, 6=PIP, 7=DIP, 8=DP, 9=tip)
    10-14    = middle (10=MCP, 11=PIP, 12=DIP, 13=DP, 14=tip)
    15-19    = ring   (15=MCP, 16=PIP, 17=DIP, 18=DP, 19=tip)
    20-24    = pinky  (20=MCP, 21=PIP, 22=DIP, 23=DP, 24=tip)
"""

import numpy as np

import mimic_retargeter_lab.utils.human_hands.mano_utils as mano_utils
from mimic_retargeter_lab.types.types import Chirality, HandLandmark
from mimic_retargeter_lab.utils import (
    HandKinematicsForward,
    HandKinematicsInverse,
    extract_joint_angles,
)

from .human_hand_base import HumanHandModel


class ManusHandModel(HumanHandModel):
    """Human hand model backed by MANUS Metagloves Pro skeleton data."""

    NUM_NODES = 25

    # Full 25-node finger-chain link topology for MuJoCo viz.
    #
    # Node ordering (confirmed via live viz):
    #   0        = wrist
    #   1-4      = thumb  (1=base, 4=tip)
    #   5-9      = index  (5=MCP, 9=tip)
    #   10-14    = middle (10=MCP, 14=tip)
    #   15-19    = ring   (15=MCP, 19=tip)
    #   20-24    = pinky  (20=MCP, 24=tip)
    HAND_LINKS_25 = [
        # Wrist → finger bases
        (0, 1),
        (0, 5),
        (0, 10),
        (0, 15),
        (0, 20),
        # Palm cross-links (MCP → MCP)
        (5, 10),
        (10, 15),
        (15, 20),
        # Thumb: 1→2→3→4
        (1, 2),
        (2, 3),
        (3, 4),
        # Index: 5→6→7→8→9
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        # Middle: 10→11→12→13→14
        (10, 11),
        (11, 12),
        (12, 13),
        (13, 14),
        # Ring: 15→16→17→18→19
        (15, 16),
        (16, 17),
        (17, 18),
        (18, 19),
        # Pinky: 20→21→22→23→24
        (20, 21),
        (21, 22),
        (22, 23),
        (23, 24),
    ]

    # MANUS node index → HandLandmark.
    #
    # Thumb has 4 nodes (no MCP), 5-joint fingers skip MCP (first node).
    _MANUS_LANDMARK_MAP: dict[int, HandLandmark] = {
        0: HandLandmark.WRIST,
        # Thumb (4 nodes: 1-4)
        1: HandLandmark.THUMB_BASE,
        # 2 = thumb mid — no HandLandmark equivalent, skipped
        3: HandLandmark.THUMB_DP,
        4: HandLandmark.THUMB_TIP,
        # Index (5 nodes: 5-9, skip MCP=5)
        6: HandLandmark.INDEX_BASE,
        # 7 = index mid — skipped
        8: HandLandmark.INDEX_DP,
        9: HandLandmark.INDEX_TIP,
        # Middle (5 nodes: 10-14, skip MCP=10)
        11: HandLandmark.MIDDLE_BASE,
        # 12 = middle mid — skipped
        13: HandLandmark.MIDDLE_DP,
        14: HandLandmark.MIDDLE_TIP,
        # Ring (5 nodes: 15-19, skip MCP=15)
        16: HandLandmark.RING_BASE,
        # 17 = ring mid — skipped
        18: HandLandmark.RING_DP,
        19: HandLandmark.RING_TIP,
        # Pinky (5 nodes: 20-24, skip MCP=20)
        21: HandLandmark.PINKY_BASE,
        # 22 = pinky mid — skipped
        23: HandLandmark.PINKY_DP,
        24: HandLandmark.PINKY_TIP,
    }

    # Reverse: HandLandmark → MANUS node index (for get_landmarks lookup).
    _LANDMARK_TO_NODE: dict[HandLandmark, int] = {
        v: k for k, v in _MANUS_LANDMARK_MAP.items()
    }

    # Number of raw nodes from MANUS.
    _NUM_MANUS_NODES = 25

    # MANUS 25 → MANO 21 reindexing for joint-angle extraction.
    # MANUS gives 5 nodes per non-thumb finger (MCP, PIP, DIP, DP, TIP);
    # MANO's kinematic tree expects 4 (MCP, proximal, distal, tip). We drop
    # the intermediate DIP node and feed the remainder through the MANO
    # IK/FK pipeline that powers ManoKeypointHandModel.to_joint_angles.
    _MANUS_TO_MANO21_INDICES = (
        0,  # wrist
        1,
        2,
        3,
        4,  # thumb (MANUS already 4-node)
        6,
        7,
        8,
        9,  # index:  MCP, PIP, DP, TIP  (drop DIP=7)
        11,
        12,
        13,
        14,  # middle: MCP, PIP, DP, TIP  (drop DIP=12)
        16,
        17,
        18,
        19,  # ring:   MCP, PIP, DP, TIP  (drop DIP=17)
        21,
        22,
        23,
        24,  # pinky:  MCP, PIP, DP, TIP  (drop DIP=22)
    )

    def __init__(self, chirality: Chirality = Chirality.RIGHT):
        super().__init__(chirality)
        self.num_fingertips = 5
        self._inverse_kinematics = HandKinematicsInverse(
            rotation_representation="matrix"
        )
        self._forward_kinematics = HandKinematicsForward(
            rotation_representation="matrix"
        )

    def get_num_fingertips(self) -> int:
        return self.num_fingertips

    def get_landmarks(
        self, joints_3d: np.ndarray, **kwargs
    ) -> dict[HandLandmark, np.ndarray]:
        """Extract named landmarks from a 25-point MANUS skeleton.

        Args:
            joints_3d: Array of shape (..., 25, 3) — the raw MANUS node
                positions as received from the ZMQ bridge.

        Returns:
            Dict mapping each available HandLandmark to its position array
            with shape (..., 3).
        """
        pts = np.asarray(joints_3d)
        return {
            landmark: pts[..., node_idx, :]
            for landmark, node_idx in self._LANDMARK_TO_NODE.items()
        }

    def get_landmark_transforms(
        self, joints_3d: np.ndarray
    ) -> dict[HandLandmark, np.ndarray]:
        """Return 4x4 transforms for configured landmarks.

        Since MANUS provides only positions (not orientations), each
        transform has identity rotation and the node position as translation.

        Parameters
        ----------
        joints_3d : (25, 3) or (N, 25, 3) array

        Returns
        -------
        dict mapping HandLandmark to (4, 4) or (N, 4, 4) ndarray.
        """
        pts = np.asarray(joints_3d, dtype=np.float32)
        is_single = pts.ndim == 2
        if is_single:
            pts = pts[None]

        result: dict[HandLandmark, np.ndarray] = {}
        for landmark, node_idx in self._LANDMARK_TO_NODE.items():
            T = np.tile(np.eye(4, dtype=np.float32), (pts.shape[0], 1, 1))
            T[:, :3, 3] = pts[:, node_idx, :]
            result[landmark] = T[0] if is_single else T
        return result

    def get_qpos_joint_names(self) -> list[str]:
        """MANUS does not use a parameterised pose space."""
        return []

    def _normalize_joints(self, joints_3d: np.ndarray) -> np.ndarray:
        # MANUS skeleton is already in a sane frame; identity passthrough so the
        # metric pipeline's defensive call sites (e.g. MANO mesh fitting) don't
        # trip on AttributeError if they ever re-enable that branch.
        return np.asarray(joints_3d)

    def to_joint_angles(self, joints_3d: np.ndarray) -> dict[str, np.ndarray]:
        """Extract MANO-style named joint angles from a 25-point MANUS skeleton.

        Returns a dict keyed by ``LOCAL_JOINT_DOFS`` entries (e.g.
        ``"wrist.x"``, ``"th_proximal.z"``, ``"ff_distal.x"``) — the same
        contract ``ManoKeypointHandModel.to_joint_angles`` exposes, so
        ``JointAngleRetargeter`` (and the hybrid retargeter) can consume
        MANUS data the same way it consumes MANO data.

        Pipeline: collapse the 25-point MANUS cloud to a MANO-equivalent
        21-point cloud (drop the intermediate DIP nodes), normalize into
        MANO's canonical hand frame, run inverse + forward kinematics to
        recover per-joint local rotations, and pass the resulting frame
        dict through ``extract_joint_angles``.
        """
        pts = np.asarray(joints_3d, dtype=np.float32)
        is_single = pts.ndim == 2
        if is_single:
            pts = pts[None]

        mano21 = pts[:, self._MANUS_TO_MANO21_INDICES, :]

        normalized = np.stack(
            [
                mano_utils.normalize_points(
                    f.copy(),
                    flip_x_axis=False,
                    flip_y_axis=True,
                    add_z_rotation=np.pi / 8,
                )
                for f in mano21
            ]
        ).astype(np.float32)

        local_repr = self._inverse_kinematics(normalized)

        batch_size = local_repr.shape[0]
        dummy_root_pos = np.zeros((batch_size, 3), dtype=np.float32)
        _, local_rotation_matrices = self._forward_kinematics(
            local_repr, dummy_root_pos
        )

        eye4 = np.eye(4, dtype=np.float32)
        local_kin_tree: dict[str, np.ndarray] = {}
        for i, joint_name in enumerate(mano_utils.JOINT_NAMES):
            T = np.broadcast_to(eye4[None], (batch_size, 4, 4)).copy()
            T[:, :3, :3] = local_rotation_matrices[:, i]
            local_kin_tree[joint_name] = T

        joint_angles = extract_joint_angles(local_kin_tree)

        if is_single:
            return {k: v[0] for k, v in joint_angles.items()}
        return joint_angles

    def to_kinematic_tree(
        self,
        joints_3d: np.ndarray,
        return_frame_dict: bool = False,
    ) -> tuple[np.ndarray | dict, list]:
        """Build 4x4 frames from the 25-point MANUS skeleton.

        Since MANUS provides only positions (not orientations) in the
        joints_3d point cloud, we construct frames with identity rotation
        and the node position as translation. For orientation-aware frames,
        use the full JSON data from the ZMQ bridge (which includes quaternions).

        Args:
            joints_3d: Array of shape (25, 3) or (B, 25, 3).
            return_frame_dict: If True, return a dict keyed by joint name
                strings; otherwise return a stacked (B, N, 4, 4) array.

        Returns:
            (frames, links) — frames as dict or array, links as list of
            (start_pos, end_pos) tuples for debug drawing.
        """
        pts = np.asarray(joints_3d, dtype=np.float32)
        is_single = pts.ndim == 2
        if is_single:
            pts = pts[None]

        batch_size = pts.shape[0]

        # Build 4x4 identity frames with positions.
        frames = np.tile(
            np.eye(4, dtype=np.float32), (batch_size, self._NUM_MANUS_NODES, 1, 1)
        )
        frames[:, :, :3, 3] = pts

        # Build links along the finger chains for debug visualisation.
        # Each finger chain is a sequence of MANUS node indices.
        finger_chains = [
            [0, 1, 2, 3, 4],  # wrist → thumb
            [0, 5, 6, 7, 8, 9],  # wrist → index
            [0, 10, 11, 12, 13, 14],  # wrist → middle
            [0, 15, 16, 17, 18, 19],  # wrist → ring
            [0, 20, 21, 22, 23, 24],  # wrist → pinky
        ]
        links = []
        for chain in finger_chains:
            for k in range(len(chain) - 1):
                start = pts[:, chain[k]]
                end = pts[:, chain[k + 1]]
                if is_single:
                    links.append((start[0], end[0]))
                else:
                    links.append((start, end))

        if return_frame_dict:
            frame_dict = {}
            for node_idx, landmark in self._MANUS_LANDMARK_MAP.items():
                key = landmark.value
                f = frames[:, node_idx]
                frame_dict[key] = f[0] if is_single else f
            return frame_dict, links

        frames_out = frames[0] if is_single else frames
        return frames_out, links
