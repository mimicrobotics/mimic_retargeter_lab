"""Reader for MANUS Metagloves Pro recordings produced by
``scripts/record_manus_zmq.py``.

Each ``.npz`` file is one episode. ``data_path`` may be:
  - a directory: every ``*.npz`` in it becomes an episode (lazy-loaded);
  - a single file: that one file is the only episode;
  - a glob pattern: matched files become episodes.

Per-file schema (named arrays):
    data         : (T, N, 7) float32 — [px, py, pz, qw, qx, qy, qz] per keypoint
    timestamps   : (T,) float64
    keypoint_ids : (N,) int32
    chirality    : scalar string ("RIGHT" | "LEFT")
    quaternion_order : scalar string ("wxyz")
    glove_ids    : array of glove IDs

Yields per-episode dicts with raw joint positions only (quat channels dropped).
Pairs with ``ManusHandModel``, which expects joints in (T, 25, 3) MANUS-native
ordering.
"""

from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy.spatial.transform import Rotation

from mimic_retargeter_lab.hand_models.manus_hand import ManusHandModel
from mimic_retargeter_lab.utils.logger_utils import get_logger

from ..base_hand_data_reader import BaseHandDataReader


class ManusNpzReader(BaseHandDataReader):
    def __init__(
        self,
        data_path: Path | str,
        num_episodes: int | None = None,
        episode_id: str | None = None,
        seed: int = 42,
    ):
        super().__init__()
        self._logger = get_logger(__name__)
        self.hand_model = ManusHandModel()

        path = Path(data_path).resolve()
        if path.is_dir():
            self._files = sorted(path.glob("*.npz"))
        elif path.is_file():
            self._files = [path]
        else:
            # Treat as glob pattern relative to its parent.
            self._files = sorted(path.parent.glob(path.name))

        if not self._files:
            raise FileNotFoundError(
                f"ManusNpzReader: no .npz files found at {data_path} (resolved {path})"
            )

        if episode_id is not None:
            self._files = [f for f in self._files if f.stem == episode_id]
            if not self._files:
                raise ValueError(
                    f"ManusNpzReader: no .npz matched episode_id={episode_id!r}"
                )

        if num_episodes is not None and num_episodes < len(self._files):
            rng = np.random.default_rng(seed)
            indices = sorted(
                rng.choice(len(self._files), num_episodes, replace=False).tolist()
            )
            self._files = [self._files[i] for i in indices]

        # `data_path` is used by the metric layer for retarget-cache keys; keep
        # it as the originally-resolved root so the key stays stable across
        # episodes within one run.
        self.data_path = path

        self._logger.info(
            f"ManusNpzReader: discovered {len(self._files)} npz file(s) at {path}"
        )
        for f in self._files:
            self._logger.info(f"  - {f.name}")

    def get_episode_iter(self) -> Iterator[dict[str, Any]]:
        for f in self._files:
            archive = np.load(f, allow_pickle=False)

            raw = np.asarray(archive["data"])
            if raw.ndim != 3 or raw.shape[-1] != 7:
                self._logger.warning(
                    f"ManusNpzReader: skipping {f.name} — expected shape (T, N, 7), "
                    f"got {raw.shape}"
                )
                continue

            joints = raw[..., :3].astype(np.float32)
            chirality = (
                str(archive["chirality"]) if "chirality" in archive.files else None
            )
            quat_order = (
                str(archive["quaternion_order"])
                if "quaternion_order" in archive.files
                else "wxyz"
            )

            self._logger.info(
                f"ManusNpzReader: loaded {f.name} ({joints.shape[0]} frames, "
                f"{joints.shape[1]} keypoints, chirality={chirality}, "
                f"quat_order={quat_order})"
            )

            yield {
                "episode_id": f.stem,
                "joints": joints,
                "keyvectors": None,
                "joint_angles": None,
                "mano_pose": None,
                "mano_shape": None,
            }

    def get_iter(self) -> Iterator[dict[str, Any]]:
        """Per-frame iterator for the kinematic retargeting scene.

        Yields the same shape contract as the live ManusHandTracker:
            transforms : (25, 4, 4)
            joints     : (1, 25, 3)
            links      : []
        Quaternions in the npz are stored as (qw, qx, qy, qz); scipy
        consumes (qx, qy, qz, qw).
        """
        for f in self._files:
            archive = np.load(f, allow_pickle=False)
            raw = np.asarray(archive["data"])
            if raw.ndim != 3 or raw.shape[-1] != 7:
                self._logger.warning(
                    f"ManusNpzReader: skipping {f.name} — expected shape (T, N, 7), "
                    f"got {raw.shape}"
                )
                continue

            positions = raw[..., :3].astype(np.float32)
            quats_wxyz = raw[..., 3:].astype(np.float32)
            quats_xyzw = quats_wxyz[..., [1, 2, 3, 0]]

            num_frames, num_nodes, _ = positions.shape
            for t in range(num_frames):
                transforms = np.tile(np.eye(4, dtype=np.float32), (num_nodes, 1, 1))
                transforms[:, :3, :3] = Rotation.from_quat(quats_xyzw[t]).as_matrix()
                transforms[:, :3, 3] = positions[t]

                yield {
                    "transforms": transforms,
                    "joints": positions[t][np.newaxis],
                    "links": [],
                }
