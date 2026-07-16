"""GeoRT-based online retargeter.

Wraps a pre-trained GeoRT IKModel and exposes it through the
BaseOnlineRetargeter interface. Loads checkpoints from the dexworld-owned
`<repo_root>/checkpoints/geort/<tag>/` directory containing
`config.json` and `last.pth` — the upstream package at `third_party/geort/`
is only used for its inference classes (IKModel, HandFormatter,
GeoRTRetargetingModel), not its checkpoint root.

Note: GeoRT's IKModel hardcodes `.cuda()` at load and forward
(see `third_party/geort/geort/export.py`). On CPU-only hosts construction
will fail with a torch CUDA error.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np

from dexworld.hand_models import HumanHandModel, RobotHandModel
from dexworld.types import HandLandmark
from dexworld.utils.retarget_utils import align_pcloud_kabsch_umeyama

from .base_online_retargeter import BaseOnlineRetargeter


# This file lives at dexworld/retargeting/online/geort_retargeter.py;
# parents[3] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CHECKPOINT_ROOT = _REPO_ROOT / "checkpoints" / "geort"


def _detect_compute_device() -> types.SimpleNamespace:
    """Return a stub device with a `.platform` attr matching the convention
    used by JAX-based retargeters (`retargeter._device.platform`).

    GeoRT's IKModel runs on PyTorch CUDA (hardcoded `.cuda()` in
    `third_party/geort/geort/export.py`), so when torch CUDA is available
    we report ``cuda``. Falls back to ``cpu`` in case the user runs a
    patched build that supports it; constructing the IK MLP would still
    fail in the upstream code path, so the report is mostly informational.
    """
    try:
        import torch

        platform = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        platform = "cpu"
    return types.SimpleNamespace(platform=platform)


class GeortRetargeter(BaseOnlineRetargeter):
    """Online retargeter backed by a pre-trained GeoRT IKModel."""

    def __init__(
        self,
        from_model: HumanHandModel,
        to_model: RobotHandModel,
        wrist_mapping: dict,
        checkpoint_tag: str,
        constant_joints: dict[str, float] | None = None,
        checkpoint_root: str | Path | None = None,
        # Same convention as sampling_based / keyvector / hybrid: per-frame
        # Kabsch-Umeyama align human landmarks into the robot's current pose
        # (read from `self._qpos_prev`). When None, alignment is skipped.
        alignment_landmarks: list[str] | None = None,
        alignment_use_scale: bool = True,
    ):
        super().__init__(from_model, to_model, wrist_mapping)

        try:
            from geort.export import GeoRTRetargetingModel
            from geort.utils.config_utils import load_json
        except ModuleNotFoundError as e:
            if e.name == "geort":
                raise ModuleNotFoundError(
                    "geort is not installed. Add it as a path dep in pyproject.toml or "
                    "run `uv sync` (workspace member) / "
                    "`pip install -e third_party/geort` (path mode)."
                ) from e
            # Real culprit is a transitive dep (e.g. open3d, torch) — surface it.
            raise

        ckpt_root = (
            Path(checkpoint_root)
            if checkpoint_root is not None
            else _DEFAULT_CHECKPOINT_ROOT
        )
        ckpt_dir = self._resolve_checkpoint_dir(ckpt_root, checkpoint_tag)
        model_path = ckpt_dir / "last.pth"
        config_path = ckpt_dir / "config.json"

        self._model = GeoRTRetargetingModel(
            model_path=str(model_path), config_path=str(config_path)
        )

        config = load_json(str(config_path))
        self._geort_joint_order: list[str] = list(config["joint_order"])

        # Alignment landmarks come from the yaml (constructor kwarg), same as
        # sampling_based / keyvector. Cast strings to HandLandmark; None
        # disables per-frame Kabsch entirely.
        if alignment_landmarks is None:
            self._alignment_landmarks: list[HandLandmark] | None = None
        else:
            self._alignment_landmarks = [
                lm if isinstance(lm, HandLandmark) else HandLandmark(str(lm).lower())
                for lm in alignment_landmarks
            ]
        self._alignment_use_scale: bool = bool(alignment_use_scale)

        # Build name → actuated-index map. Accept both the actuated name
        # (e.g. "A_thumb_base2cmc") and its unprefixed qpos equivalent
        # ("thumb_base2cmc"), mirroring MjcfHandKinematicModel's convention.
        # The IK MLP outputs values in joint_order space, semantically equal
        # to actuator commands — they go straight into mj_data.ctrl by way of
        # MujocoHandInterface.set_joint_angles. MJCF tendons / equality
        # constraints handle joint coupling at sim time.
        actuated_names = list(to_model.get_actuated_joint_names())
        self._name_to_actuated_idx: dict[str, int] = {}
        for i, a_name in enumerate(actuated_names):
            self._name_to_actuated_idx[a_name] = i
            qpos_eq = to_model._joint_name_from_actuated_name(a_name)
            if qpos_eq != a_name:
                self._name_to_actuated_idx[qpos_eq] = i

        try:
            self._reorder_idx = np.array(
                [self._name_to_actuated_idx[name] for name in self._geort_joint_order],
                dtype=np.int32,
            )
        except KeyError as e:
            raise ValueError(
                f"GeoRT checkpoint joint_order references {e.args[0]!r}, which is "
                f"not in to_model.get_actuated_joint_names() (or unprefixed "
                f"equivalents). Checkpoint: {ckpt_dir}"
            ) from e

        self.constant_joints = dict(constant_joints) if constant_joints else {}
        unknown_const = [
            n for n in self.constant_joints if n not in self._name_to_actuated_idx
        ]
        if unknown_const:
            raise ValueError(
                f"constant_joints names not in actuated/qpos-equivalent names: "
                f"{unknown_const}. Known: {sorted(self._name_to_actuated_idx)}"
            )

        # `_device` is read by the latency metric (utils/retarget_utils.py:184)
        # to label per-episode timings as cpu/cuda in the dashboard.
        self._device = _detect_compute_device()

    @staticmethod
    def _resolve_checkpoint_dir(root: Path, tag: str) -> Path:
        if not root.is_dir():
            raise FileNotFoundError(
                f"Checkpoint root {root} does not exist. Expected GeoRT checkpoints "
                f"under `<repo_root>/checkpoints/geort/<tag>/`."
            )
        matches = [d for d in root.iterdir() if d.is_dir() and tag in d.name]
        if not matches:
            raise FileNotFoundError(
                f"No checkpoint directory under {root} contains tag {tag!r}."
            )
        if len(matches) > 1:
            names = "\n  ".join(sorted(d.name for d in matches))
            raise ValueError(
                f"Tag {tag!r} matches multiple directories under {root}:\n  {names}\n"
                f"Use a more specific tag."
            )
        return matches[0]

    def retarget(
        self,
        pcloud: np.ndarray,
        wrist_transform: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        keypoints = np.asarray(pcloud, dtype=np.float32)
        if keypoints.ndim == 3 and keypoints.shape[0] == 1:
            keypoints = keypoints[0]

        if self._alignment_landmarks is not None:
            # Per-frame Kabsch-Umeyama: align human landmarks into the robot's
            # current pose (from self._qpos_prev). Same pattern as
            # sampling_based_retargeter.retarget.
            src_landmarks_all = self.from_model.get_landmarks(keypoints)
            tgt_landmarks_all = self.to_model.get_landmarks(
                qpos=np.asarray(self._qpos_prev, dtype=np.float32)
            )
            src_landmarks = np.stack(
                [src_landmarks_all[lm] for lm in self._alignment_landmarks]
            )
            tgt_landmarks = np.stack(
                [tgt_landmarks_all[lm] for lm in self._alignment_landmarks]
            )
            precomputed_scale = None if self._alignment_use_scale else 1.0
            keypoints, _, _ = align_pcloud_kabsch_umeyama(
                points=keypoints,
                source_landmarks=src_landmarks,
                target_landmarks=tgt_landmarks,
                precomputed_scale=precomputed_scale,
            )
            keypoints = np.asarray(keypoints, dtype=np.float32)

        qpos_geort = np.asarray(self._model.forward(keypoints), dtype=np.float32)

        # Scatter the IK output directly into ctrl space. MJCF coupling
        # (tendons, equality constraints) is handled by MuJoCo at sim time.
        out_actuated = np.zeros(self.to_model.num_actuated_dofs, dtype=np.float32)
        out_actuated[self._reorder_idx] = qpos_geort

        for tgt_name, const_val in self.constant_joints.items():
            out_actuated[self._name_to_actuated_idx[tgt_name]] = float(const_val)

        tgt_wrist_transform = None
        if wrist_transform is not None:
            tgt_wrist_transform = self.wrist_retargeter.retarget(wrist_transform)

        return out_actuated[None, :], tgt_wrist_transform
