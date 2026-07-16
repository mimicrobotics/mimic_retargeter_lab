from __future__ import annotations

import os
from abc import abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from dexworld.types import Chirality, HandLandmark, MujocoLandmark
from dexworld.utils.mj_utils import get_mj_context

from .base_hand import BaseHandModel


class _NullContext:
    """Trivial no-op context manager (for the ``device=None`` path)."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc) -> None:
        return None


class RobotHandModel(BaseHandModel):
    """Base class for MJCF-based robot hand models.

    Consolidates the shared plumbing that all robot hands need:
    MJCF chain building, ctrl/qpos joint-angle management, FK-based
    fingertip and keyvector computation, joint limits, and kinematic-tree
    traversal.

    Subclasses must provide hand-specific configuration by implementing
    the abstract methods (joint names, couplings, wrist/tip frame names).
    """

    # Subclasses override to True when the MJCF is mirror-symmetric and a
    # single file (named by :attr:`_symmetric_mjcf_name`) is used for both
    # chiralities. ``chirality`` remains meaningful: it tells downstream code
    # which hand this *instance* represents (e.g. in bimanual scenes), even
    # when the underlying MJCF is the same for left and right.
    _mjcf_is_symmetric: bool = False
    _symmetric_mjcf_name: str = "hand.xml"

    def __init__(self, robot_hand_base_path: Path, chirality: Chirality):
        super().__init__(chirality)
        self.robot_hand_base_path = robot_hand_base_path
        if self._mjcf_is_symmetric:
            self.hand_model_path = robot_hand_base_path / self._symmetric_mjcf_name
        else:
            self.hand_model_path = robot_hand_base_path / f"{chirality.value}_hand.xml"

        self.num_fingertips: int
        self.num_actuated_dofs: int
        self.num_qpos_dofs: int

        self._landmark_config: dict[HandLandmark, MujocoLandmark]
        self.joint_map: np.ndarray

        # Populated by create_mjx_kinematic_model() (optional JAX / MJX path).
        self._mj_model: Any = None
        self._mjx_model: Any = None
        self._jax_fk_batch_qpos: Any = None
        self._jax_fk_batch_ctrl: Any = None
        self._landmark_pose_id_cache: dict[tuple[str, str], int] = {}

    # ------------------------------------------------------------------
    # XML / path helpers
    # ------------------------------------------------------------------

    def get_xml_string(self) -> str:
        return self.hand_model_path.read_text()

    def get_model_path(self) -> str:
        return str(self.hand_model_path)

    # ------------------------------------------------------------------
    # Joint-name and coupling abstracts (subclass-specific)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_qpos_joint_names(self) -> list[str]:
        """Return the names of the qpos joints in the same order as
        get_qpos_joint_names().

        IMPORTANT: ordering is a hard contract. Column i of any qpos/control
        vector corresponds to name[i] (after applying joint_map when needed).
        Reordering here changes which physical joint each command drives.
        """
        pass

    @abstractmethod
    def get_actuated_joint_names(self) -> list[str]:
        """Return the names of the actuated joints in the same order as
        get_actuated_joint_names().

        IMPORTANT: keep ordering consistent with the command vector and
        with compute_joint_map() columns.
        """
        pass

    @abstractmethod
    def compute_joint_map(self) -> np.ndarray:
        """Compute the joint map from qpos to actuated DOFs.

        joint_map shape is (num_qpos_dofs, num_actuated_dofs) and maps:
        qpos = ctrl @ joint_map.T. Row/column ordering must match the
        two name helpers above exactly.
        """
        pass

    def _joint_name_from_actuated_name(self, actuated_joint_name: str) -> str:
        """Map an actuated name to a directly-driven qpos joint name."""
        return actuated_joint_name

    def _joint_couplings(self) -> list[dict[str, Any]]:
        """Return coupling config consumed by _build_joint_map_from_couplings."""
        return []

    def _build_joint_map_from_couplings(
        self,
        joint_couplings: list[dict[str, Any]],
    ) -> np.ndarray:
        """Generic qpos<-ctrl map builder with direct and coupling fallback.

        joint_map shape is (num_qpos_dofs, num_actuated_dofs) and maps:
        qpos = ctrl @ joint_map.T. Row/column ordering must match the
        two name helpers above exactly.
        """
        joint_map = np.zeros(
            (self.num_qpos_dofs, self.num_actuated_dofs), dtype=np.float32
        )

        actuated_names = self.get_actuated_joint_names()
        qpos_names = self.get_qpos_joint_names()
        actuated_idx = {name: i for i, name in enumerate(actuated_names)}
        qpos_idx = {name: i for i, name in enumerate(qpos_names)}

        direct_map = {
            self._joint_name_from_actuated_name(a_name): a_name
            for a_name in actuated_names
        }

        for q_name in qpos_names:
            if q_name in direct_map:
                joint_map[qpos_idx[q_name], actuated_idx[direct_map[q_name]]] = 1.0
                continue

            filled = False
            for coupling in joint_couplings:
                if q_name in coupling["children"]:
                    a_name = coupling["parent"]
                    joint_map[qpos_idx[q_name], actuated_idx[a_name]] = float(
                        coupling["children"][q_name]["mult"]
                    )
                    filled = True
                    break
            if not filled:
                raise ValueError(f"Joint {q_name} not found in joint map")

        return joint_map

    def compute_fingertip_jacobians(
        self, joint_angles: Any
    ) -> dict[HandLandmark, jnp.ndarray]:
        """Jacobian of fingertip positions wrt actuated controls using JAX."""
        ctrl = jnp.asarray(joint_angles, dtype=jnp.float32)
        if ctrl.ndim == 2 and ctrl.shape[0] == 1:
            ctrl = ctrl.squeeze(0)

        def _get_tip_positions(c: jnp.ndarray) -> dict[str, jnp.ndarray]:
            # FK returns dict[str, Array(B, 3)]. We slice [0] to remove the batch dim for jacfwd.
            pos = self.mjx_fk_body_positions(c[None, :], joint_space="ctrl")
            return {k: v[0] for k, v in pos.items()}

        # JAX magically returns a dictionary of Jacobians keyed by the exact same strings!
        jacobian_fn = jax.jit(jax.jacfwd(_get_tip_positions))
        full_jac = jacobian_fn(ctrl)

        tip_landmarks = self.get_fingertip_landmarks()

        jac_dict = {}
        for lm in tip_landmarks:
            # Look up the string name, then grab the Jacobian directly from the dict
            link_name = self._landmark_config[lm][0]
            jac_dict[lm] = full_jac[link_name]

        return jac_dict

    def get_landmarks(
        self, qpos: np.ndarray | jnp.ndarray
    ) -> dict[HandLandmark, np.ndarray]:
        q = np.asarray(qpos, dtype=np.float32)
        if q.ndim == 1:
            q = np.expand_dims(q, axis=0)
        if q.ndim != 2:
            raise ValueError(
                f"qpos must have shape (D,) or (B, D), got {tuple(q.shape)}"
            )
        if q.shape[1] == self.num_actuated_dofs:
            q = q @ self.joint_map.T
        elif q.shape[1] != self.num_qpos_dofs:
            raise ValueError(
                f"Expected {self.num_actuated_dofs} (ctrl) or {self.num_qpos_dofs} (qpos) "
                f"dims, got {q.shape[1]}"
            )

        q_np = q
        if q_np.ndim == 2 and q_np.shape[0] == 1:
            q_np = q_np[0]

        model_path = self.get_model_path()
        model, data = get_mj_context(model_path)

        # Safely map the hand's qpos into the global MuJoCo data.qpos buffer
        _, qpos_adrs = self._mj_qpos_address_per_joint(
            model, self.get_qpos_joint_names()
        )
        if q_np.shape[0] != len(qpos_adrs):
            raise ValueError(
                f"Expected {len(qpos_adrs)} qpos values, got {q_np.shape[0]}"
            )

        # Scatter the values
        data.qpos[qpos_adrs] = q_np
        mujoco.mj_forward(model, data)

        landmarks: dict[HandLandmark, np.ndarray] = {}
        for landmark, mj_landmark in self._landmark_config.items():
            cache_key = (mj_landmark.object_type, mj_landmark.name)

            # Look up or cache the integer ID
            if cache_key not in self._landmark_pose_id_cache:
                # Translate our Literal string into MuJoCo's C-enum integer
                mj_obj_int = {
                    "joint": mujoco.mjtObj.mjOBJ_JOINT,
                    "body": mujoco.mjtObj.mjOBJ_BODY,
                    "site": mujoco.mjtObj.mjOBJ_SITE,
                }.get(mj_landmark.object_type)

                if mj_obj_int is None:
                    raise ValueError(
                        f"Unsupported object_type {mj_landmark.object_type!r}"
                    )

                obj_id = mujoco.mj_name2id(model, mj_obj_int, mj_landmark.name)
                if obj_id < 0:
                    raise ValueError(
                        f"{mj_landmark.object_type} {mj_landmark.name!r} not found in model {model_path}"
                    )

                self._landmark_pose_id_cache[cache_key] = int(obj_id)

            # Fetch positions using the cached ID
            obj_id = self._landmark_pose_id_cache[cache_key]

            if mj_landmark.object_type == "joint":
                landmarks[landmark] = data.xanchor[obj_id].copy()
            elif mj_landmark.object_type == "body":
                landmarks[landmark] = data.xpos[obj_id].copy()
            elif mj_landmark.object_type == "site":
                landmarks[landmark] = data.site_xpos[obj_id].copy()
            else:
                error_msg = f"Unsupported object_type {mj_landmark.object_type!r}"
                raise ValueError(error_msg)

        return landmarks

    def get_landmark_transforms(
        self,
        joint_angles: np.ndarray,
        joint_space: Literal["ctrl", "qpos"] = "ctrl",
    ) -> dict[HandLandmark, np.ndarray]:
        """Return 4x4 homogeneous transforms for all configured landmarks.

        Unlike ``to_kinematic_tree``, this method is not affected by
        ``_kinematic_tree_exclude_frames`` and queries MuJoCo directly for
        every entry in ``_landmark_config``.

        Parameters
        ----------
        joint_angles : (D,) or (N, D) array
        joint_space : "ctrl" or "qpos"

        Returns
        -------
        dict mapping HandLandmark → ndarray of shape (4, 4) or (N, 4, 4).
        """
        q_batch = np.asarray(joint_angles, dtype=np.float32)
        is_single = q_batch.ndim == 1
        if is_single:
            q_batch = q_batch[None, :]

        model, data = get_mj_context(self.get_model_path())

        # Pre-resolve landmark IDs (reuses the same cache as get_landmarks)
        resolved: list[tuple[HandLandmark, str, int]] = []
        for landmark, mj_landmark in self._landmark_config.items():
            cache_key = (mj_landmark.object_type, mj_landmark.name)
            if cache_key not in self._landmark_pose_id_cache:
                mj_obj_int = {
                    "joint": mujoco.mjtObj.mjOBJ_JOINT,
                    "body": mujoco.mjtObj.mjOBJ_BODY,
                    "site": mujoco.mjtObj.mjOBJ_SITE,
                }[mj_landmark.object_type]
                obj_id = mujoco.mj_name2id(model, mj_obj_int, mj_landmark.name)
                if obj_id < 0:
                    raise ValueError(
                        f"{mj_landmark.object_type} {mj_landmark.name!r} not found"
                    )
                self._landmark_pose_id_cache[cache_key] = int(obj_id)
            resolved.append(
                (
                    landmark,
                    mj_landmark.object_type,
                    self._landmark_pose_id_cache[cache_key],
                )
            )

        all_frames: list[dict[HandLandmark, np.ndarray]] = []
        for q in q_batch:
            if joint_space == "ctrl":
                qpos = q @ self.joint_map.T
            else:
                qpos = q
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)

            frame: dict[HandLandmark, np.ndarray] = {}
            for landmark, obj_type, obj_id in resolved:
                T = np.eye(4, dtype=np.float32)
                if obj_type == "body":
                    T[:3, :3] = data.xmat[obj_id].reshape(3, 3)
                    T[:3, 3] = data.xpos[obj_id]
                elif obj_type == "site":
                    T[:3, :3] = data.site_xmat[obj_id].reshape(3, 3)
                    T[:3, 3] = data.site_xpos[obj_id]
                elif obj_type == "joint":
                    T[:3, :3] = data.xmat[model.jnt_bodyid[obj_id]].reshape(3, 3)
                    T[:3, 3] = data.xanchor[obj_id]
                frame[landmark] = T
            all_frames.append(frame)

        if is_single:
            return all_frames[0]

        return {lm: np.stack([f[lm] for f in all_frames]) for lm in all_frames[0]}

    @staticmethod
    def _mj_qpos_address_per_joint(
        mj_model: Any, joint_names: list[str]
    ) -> tuple[int, np.ndarray]:
        """Return (nq, int32 array of qpos indices) for one scalar per named joint."""
        nq = int(mj_model.nq)
        adrs: list[int] = []
        for name in joint_names:
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"MuJoCo joint {name!r} not found in MJCF model.")
            adr = int(mj_model.jnt_qposadr[jid])
            jt = int(mj_model.jnt_type[jid])
            if jt in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                pass
            else:
                raise NotImplementedError(
                    f"Joint {name!r} has type {jt}; only hinge/slide (1 qpos) is supported."
                )
            adrs.append(adr)
        adr_arr = np.asarray(adrs, dtype=np.int32)
        if adr_arr.size != len(joint_names):
            raise RuntimeError("internal: qpos address list length mismatch")
        if len(np.unique(adr_arr)) != len(adr_arr):
            raise ValueError("Duplicate qpos addresses for hand joints; check MJCF.")
        return nq, adr_arr

    def create_mjx_kinematic_model(self, device: Any | None = None) -> None:
        """Load this hand's MJCF into MuJoCo/MJX and build batched FK callables.

        All MJX/JAX arrays placed during this call (the ``mjx_model``, body/joint/
        site index arrays, joint-map, and the JIT'd FK functions) are pinned to
        ``device`` if provided — otherwise they land on JAX's process-default
        device. Pass a specific ``jax.Device`` (e.g. ``jax.devices('cuda')[0]``)
        when a downstream consumer (e.g. ``SamplingBasedRetargeter`` configured
        for GPU) needs the FK kernel resident on that device. Calling this
        method again on the same model is supported — it discards previous
        builds and re-traces under the new ``device``.
        """

        # 1. Parse the unified config into separate categories
        body_names, joint_names, site_names = [], [], []
        for name, kind in self._landmark_config.values():
            if kind == "body" and name not in body_names:
                body_names.append(name)
            elif kind == "joint" and name not in joint_names:
                joint_names.append(name)
            elif kind == "site" and name not in site_names:
                site_names.append(name)

        curr_dir = os.getcwd()
        os.chdir(self.robot_hand_base_path)
        try:
            mj_model = mujoco.MjModel.from_xml_path(str(self.hand_model_path))
        finally:
            os.chdir(curr_dir)

        mj_nq, qpos_adrs_np = self._mj_qpos_address_per_joint(
            mj_model, self.get_qpos_joint_names()
        )
        if int(qpos_adrs_np.shape[0]) != self.num_qpos_dofs:
            raise ValueError(
                f"MJCF qpos layout: expected {self.num_qpos_dofs} scalar joints, "
                f"got {qpos_adrs_np.shape[0]} from get_qpos_joint_names()."
            )

        # 2. Get IDs for all tracked objects (host-side; no device dependency)
        body_ids = [
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, n) for n in body_names
        ]
        joint_ids = [
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in joint_names
        ]
        site_ids = [
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, n) for n in site_names
        ]

        # Check for missing IDs (-1)
        if -1 in body_ids:
            raise ValueError(f"Missing body in MJCF: {body_names[body_ids.index(-1)]}")
        if -1 in joint_ids:
            raise ValueError(
                f"Missing joint in MJCF: {joint_names[joint_ids.index(-1)]}"
            )
        if -1 in site_ids:
            raise ValueError(f"Missing site in MJCF: {site_names[site_ids.index(-1)]}")

        # 3. Place all MJX/JAX state on the requested device. The closure below
        # captures these arrays, and the resulting JIT'd function specializes
        # to the captured devices — so this context manager is what actually
        # pins FK to ``device``. If ``device`` is None, JAX uses its current
        # default (typically CPU: dexworld pins ``JAX_PLATFORMS`` CPU-first).
        device_ctx = (
            jax.default_device(device) if device is not None else _NullContext()
        )
        with device_ctx:
            # ``mjx.put_model`` ignores ``jax.default_device(...)`` — it resolves
            # the placement device through its own ``_resolve_impl_and_device``
            # path, which checks the explicit ``device`` kwarg first.
            mjx_model = mjx.put_model(mj_model, device=device)
            qpos_adrs = jnp.asarray(qpos_adrs_np, dtype=jnp.int32)
            body_ids_jnp = jnp.asarray(body_ids, dtype=jnp.int32)
            joint_ids_jnp = jnp.asarray(joint_ids, dtype=jnp.int32)
            site_ids_jnp = jnp.asarray(site_ids, dtype=jnp.int32)
            mjx_joint_map = jnp.asarray(self.joint_map)

            def fk_qpos_row(q_hand: jnp.ndarray) -> dict[str, jnp.ndarray]:
                """Forward kinematics for a single qpos vector."""
                qpos = jnp.zeros(mj_nq, dtype=jnp.float32)
                qpos = qpos.at[qpos_adrs].set(q_hand.astype(jnp.float32))
                # ``device=device`` keeps ``Data`` placement consistent with the
                # ``mjx_model`` above; otherwise MJX falls into its default
                # branch (``jax.devices()[0]``, which is CPU under
                # ``JAX_PLATFORMS`` CPU-first) and logs/places on CPU regardless
                # of where the model lives.
                d = mjx.make_data(mjx_model, device=device)
                d = d.replace(qpos=qpos)
                d = mjx.kinematics(mjx_model, d)

                # 4. Pull from xpos, xanchor, and site_xpos dynamically
                out = {}
                if len(body_names) > 0:
                    out.update(
                        {
                            bname: d.xpos[bid]
                            for bname, bid in zip(body_names, body_ids_jnp)
                        }
                    )
                if len(joint_names) > 0:
                    out.update(
                        {
                            jname: d.xanchor[jid]
                            for jname, jid in zip(joint_names, joint_ids_jnp)
                        }
                    )
                if len(site_names) > 0:
                    out.update(
                        {
                            sname: d.site_xpos[sid]
                            for sname, sid in zip(site_names, site_ids_jnp)
                        }
                    )
                return out

            def fk_ctrl_row(ctrl: jnp.ndarray) -> dict[str, jnp.ndarray]:
                """Forward kinematics for a single ctrl vector."""
                q_hand = ctrl.astype(jnp.float32) @ mjx_joint_map.T
                return fk_qpos_row(q_hand)

            self._mj_model = mj_model
            self._mjx_model = mjx_model
            self._jax_fk_batch_qpos = jax.jit(jax.vmap(fk_qpos_row))
            self._jax_fk_batch_ctrl = jax.jit(jax.vmap(fk_ctrl_row))
            self._mjx_fk_device = device

    def mjx_fk_body_positions(
        self,
        joint_angles: Any,
        joint_space: Literal["ctrl", "qpos"] = "qpos",
    ) -> dict[str, Any]:
        """Batched MJX forward kinematics: world positions of :meth:`get_mjx_fk_body_names`.

        Args:
            joint_angles: Array-like or JAX array of shape (B, D) where D is
                ``num_qpos_dofs`` or ``num_actuated_dofs`` depending on ``joint_space``.
            joint_space: ``"qpos"`` matches :meth:`get_qpos_joint_names` order;
                ``"ctrl"`` matches :meth:`get_actuated_joint_names` and applies
                ``joint_map`` the same way as :meth:`compute_keyvectors`.

        Returns:
            JAX array (B, K, 3) with body positions in MuJoCo world frame.
        """
        if self._jax_fk_batch_qpos is None:
            self.create_mjx_kinematic_model()

        q = jnp.asarray(joint_angles, dtype=jnp.float32)
        if q.ndim != 2:
            raise ValueError(
                f"mjx_fk_body_positions expects shape (B, D), got {tuple(q.shape)}"
            )

        if joint_space == "qpos":
            if q.shape[1] != self.num_qpos_dofs:
                raise ValueError(
                    f"qpos joint angles must have {self.num_qpos_dofs} dims, "
                    f"got {q.shape[1]}"
                )
            return self._jax_fk_batch_qpos(q)
        if joint_space == "ctrl":
            if q.shape[1] != self.num_actuated_dofs:
                raise ValueError(
                    f"ctrl joint angles must have {self.num_actuated_dofs} dims, "
                    f"got {q.shape[1]}"
                )
            return self._jax_fk_batch_ctrl(q)

        raise ValueError(f"joint_space must be 'ctrl' or 'qpos', got {joint_space!r}")

    def _keyvectors_from_body_positions_jax(
        self, pos: dict[str, Any]
    ) -> dict[str, Any]:
        """Pairwise keyvectors from dictionary of body positions, keyed by Enum names.

        Mirrors :meth:`BaseHandModel.compute_keyvectors`: emits all pairs among
        ``{WRIST, *_TIP}`` plus per-finger ``<FINGER>_DP_to_<FINGER>_TIP`` for
        any finger where both the DP and TIP landmarks are registered.
        """

        # 1. Gather all the landmarks we care about (Wrist + Fingertips)
        landmarks = []
        if HandLandmark.WRIST in self._landmark_config:
            landmarks.append(HandLandmark.WRIST)
        landmarks.extend(self.get_fingertip_landmarks())

        # 2. Verify all required frames actually exist in the JAX FK output
        for lm in landmarks:
            mj_name = self._landmark_config[lm][0]
            if mj_name not in pos:
                raise KeyError(
                    f"Required keyvector frame '{mj_name}' (for {lm.name}) "
                    "missing from MJX FK output."
                )

        # 3. Calculate pairwise vectors and assign to Universal string keys
        num = len(landmarks)
        out: dict[str, Any] = {}
        for i in range(num):
            for j in range(i + 1, num):
                lm_i = landmarks[i]
                lm_j = landmarks[j]

                # Fetch (x,y,z) positions using the raw MuJoCo string names
                name_i = self._landmark_config[lm_i][0]
                name_j = self._landmark_config[lm_j][0]
                vec = pos[name_j] - pos[name_i]

                # Create the Universal Key! (e.g., "WRIST_to_THUMB_TIP")
                universal_key = f"{lm_i.name}_to_{lm_j.name}"
                out[universal_key] = vec

        # 4. Per-finger DP->TIP orientation vectors.
        finger_dp_tip_pairs = [
            (HandLandmark.THUMB_DP, HandLandmark.THUMB_TIP),
            (HandLandmark.INDEX_DP, HandLandmark.INDEX_TIP),
            (HandLandmark.MIDDLE_DP, HandLandmark.MIDDLE_TIP),
            (HandLandmark.RING_DP, HandLandmark.RING_TIP),
            (HandLandmark.PINKY_DP, HandLandmark.PINKY_TIP),
        ]
        for dp, tip in finger_dp_tip_pairs:
            if dp not in self._landmark_config or tip not in self._landmark_config:
                continue
            dp_name = self._landmark_config[dp][0]
            tip_name = self._landmark_config[tip][0]
            if dp_name not in pos or tip_name not in pos:
                continue
            out[f"{dp.name}_to_{tip.name}"] = pos[tip_name] - pos[dp_name]

        return out

    # ------------------------------------------------------------------
    # Joint limits
    # ------------------------------------------------------------------

    def get_joint_limits(self) -> dict[str, tuple[float, float]]:
        """Fetch joint limits directly from the initialized MuJoCo model."""
        if self._mj_model is None:
            self.create_mjx_kinematic_model()

        limits = {}
        for name in self.get_qpos_joint_names():
            jid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                # MuJoCo stores limits in jnt_range as [lower, upper]
                limits[name] = tuple(self._mj_model.jnt_range[jid])
        return limits

    def get_frame_names(self) -> list[str]:
        """Return the names of all bodies in the MuJoCo model.

        Used by consumers (e.g. contact-state queries) to identify which
        MuJoCo bodies belong to this hand.
        """
        if self._mj_model is None:
            self.create_mjx_kinematic_model()

        names = []
        for i in range(self._mj_model.nbody):
            name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name is not None:
                names.append(name)
        return names

    def get_actuated_joint_limits(self) -> dict[str, tuple[float, float]]:
        """Return limits for actuated controls."""
        joint_limits = self.get_joint_limits()
        actuated = self.get_actuated_joint_names()
        qpos = self.get_qpos_joint_names()

        # Cleaned up: No more torch.Tensor checks
        joint_map = np.asarray(self.joint_map, dtype=np.float32)

        out: dict[str, tuple[float, float]] = {}

        for j, aname in enumerate(actuated):
            candidates = [aname]
            if "_A" in aname:
                candidates.append(aname.replace("_A", ""))

            found = None
            for key in candidates:
                if key in joint_limits:
                    found = tuple(map(float, joint_limits[key]))
                    break

            if found is not None:
                out[aname] = found
                continue

            col = joint_map[:, j]
            nz = np.where(np.abs(col) > 1e-8)[0]
            if len(nz) == 0:
                raise KeyError(
                    f"Cannot infer limits for actuated joint '{aname}': no nonzero joint_map entries."
                )

            lo_acc, hi_acc = -np.inf, np.inf

            for qi in nz:
                qname = qpos[qi]
                if qname not in joint_limits:
                    raise KeyError(
                        f"Missing qpos limit for '{qname}' while inferring '{aname}'."
                    )
                qlo, qhi = map(float, joint_limits[qname])
                a = float(col[qi])

                if a > 0:
                    alo, ahi = qlo / a, qhi / a
                else:
                    alo, ahi = qhi / a, qlo / a

                lo_acc = max(lo_acc, min(alo, ahi))
                hi_acc = min(hi_acc, max(alo, ahi))

            out[aname] = (lo_acc, hi_acc)

        return out

    # ------------------------------------------------------------------
    # Pose defaults
    # ------------------------------------------------------------------

    def get_zero_qpos_pose(self) -> np.ndarray:
        return np.zeros(self.num_qpos_dofs, dtype=np.float32)

    def get_zero_ctrl_pose(self) -> np.ndarray:
        return np.zeros(self.num_actuated_dofs, dtype=np.float32)

    def get_neutral_qpos_pose(self) -> np.ndarray:
        return np.zeros(self.num_qpos_dofs, dtype=np.float32)

    def get_neutral_ctrl_pose(self) -> np.ndarray:
        return np.zeros(self.num_actuated_dofs, dtype=np.float32)

    # ------------------------------------------------------------------
    # FK helpers
    # ------------------------------------------------------------------

    def get_num_fingertips(self) -> int:
        return self.num_fingertips

    def get_fingertip_landmarks(self) -> list[HandLandmark]:
        """Return available fingertip landmarks in canonical order.

        Some embodiments expose 4 fingertips (no pinky), others expose 5.
        We derive availability from the model's landmark-frame mapping so
        callers don't need per-hand branching.
        """
        canonical = [
            HandLandmark.THUMB_TIP,
            HandLandmark.INDEX_TIP,
            HandLandmark.MIDDLE_TIP,
            HandLandmark.RING_TIP,
            HandLandmark.PINKY_TIP,
        ]
        return [lm for lm in canonical if lm in self._landmark_config]

    def compute_keyvectors_jax(
        self,
        joint_angles: jnp.ndarray,
        joint_space: Literal["ctrl", "qpos"] = "ctrl",
    ) -> dict[str, jnp.ndarray]:
        """Stateless, pure JAX keyvector computation."""
        pos = self.mjx_fk_body_positions(joint_angles, joint_space=joint_space)
        return self._keyvectors_from_body_positions_jax(pos)

    # ------------------------------------------------------------------
    # Kinematic-tree traversal
    # ------------------------------------------------------------------

    def _kinematic_tree_exclude_frames(self) -> list[str]:
        return ["world"]

    def _kinematic_tree_extra_links(
        self, fk_out: dict[str, np.ndarray]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        return []

    def to_kinematic_tree(
        self,
        joint_angles: Any,
        joint_space: Literal["ctrl", "qpos"] = "ctrl",
        exclude_frames: list[str] | None = None,
        return_frame_dict: bool = False,
    ):
        """Builds a kinematic tree by querying MuJoCo's native parent-child relationships."""
        if exclude_frames is None:
            exclude_frames = self._kinematic_tree_exclude_frames()

        if self._mj_model is None:
            self.create_mjx_kinematic_model()

        # 1. Ensure input is batched
        q_batch = np.asarray(joint_angles, dtype=np.float32)
        is_single = False
        if q_batch.ndim == 1:
            q_batch = q_batch[None, :]
            is_single = True

        model, data = get_mj_context(self.get_model_path())

        all_frames = []
        all_links = []

        # 2. Iterate over the time sequence
        for q in q_batch:
            if joint_space == "ctrl":
                qpos = q @ self.joint_map.T
            else:
                qpos = q

            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)

            frames: dict[str, np.ndarray] = {}
            links: list[tuple[np.ndarray, np.ndarray]] = []

            for i in range(model.nbody):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                if not name or name in exclude_frames:
                    continue

                pos = data.xpos[i]
                mat = data.xmat[i].reshape(3, 3)
                transform = np.eye(4, dtype=np.float32)
                transform[:3, :3] = mat
                transform[:3, 3] = pos
                frames[name] = transform

                parent_id = model.body_parentid[i]
                if parent_id != 0:
                    parent_name = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_BODY, parent_id
                    )
                    if parent_name and parent_name not in exclude_frames:
                        parent_pos = data.xpos[parent_id]
                        links.append((parent_pos.copy(), pos.copy()))

            # Also include sites so metrics can reference site-based landmarks.
            for si in range(model.nsite):
                site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, si)
                if not site_name or site_name in exclude_frames:
                    continue
                site_pos = data.site_xpos[si]
                site_mat = data.site_xmat[si].reshape(3, 3)
                site_transform = np.eye(4, dtype=np.float32)
                site_transform[:3, :3] = site_mat
                site_transform[:3, 3] = site_pos
                frames[site_name] = site_transform

            links.extend(self._kinematic_tree_extra_links(frames))
            all_frames.append(frames)
            all_links.append(links)

        # 3. Collate and return
        if return_frame_dict:
            batched_frames = {
                k: np.stack([f[k] for f in all_frames]) for k in all_frames[0].keys()
            }
            return (batched_frames if not is_single else all_frames[0]), all_links[0]

        frame_tensors = np.stack([np.stack(list(f.values())) for f in all_frames])
        return (frame_tensors if not is_single else frame_tensors[0]), all_links[0]

    def get_mesh_geoms(
        self,
        joint_angles: Any,
        joint_space: Literal["ctrl", "qpos"] = "ctrl",
    ) -> list[dict[str, np.ndarray]]:
        """Return world-frame triangle meshes for this hand's visual geoms.

        Pulls mesh vertices and faces straight from ``model.mesh_vert`` /
        ``model.mesh_face`` (already scaled at MJCF load time), and applies
        the per-geom world pose from ``data.geom_xpos`` / ``data.geom_xmat``
        after ``mj_forward``. No STL files are re-read at runtime.

        Visual geoms are selected via the portable "non-colliding" test
        (``contype == 0 and conaffinity == 0``) — different hands place
        visuals on different ``geom_group`` numbers, but all of our MJCFs
        mark visual-only geoms with zeroed collision masks.

        Args:
            joint_angles: 1D array of shape (D,) — a single pose.
            joint_space: "ctrl" (then mapped via ``joint_map``) or "qpos".

        Returns:
            List of dicts: ``{"vertices": (V, 3), "faces": (F, 3), "rgba": (4,)}``.
        """
        if self._mj_model is None:
            self.create_mjx_kinematic_model()

        q = np.asarray(joint_angles, dtype=np.float32)
        if q.ndim != 1:
            raise ValueError(
                f"get_mesh_geoms expects a single pose (D,), got shape {tuple(q.shape)}"
            )
        qpos = q @ self.joint_map.T if joint_space == "ctrl" else q

        model, data = get_mj_context(self.get_model_path())
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        geoms: list[dict[str, np.ndarray]] = []
        mesh_geom = int(mujoco.mjtGeom.mjGEOM_MESH)

        # First pass: collect visual geoms (contype==0, conaffinity==0)
        # and track which bodies already have a visual geom.
        bodies_with_visual: set[int] = set()
        visual_geom_ids: list[int] = []
        for gi in range(model.ngeom):
            if int(model.geom_type[gi]) != mesh_geom:
                continue
            if (
                int(model.geom_contype[gi]) == 0
                and int(model.geom_conaffinity[gi]) == 0
            ):
                visual_geom_ids.append(gi)
                bodies_with_visual.add(int(model.geom_bodyid[gi]))

        # Second pass: for bodies that have NO visual geom, fall back to
        # their collision mesh so the dashboard can still render them.
        fallback_geom_ids: list[int] = []
        for gi in range(model.ngeom):
            if int(model.geom_type[gi]) != mesh_geom:
                continue
            bid = int(model.geom_bodyid[gi])
            if bid not in bodies_with_visual:
                fallback_geom_ids.append(gi)
                bodies_with_visual.add(bid)  # only take one per body

        for gi in visual_geom_ids + fallback_geom_ids:
            mid = int(model.geom_dataid[gi])
            if mid < 0:
                continue

            v_adr = int(model.mesh_vertadr[mid])
            v_num = int(model.mesh_vertnum[mid])
            f_adr = int(model.mesh_faceadr[mid])
            f_num = int(model.mesh_facenum[mid])
            if v_num <= 0 or f_num <= 0:
                continue

            verts_local = np.asarray(
                model.mesh_vert[v_adr : v_adr + v_num], dtype=np.float32
            )
            faces = np.asarray(model.mesh_face[f_adr : f_adr + f_num], dtype=np.int32)

            pos = np.asarray(data.geom_xpos[gi], dtype=np.float32)
            rot = np.asarray(data.geom_xmat[gi], dtype=np.float32).reshape(3, 3)
            verts_world = verts_local @ rot.T + pos

            rgba = np.asarray(model.geom_rgba[gi], dtype=np.float32).copy()
            # Fallback collision geoms may have invisible material (alpha=0);
            # force them visible with a default grey color.
            if rgba[3] < 0.01:
                rgba = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)

            geoms.append(
                {
                    "vertices": verts_world,
                    "faces": faces,
                    "rgba": rgba,
                }
            )

        return geoms
