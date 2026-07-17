# GeoRT learned retargeter

mimic_retargeter_lab integration of [GeoRT](https://github.com/facebookresearch/GeoRT) — a
learning-based hand retargeter (forward-kinematics MLP + inverse-kinematics
MLP) — adapted to drive MJCF-defined robot hands without any URDF in the
loop.

The upstream package is vendored at
[third_party/geort/](../third_party/geort/) and used unmodified except for a
4-line lazy-import patch on [hand_min.py](../third_party/geort/geort/env/hand_min.py)
so machines without Pinocchio can still `import geort.trainer`. The IK MLP
architecture, FK MLP architecture, and all loss functions are inherited
verbatim from upstream — only the kinematic backend changes.

## Layout

```
third_party/geort/                     # vendored upstream (path-dep in pyproject.toml)
checkpoints/geort/<run-name>/          # trained models — config.json + last.pth (gitignored)
mimic_retargeter_lab/retargeting/online/
    geort_retargeter.py                # inference adapter (BaseOnlineRetargeter)
    geort/
        mjcf_kinematic_model.py        # MjcfHandKinematicModel — MJCF-backed FK
        trainer.py                     # MjcfGeoRTTrainer — subclass of upstream
        train.py                       # CLI entrypoint
config/
    retargeter/geort.yaml
    retargeter_cfg/geort/human_hand_to_<hand>.yaml
    train_geort.yaml
```

## Pipelines

Two pipelines that share the same Kabsch-Umeyama alignment convention as
the rest of mimic_retargeter_lab's retargeters (`sampling_based`, `keyvector`,
`hybrid`): per-frame fit between human landmarks and robot landmarks via
[`align_pcloud_kabsch_umeyama`](../mimic_retargeter_lab/utils/retarget_utils.py#L82).

### Training (one shot per hand+tracker+dataset)

Canonical command (run from repo root):

```bash
JAX_PLATFORMS=cpu python -m mimic_retargeter_lab.retargeting.online.geort.train \
    hand=mimic_p050_hand \
    chirality=right \
    tracker=manus \
    human_data=./dataset/manus \
    exp_id=001
```

Common Hydra overrides:

| Override | Purpose |
|---|---|
| `hand=` | Robot hand type (selects MJCF + per-hand yaml). E.g. `mimic_p050_hand`, `shadow_hand`. |
| `chirality=` | `right` or `left`. Picks `assets/mjcf/<hand>/<chirality>_hand.xml`. |
| `tracker=` | Selects the `training.trackers.<tracker>.human_hand_ids` block in the per-hand yaml. Currently only `manus` is wired. |
| `human_data=` | Path to `.npz` (single episode) or a directory of `.npz` files (multi-episode — concatenated). |
| `exp_id=` | Run identifier. Auto-published as `checkpoints/geort/<hand>_<chirality>_<tracker>_mjcf_run-<exp_id>/`. |
| `training_kwargs.epoch=` | Override total IK MLP epoch count (default 200). |
| `training_kwargs.w_chamfer=`, `.w_curvature=`, `.w_pinch=`, `.w_collision=` | Per-loss weight overrides. |
| `auto_publish=false` | Skip the copy step into `checkpoints/geort/`. Use when you want to inspect raw output before promoting. |

What happens, in order:

1. **Hydra composition** — `train_geort.yaml` pulls in the per-hand
   `retargeter_cfg/geort/human_hand_to_<hand>.yaml`, which carries
   `wrist_mapping`, `checkpoint_tag` (for inference), `alignment_landmarks`,
   per-finger `joint` chains, and per-tracker `human_hand_ids`.

2. **Build robot from MJCF** — `create_robot_hand(...)` produces a
   `RobotHandModel` with MJX FK ready, joint limits resolved from the MJCF.

3. **Load human mocap** — `ManusNpzReader(human_data_path)` yields a
   `(T, 25, 3)` keypoint stream and a matching `from_model = ManusHandModel()`.

4. **Per-frame Kabsch-Umeyama alignment** —
   - **Source**: `from_model.get_landmark_transforms(joints_3d=...)` for the
     configured `alignment_landmarks` (per frame).
   - **Target**: `to_model.get_landmarks(qpos=neutral)` for the same
     landmarks. Fixed across every training frame.
   - Apply `align_pcloud_kabsch_umeyama` per frame to all 25 keypoints.
     Output: `(T, 25, 3)` aligned. The IK MLP sees humans expressed in the
     robot's coordinate frame and at the robot's scale.

5. **Synthesize the GeoRT-shaped config dict in memory** — `joint_order`
   (16 unprefixed qpos names), `fingertip_link` (5 fingers with `link`,
   `joint`, `human_hand_id`, `center_offset`), an `alignment` block recording
   the landmarks/use_scale/target choice for run reproducibility. *No
   `urdf_path`*. Cache name suffixed `_mjcf` so we don't collide with
   URDF-trained FK MLPs.

6. **`os.chdir(third_party/geort)`** — upstream writes `data/<name>.npz`
   and `checkpoint/<name>_<datetime>_<tag>/` cwd-relative.

7. **`MjcfGeoRTTrainer(config, to_model)`** — skips `super().__init__()`
   (which would try to load a URDF) and replaces it with
   `MjcfHandKinematicModel.build_from_config(config, to_model=to_model)`.
   Everything else inherits from the upstream `GeoRTTrainer`.

8. **`trainer.train(aligned, ...)`** — three sub-stages, all upstream code:

   - **Workspace generation** (~100k samples): random qpos within MJCF
     joint limits → `MjcfHandKinematicModel.keypoint_from_qpos` →
     `mj_forward` resolves the 16-dim ctrl through `joint_map.T` to the
     full 20-dim qpos (DIP coupling honored by the MJCF) → fingertip
     positions read from `data.xpos`/`data.site_xpos`. Saved to
     `data/<name>.npz`.
   - **FK MLP training** (200 epochs): per-finger MLP, qpos → fingertip
     position. Cached to `checkpoint/fk_model_<name>.pth`. Provides a
     differentiable forward model for the IK loss. Validate with
     [`scripts/compare_fk_models.py`](../scripts/compare_fk_models.py).
   - **IK MLP training** (200 epochs): the actual retargeter. Input:
     batches of fingertip targets from the *aligned* human data. Output:
     16-dim joint angles. Loss = `chamfer * 80 + curvature * 0.1 +
     pinch * 1.0`.

9. **Save checkpoint** — `save_json(self.config, "config.json")`. The
   `alignment` block we put in goes into the saved JSON for traceability.
   `last.pth` carries the IKModel state dict.

10. **Auto-publish** — copy `config.json + last.pth` from upstream's verbose
    `third_party/geort/checkpoint/<cache_name>_<datetime>_<auto-tag>/` into
    a clean mimic_retargeter_lab-owned `checkpoints/geort/<cache_name>_run-<exp_id>/`.
    The verbose name stays in `third_party/` for traceability; the clean
    `run-<exp_id>` form is what `GeortRetargeter`'s `checkpoint_tag`
    substring-matches against. Re-running the same `exp_id` overwrites the
    published copy. Disable the publish entirely with `auto_publish=false`.

### Inference (per-frame at runtime)

Entry: `python scripts/run_offline_retargeting.py retargeter=geort hand=<hand> offline_source=manus`

1. **Scene construction** — `KinematicRetargetingScene` calls
   `create_retargeter(GEORT, ...)` with the `config:` block of the yaml
   as kwargs (so `alignment_landmarks` flows in directly).

2. **`GeortRetargeter.__init__`** — resolve
   `checkpoints/geort/<checkpoint_tag*>/` (substring match), load
   `GeoRTRetargetingModel(model_path, config_path)` onto CUDA, read
   `joint_order` from `config.json`. Build a name → actuated-index map
   accepting both the `A_`-prefixed actuator name and its unprefixed qpos
   equivalent (mirroring `MjcfHandKinematicModel`'s convention). Cast
   `alignment_landmarks` strings to `HandLandmark` enums. Set
   `self._device` so the latency metric can label timings (cuda/cpu — see
   *Device reporting* below).

3. **Per-frame `retarget(pcloud, wrist_transform)`**:

   ```text
   (1, 25, 3)         from ManusNpzReader.get_iter()
      ↓ squeeze leading batch dim
   (25, 3)            raw manus keypoints
      ↓ Kabsch-Umeyama align:
      ↓   src = from_model.get_landmarks(keypoints)
      ↓   tgt = to_model.get_landmarks(qpos=self._qpos_prev)
      ↓   (same convention as sampling_based_retargeter)
   (25, 3)            aligned keypoints — in robot frame at robot scale
      ↓ GeoRTRetargetingModel.forward()
   (len(joint_order),)  joint angles, semantically ctrl, in joint_order
      ↓ scatter via reorder_idx into ctrl space
   (num_actuated,)    actuated ctrl
      ↓ apply constant_joints overrides
      ↓ add batch dim
   (1, num_actuated)  ctrl — fed to mj_data.ctrl by MujocoHandInterface.
                      MJCF tendons / equality constraints handle DIP coupling
                      at sim time; the retargeter does NOT pre-expand or
                      pre-invert the joint_map.
   ```

   Wrist transform (if provided) goes through the standard
   `WristRetargeter` and is returned alongside the ctrl.

## Train/inference target drift

Both phases use the same alignment primitive and source/target convention,
but the qpos used to compute the target landmarks differs:

| | `qpos` at target | Notes |
|---|---|---|
| Training | `to_model.get_neutral_qpos_pose()` (fixed) | Deterministic, same for every training frame. |
| Inference | `self._qpos_prev` (evolves) | Starts at neutral (= training match), drifts as the model produces outputs. |

Frame 1 at inference is in-distribution by construction. Subsequent frames
see a target that increasingly differs from training — the IK MLP relies on
local invariance to small target shifts, the same property `sampling_based`
exploits when warmstarting from neutral.

For older (URDF-trained, pre-MJCF) checkpoints with no compatible
alignment block, set `alignment_landmarks: null` (or omit it) in the yaml
and inference will skip the Kabsch step entirely. Quality will drop but
the model will still load and run.

## What touches MJCF vs URDF

- **MJCF (mimic_retargeter_lab-owned)**: workspace sampling FK, joint limits,
  fingertip positions, the `joint_map` for coupled-DIP handling. Drives
  both training and inference.
- **URDF**: nothing. The lazy-import patch in
  `third_party/geort/geort/env/hand_min.py` keeps Pinocchio entirely off
  the import path for the MJCF flow.

## Configuration knobs

`config/retargeter_cfg/geort/human_hand_to_<hand>.yaml`:

| Field | Read by | Purpose |
|---|---|---|
| `config.checkpoint_tag` | inference | Substring matched against `checkpoints/geort/` directory names |
| `config.wrist_mapping` | inference | Forwarded to `WristRetargeter` |
| `config.constant_joints` | inference | Override specific joints with constants |
| `config.alignment_landmarks` | training **and** inference | Landmarks for the per-frame Kabsch fit (same convention as `sampling_based`) |
| `config.alignment_use_scale` | training **and** inference | Whether to estimate the Umeyama scale factor |
| `training.fingertips` | training | Per-finger `joint` chain + landmark name (robot side) |
| `training.trackers.<tracker>.human_hand_ids` | training | Per-finger keypoint index (human side) |

`config/train_geort.yaml`:

| Field | Purpose |
|---|---|
| `hand` | Robot hand type (selects the MJCF) |
| `chirality` | `left` or `right` |
| `tracker` | Picks the `human_hand_ids` block |
| `human_data` | Path to `.npz` file (single episode) or directory (multi-episode) |
| `exp_id` | Tagged into the checkpoint directory name |
| `training_kwargs.{w_chamfer, w_curvature, w_collision, w_pinch, epoch}` | Forwarded to `GeoRTTrainer.train` |
| `auto_publish` | Copy `last.pth + config.json` into `checkpoints/geort/` |

## Device reporting

The latency metric reads `retargeter._device.platform`
([`utils/retarget_utils.py:184-185`](../mimic_retargeter_lab/utils/retarget_utils.py#L184-L185))
to label dashboard timings. JAX-based retargeters set this via
`resolve_jax_device(...)`. GeoRT's IK MLP runs in **PyTorch on CUDA** (the
upstream `.cuda()` is hardcoded in
[`export.py:26-27`](../third_party/geort/geort/export.py#L26-L27)), so
`GeortRetargeter.__init__` builds a stub
`types.SimpleNamespace(platform="cuda")` (or `"cpu"` if torch CUDA isn't
detected) — duck-typed against the `jax.Device.platform` attr the latency
metric expects. The dashboard's "Device:" badge shows `cuda` for a normal
training/inference setup.

The latency timer's `jax.block_until_ready(...)` sync is a no-op on the
numpy outputs we produce, but timing is still accurate because
`GeoRTRetargetingModel.forward` does
`joint_normalized.detach().cpu().numpy()` — the `.cpu()` call implicitly
synchronizes the CUDA stream.

## Validation

- [`scripts/compare_fk_models.py`](../scripts/compare_fk_models.py) —
  sanity-checks the trained FK MLP against MJX FK ground truth for any
  hand. Sub-mm RMSE = good fit; mm-scale = undertrained or wired wrong.
- `scripts/compute_hand_retargeter_pair_metrics.py retargeter=geort dataset=manus` —
  runs the full metrics suite (motion preservation, keyvector matching,
  flatness, workspace, collision, latency) against the GeoRT retargeter.

## CUDA caveats

- GeoRT's IKModel hardcodes `.cuda()` in
  [`export.py:26-27`](../third_party/geort/geort/export.py#L26-L27);
  inference will fail on CPU-only hosts.
- mimic_retargeter_lab pins `JAX_PLATFORMS` in `mimic_retargeter_lab/__init__.py`, appending `cuda`
  only when `jax[cuda12]`'s plugin is installed; CPU-only machines get
  `JAX_PLATFORMS=cpu` and need no manual override.

## See also

- [workspace_metric.md](workspace_metric.md) — for the workspace coverage metric used during evaluation.
- [Upstream GeoRT README](../third_party/geort/README.md) — the original package's docs.
