# Workspace utilization — Monte Carlo sphere-union method

This document explains the implementation of the **Workspace** metric in
[`mimic_retargeter_lab/metrics/workspace.py`](../mimic_retargeter_lab/metrics/workspace.py), what
it measures, and what changed when we replaced the previous voxel-grid
implementation with a gridless Monte Carlo (MC) sphere-union estimator
(GeoRT-style).

## TL;DR

For each fingertip, the metric estimates *what fraction of the robot's
reachable Cartesian workspace was actually exercised by the retargeted
trajectory.*

```
1. Obtain P_robot (N × 3) — a dense point cloud of the fingertip's reachable
   workspace. Loaded from the precomputed cache below if present; otherwise
   sampled live (uniform random joint configs → FK).
2. Build cKDTree on the retargeted trajectory P_traj (T × 3).
3. For each P_robot sample, query the nearest trajectory point.
   A sample is a "hit" iff that distance ≤ radius.
4. utilization = hits / N.
```

`P_robot` is computed once per fingertip and reused across every episode
in the run.

## Precomputing the reachable workspace (recommended)

`P_robot` only depends on the robot's URDF/MJCF and its joint limits, so
sampling it on every metrics run wastes work. Run this once per hand to
produce a cached point cloud:

```bash
python scripts/precompute_workspace.py --num-samples 1000000
# or a subset:
python scripts/precompute_workspace.py --hands mimic_p050_hand shadow_hand
```

The script samples joint configurations uniformly within actuator limits,
runs batched FK via `RobotHandModel.mjx_fk_body_positions`, and writes
`assets/workspace_cache/<hand_name>.npz` keyed by MuJoCo body name.

When `WorkspaceMetric` runs, it resolves each configured fingertip
(`HandLandmark.<TIP>`) to its body name via
`robot_hand_model._landmark_config[lm].name` and pulls the matching
array from the npz. If any landmark isn't covered, only those landmarks
fall back to the live-sampling path; cached ones are still used. The
cache's actual sample count overrides the YAML `num_samples` so the
hits / total ratio reported on the dashboard reflects what was queried.

The previous voxelized approach (precomputed `KP_R` cache + voxel
rasterization + Chamfer distance) is gone. The Chamfer-distance and
voxel-related fields are no longer in the per-episode metric output.

## What changed, mechanically

| | Old (voxel + KP_R) | New (Monte Carlo) |
|---|---|---|
| Reference workspace | Precomputed `assets/workspace_cache/<hand>.npz` (KP_R) | Same precomputed cache when present, live-sampled fallback otherwise |
| Discretization | Fixed voxel grid (`resolution = 2 mm`) | Gridless — exact distance check |
| Coverage test | Voxelize KP_R; mark all voxels within radius of each trajectory point; intersect | `cKDTree(P_traj).query(P_robot, k=1) ≤ radius` |
| Output schema | `{utilization, covered_voxels, total_kp_voxels, resolution, radius}` | `{utilization, hits, num_samples, radius}` |
| Companion metric | `chamfer_distances[fingertip]` — bidirectional NN distance | dropped |
| Error scaling | O(resolution³) bias from voxel quantization | O(1/√N) statistical noise |
| `radius` semantics | Approximated to ceil(radius / resolution) voxel offsets | Exact continuous distance |

## What the metric actually means (subtle but important)

Both methods estimate "fraction of reachable workspace covered." But the
*denominator* — what counts as "reachable workspace" — is constructed
differently between the two approaches, and that changes the
interpretation:

- **Old method:** the reachable workspace was a Cartesian point cloud
  (KP_R), then voxelized. The denominator (`total_kp_voxels`) is
  effectively *uniform in Cartesian space* — each reachable voxel counts
  once regardless of how many joint configurations map there.

- **New method:** we sample joint configurations uniformly within
  actuator limits and push through FK. The Cartesian distribution of
  `P_robot` is **not** uniform in Cartesian space — it's the pushforward
  of uniform-in-joint-space under the FK map. Regions reachable by many
  joint configurations (kinematic redundancy zones, the nominal pose
  neighborhood) get more samples; geometric singularities at the
  envelope get fewer.

So the two metrics answer slightly different questions:

- **Old denominator:** "fraction of Cartesian reachable volume the
  trajectory covers"
- **New denominator:** "fraction of joint-space configurations whose
  fingertip lies within `radius` of the trajectory"

For most retargeting analysis these track each other tightly — *more
joint configs near the trajectory ≈ more Cartesian volume covered near
the trajectory*. But they are not identical. If you ever need a strictly
Cartesian-uniform interpretation, you'd reject-sample `P_robot` against
a uniform spatial grid or reweight by inverse Jacobian determinant. The
GeoRT formulation (and what we use here) accepts the joint-space
pushforward as-is — it's what you actually want for *"is this
retargeter exercising the robot's redundancy-rich operating regions?"*

## Configuration

In each `config/metrics/<hand>/workspace*.yaml`:

```yaml
config:
  display_name: "Workspace"
  num_samples: 1_000_000   # MC sample budget per fingertip
  radius: 0.005            # m — sphere radius around each trajectory point
  seed: 42                 # for reproducibility
  fk_chunk_size: 10_000    # FK batch size to bound memory
  task_space_mapping:
    - landmark: THUMB_TIP
    - landmark: INDEX_TIP
    ...
```

Tuning notes:

- **`num_samples`**: error of the percentage scales as ~1/√N. At
  `N = 1e6` the standard error on a 20% utilization is ~0.04 pp —
  effectively below display rounding. Drop to `N = 1e5` if you want
  faster iteration; expect ~0.13 pp standard error.
- **`radius`**: this is the *meaning* parameter, not just a precision
  parameter. 5 mm answers "did the trajectory pass *near* this point?";
  1 mm answers "did the trajectory actually visit this point?".
  Expect roughly an order of magnitude lower utilization at 1 mm vs
  5 mm — that's correct, not a bug.
- **`fk_chunk_size`**: the FK call is batched and memory scales with
  `chunk × num_landmarks × 16` floats. 10k is fine for most hands.

## Output schema

`metrics_stats["Workspace"][episode_id]`:

```
{
  "workspace_pts": {
    "<landmark>": {"human": (T, 3) array, "robot": (T, 3) array},
    ...
  },
  "utilization": {
    "<landmark>": {
      "utilization": float in [0, 1],
      "hits":         int,
      "num_samples":  int,
      "radius":       float (m),
    },
    ...
  },
}
```

The dashboard renders the per-frame statistics line as:

```
Statistics: utilization=18.14% (9069/50000 samples) | radius=0.005 m
```

## Implementation pointers

- Sphere-union utilization function:
  [`mimic_retargeter_lab/metrics/workspace.py:_compute_sphere_union_utilization`](../mimic_retargeter_lab/metrics/workspace.py)
- Joint-space sampling + shared FK sweep:
  [`mimic_retargeter_lab/metrics/workspace.py:WorkspaceMetric._ensure_p_robot`](../mimic_retargeter_lab/metrics/workspace.py)
- Dashboard rendering:
  [`mimic_retargeter_lab/dashboard/pages/workspace_page.py`](../mimic_retargeter_lab/dashboard/pages/workspace_page.py)
- Reachable-workspace sample is FK'd from
  `RobotHandModel.get_actuated_joint_limits()` and
  `get_landmark_transforms(joint_angles=..., joint_space="ctrl")`.

## What was removed

- Voxel-grid utilization function (was `_compute_workspace_utilization`).
- Per-episode `chamfer_distances` field.
- Dashboard fields: `covered_voxels`, `total_kp_voxels`, `resolution`,
  inline chamfer.

The precompute infrastructure (`scripts/precompute_workspace.py`,
`assets/workspace_cache/`) was already present from the previous voxel
implementation — it's now reused as the source of `P_robot`. The
`chamfer_distance` helper (`mimic_retargeter_lab/objectives/chamfer_distance.py`) is
still present but no longer consumed by the Workspace metric path; safe
to remove in a follow-up.
