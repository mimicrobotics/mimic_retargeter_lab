# Integrating a New Robot Hand

This section walks through every file you need to create or modify. We'll use `leap_hand` as a running example — substitute your own hand name throughout.

### 1. MJCF Model

Place your MuJoCo XML in `assets/mjcf/<hand_name>/right_hand.xml` (and optionally `left_hand.xml`).

**Key requirements:**

- **`arm_attachment` wrapper body.** The first body under `<worldbody>` must be named `arm_attachment`. The retargeting scene resolves it as the hand's wrist body ([`kinematic_retargeting.py`](../mimic_retargeter_lab/scenes/kinematic_retargeting.py)) in order to position the hand in the scene. If you're converting from a URDF that has a `freejoint`, replace it:
  ```xml
  <worldbody>
    <body name="arm_attachment" pos="0 0 0" childclass="robot">
      <body name="base" ...>
        <body name="palm_lower" ...>
          <!-- fingers... -->
        </body>
      </body>
    </body>
  </worldbody>
  ```

- **Position actuators (not motor).** Use `<position>` actuators with `kp`/`kv` gains so the hand holds its pose via PD control. Motor (torque) actuators will cause the hand to flop under gravity.
  ```xml
  <actuator>
    <position name="1_ctrl" joint="1" ctrlrange="-0.314 2.23" kp="3.0" kv="0.01" />
    ...
  </actuator>
  ```

- **Comment out (or remove) the wrist joint and actuator.** This repo focuses on hand retargeting — wrist motion is out of scope, so the wrist must not appear as an actuated DoF. Comment out `<joint>` tags for the wrist and the matching `<position>` actuators; keep the wrist **body** (a jointless body is rigidly welded to its parent in MuJoCo, so it still resolves as the `HandLandmark.WRIST` frame). If your URDF has multiple wrist joints (e.g. roll/pitch), comment out all of them. Do **not** simulate "frozen" by setting `range="0 0"` — that keeps the joint in `qpos` and forces every retargeter to clamp it. See the existing `shadow_hand` and `orca_v2_hand` MJCFs for examples.

- **Collision geoms hidden by default.** Put collision geoms in `group="3"` so they don't clutter the viewer. Visual geoms go in `group="1"`.

- **Fingertip landmarks.** Make sure you have identifiable bodies (or sites) at the actual fingertip contact points. If your URDF has `tip_head` links offset far from the contact surface, use the distal phalanx body instead — the retargeter needs landmarks where contact actually happens.

- **Mesh file mapping.** If converting from URDF, MuJoCo can only read `.obj` and `.stl` (not `.glb`). You may need to remap mesh references to the available `.obj` files. Set `meshdir` in the `<compiler>` tag, and make sure you only have **one** `<compiler>` element.

### 2. Add to `RobotHandType` Enum

**File:** `mimic_retargeter_lab/types/types.py`

```python
class RobotHandType(str, enum.Enum):
    ...
    LEAP_HAND = "leap_hand"  # Add your hand here
```

### 3. Create the Hand Model Class

**File:** `mimic_retargeter_lab/hand_models/<hand_name>.py`

Use an existing 4-finger hand (e.g., `wonik_allegro_hand.py`) or 5-finger hand (e.g., `mimic_p050_hand.py`) as a template. You need to implement:

- **`_landmark_config`** — Maps `HandLandmark` enums to MuJoCo body/site/joint names. This is the single source of truth for all landmark lookups. At minimum, define:
  - `WRIST`, `THUMB_TIP`, `INDEX_TIP`, `MIDDLE_TIP`, `RING_TIP` (and `PINKY_TIP` for 5-finger hands)
  - `THUMB_BASE`, `INDEX_BASE`, `RING_BASE` (used for Kabsch alignment)
  - `THUMB_DP`, `INDEX_DP`, etc. (used by AKO retargeter)

- **`get_qpos_joint_names()`** — Return the MuJoCo **joint** names in qpos order.

- **`get_actuated_joint_names()`** — Return the MuJoCo **actuator** names. These often differ from joint names (e.g., joint `"1"` has actuator `"1_ctrl"`).

- **`_joint_name_from_actuated_name()`** — Override this if actuator names differ from joint names. It maps an actuator name back to a qpos joint name for the joint map builder.
  ```python
  def _joint_name_from_actuated_name(self, actuated_joint_name: str) -> str:
      # Example: "1_ctrl" -> "1"
      if actuated_joint_name.endswith("_ctrl"):
          return actuated_joint_name[:-5]
      return actuated_joint_name
  ```

- **`compute_joint_map()`** — Return an identity matrix for hands with no coupled joints, or use `_build_joint_map_from_couplings()` for hands with tendon couplings (see `mimic_p050_hand.py`).

- **`get_neutral_ctrl_pose()`** — Override if the hand needs a non-zero neutral pose. Hands with large negative joint ranges (allowing hyperextension) should use a slightly flexed neutral pose to avoid local minima during cold-start optimization.

- **`self.create_mjx_kinematic_model()`** — Call this in `__init__` to compile the MJX kinematic model for GPU-accelerated FK.

### 4. Register the Hand Model

**File:** `mimic_retargeter_lab/hand_models/__init__.py`

Add to the registry:
```python
ROBOT_HAND_REGISTRY = {
    ...
    RobotHandType.LEAP_HAND: LeapHandModel,
}
```

### 5. Create Configuration Files

#### Hand config

**File:** `config/hand/<hand_name>.yaml`
```yaml
name: "<hand_name>"
```

#### Retargeter configs (one per retargeter type)

Create one file per retargeter under `config/retargeter_cfg/<retargeter>/human_hand_to_<hand_name>.yaml`:

| Config file | Key fields |
|-------------|------------|
| `keyvector/human_hand_to_<hand_name>.yaml` | `wrist_mapping` (tgt_key, root_key), `alignment_landmarks`, `precomputed_scale`, `keyvectors_cfg` |
| `dexpilot/human_hand_to_<hand_name>.yaml` | Same as keyvector + `retargeter_params` (project_distance, escape_distance, eta) |
| `ako/human_hand_to_<hand_name>.yaml` | Same as keyvector + `retargeter_params` (epsilon, huber_delta) + `regularized_joints` |
| `joint_angle/human_hand_to_<hand_name>.yaml` | `joint_mapping` entries with `tgt_key` matching **qpos joint names** |

**Important notes:**
- `wrist_mapping.tgt_key` and `root_key` must match a body name in your MJCF (e.g., `base` or `palm_lower`).
- `precomputed_scale`: set to `1.0` if the hand is similar in size to a human hand. Use `null` for auto-scale (Kabsch-Umeyama), but beware that auto-scale can over-shrink if the alignment landmarks have very different proportions than the human hand.
- `alignment_landmarks`: for 4-finger hands, use `[thumb_base, index_base, ring_base]` (not `pinky_base`).
- For the `joint_angle` config, `tgt_key` must match the **qpos joint name** (not the actuator name).
- `rot_offset_euler` and `trans_offset` in `wrist_mapping` are auto-computed from the robot's neutral-pose landmarks when omitted. Manual overrides are rarely needed.

#### Metrics configs

Create `config/metrics/<hand_name>/` with one file per metric. Copy from an existing hand (e.g., `wonik_allegro_hand`) and update the `tgt` field in `task_space_mapping` to match your MJCF body names:

```
config/metrics/<hand_name>/
├── flatness.yaml
├── keyvector_matching.yaml
├── motion_preservation.yaml
├── pinch_grasps.yaml
├── response.yaml
└── workspace.yaml
```

### 6. Create Unit Tests

**File:** `tests/hand_models/test_<hand_name>_model.py`

```python
import pytest
from mimic_retargeter_lab.hand_models import LeapHandModel
from mimic_retargeter_lab.types import HandLandmark
from tests.hand_models.test_robot_hand_models_base import BaseHandModelRegressionSuite

@pytest.fixture
def model(leap_hand_path, right):
    return LeapHandModel(leap_hand_path, right)

@pytest.fixture
def golden(load_golden):
    return load_golden("leap_hand_right_golden.npz")

class TestLeapHandModel(BaseHandModelRegressionSuite):
    EXPECTED_QPOS_DOFS = 16
    EXPECTED_ACTUATED_DOFS = 16
    EXPECTED_FINGERTIPS = 4
    EXPECTED_LANDMARK_ORDER = [
        HandLandmark.THUMB_TIP,
        HandLandmark.INDEX_TIP,
        HandLandmark.MIDDLE_TIP,
        HandLandmark.RING_TIP,
    ]
    EXPECTED_JOINT_MAP_SEMANTIC_COUPLINGS = [
        ("1", "1_ctrl", 1.0),
        # ... one entry per (qpos_joint, actuator, coefficient)
    ]
```

Add a path fixture in `tests/hand_models/conftest.py`:
```python
@pytest.fixture
def leap_hand_path(assets_path):
    return assets_path / "leap_hand"
```

Generate golden data by running `scripts/generate_robot_hand_golden_data.py hand=<hand_name>` or manually creating the `.npz` fixture (see existing golden files in `tests/fixtures/` for the expected format).

### 7. Smoke Tests

Run each script to verify end-to-end integration:

```bash
# Offline retargeting
python scripts/run_offline_retargeting.py hand=<hand_name>

# Compute metrics (opens dashboard at http://127.0.0.1:8050/)
python scripts/compute_hand_retargeter_pair_metrics.py hand=<hand_name> retargeter=dexpilot

# Online retargeting (OPTIONAL — requires MANUS gloves + SpaceMouse hardware)
python scripts/run_online_retargeting.py hand=<hand_name>
```

### Summary Checklist

| Step | Files |
|------|-------|
| MJCF model | `assets/mjcf/<hand_name>/right_hand.xml` (+ `left_hand.xml`) |
| Type enum | `mimic_retargeter_lab/types/types.py` |
| Hand model class | `mimic_retargeter_lab/hand_models/<hand_name>.py` |
| Factory registration | `mimic_retargeter_lab/hand_models/__init__.py` |
| Hand config | `config/hand/<hand_name>.yaml` |
| Retargeter configs (x4) | `config/retargeter_cfg/{keyvector,dexpilot,ako,joint_angle}/human_hand_to_<hand_name>.yaml` |
| Metrics configs (x6) | `config/metrics/<hand_name>/{flatness,keyvector_matching,motion_preservation,pinch_grasps,response,workspace}.yaml` |
| Unit tests | `tests/hand_models/test_<hand_name>_model.py` + golden `.npz` fixture |
| Conftest fixture | `tests/hand_models/conftest.py` (add path fixture) |

