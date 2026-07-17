# Hand Trackers

Hand trackers are live data sources that stream hand pose observations into the retargeting pipeline. Each tracker implements `BaseHandDataSource` (defined in `mimic_retargeter_lab/data_sources/base_hand_data_source.py`) and must provide a `get_iter()` generator that yields dicts with:

- `transforms` — `np.ndarray` of shape `(21, 4, 4)`: per-joint 4x4 homogeneous transforms
- `joints` — `np.ndarray` of shape `(1, 21, 3)`: 3D joint positions (MANO 21-point topology)
- `links` — `list`: skeleton edges for debug visualization (can be empty)

## Available Trackers

### OAK-D (MediaPipe)

Uses a DepthAI OAK-D camera with MediaPipe hand detection.

**Config:** `config/hand_tracker/oakd_mediapipe.yaml`

**Dependencies:** `depthai`, `depthai-sdk` (in `pyproject.toml`)

**Usage:**
```bash
uv run python scripts/run_online_retargeting.py hand_tracker=oakd_mediapipe
```

### MANUS Metagloves Pro

Uses MANUS data gloves via a ZMQ bridge that reads from the MANUS Core SDK.

**Config:** `config/hand_tracker/manus_metagloves_pro.yaml`

**Dependencies:** `pyzmq` (in `pyproject.toml`), plus the native MANUS SDK setup below.

#### Setup

1. **Obtain the MANUS SDK.** Contact MANUS support (support@manus-meta.com) to get the Linux installer (.deb). This is not publicly available.

2. **Install MANUS Core.** Run the .deb installer — this provides the `manuscore` daemon and SDK libraries.

3. **Copy shared libraries.** Place `libManusSDK.so` and `libManusSDK_Integrated.so` into:
   ```
   mimic_retargeter_lab/data_sources/hand_trackers/manus_hand_tracker/ManusSDK/lib/
   ```
   These files are `.gitignore`'d due to their size (~250MB total).

4. **Install build dependencies:**
   ```bash
   sudo apt install build-essential libzmq3-dev libncurses5-dev
   ```

5. **Build the ZMQ bridge:**
   ```bash
   cd mimic_retargeter_lab/data_sources/hand_trackers/manus_hand_tracker/SDKClient_Linux
   make
   ```
   This produces `SDKClient_Linux.out`.

6. **Calibrate your gloves** using the MANUS calibration tool (run once per user).

7. **Update the landmark map.** The config file (`config/hand_tracker/manus_metagloves_pro.yaml`) contains a `landmark_map` that maps each of the 21 MANO hand topology indices to a MANUS skeleton node index. Inspect the raw MANUS output and adjust these mappings to match your glove's skeleton.

#### Running

```bash
# Terminal 1: start MANUS Core daemon
sudo systemctl start manuscore

# Terminal 2: start ZMQ bridge
cd mimic_retargeter_lab/data_sources/hand_trackers/manus_hand_tracker/SDKClient_Linux
./SDKClient_Linux.out

# Terminal 3: run online retargeting
uv run python scripts/run_online_retargeting.py \
    hand_tracker=manus_metagloves_pro \
    retargeter=dexpilot \
    hand=shadow_hand
```

## Adding a New Tracker

1. Create a new file in this directory (e.g., `my_tracker.py`).
2. Subclass `BaseHandDataSource` and implement `get_iter()`.
3. Create a Hydra config under `config/hand_tracker/` with a `_target_` pointing to your class.
4. Re-export in `__init__.py` and `mimic_retargeter_lab/data_sources/__init__.py`.
5. Run with `hand_tracker=<your_config_name>`.
