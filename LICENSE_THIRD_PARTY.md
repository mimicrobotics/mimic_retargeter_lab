# Third-Party Licenses

This project bundles or depends on the following third-party components.

---

## geort

**Location:** `third_party/geort/`
**License:** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
**Source:** https://github.com/eleramp/geometric-retargeting

This is the primary reason the overall project is licensed CC BY-NC 4.0.

---

## Hand URDFs / MJCFs

**Source:** [dex-urdf](https://github.com/dexsuite/dex-urdf)
**License:** Apache 2.0

**Source:** [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
**License:** Apache 2.0 (per-model licenses vary — see individual model directories)

---

## MANUS SDK

**Location:** `mimic_retargeter_lab/data_sources/hand_trackers/manus_hand_tracker/zmq_bridge/` (bridge only)
**License:** Proprietary — Manus Software License Agreement

The MANUS SDK binaries and headers are **not** distributed with this repository.
Only the ZMQ bridge wrapper (our own code) is included. You must obtain the SDK
separately from Manus if you need live glove support.

---

## Other Dependencies

All Python package dependencies (PyTorch, MuJoCo, JAX, etc.) are listed in
`pyproject.toml` with permissive licenses (MIT, Apache 2.0, BSD). See each
package's own license for details.
