# JAX / GPU setup notes for retargeters

This document captures why running `SamplingBasedRetargeter` on GPU was
non-trivial in dexworld, what the failure modes looked like, and what
the fixes do. Read this if you're installing GPU support, integrating a
new GPU-accelerated retargeter, debugging a "looks GPU but runs at CPU
speed" symptom, or extending the pattern to other retargeters.

## Installing GPU support

Only the `sampling_based` retargeter needs a GPU: it evaluates thousands
of candidate joint configurations in parallel via JAX-batched FK, and is
impractically slow on CPU. Every other retargeter (`keyvector`,
`dexpilot`, `ako`, `joint_angle`, `hybrid`, `geort`) runs fine on CPU
with a plain `uv sync` — if you don't have an NVIDIA GPU, there is
nothing to do here.

With an NVIDIA GPU and a CUDA 12 driver (check with `nvidia-smi`),
install the `gpu` extra to pull a CUDA-enabled `jaxlib`:

```bash
uv sync --extra gpu
```

Verify it took:

```bash
uv run python -c "import jax; print(jax.default_backend(), jax.devices())"
# expect: gpu  [CudaDevice(id=0)]
```

Note that a later plain `uv sync` will silently *uninstall* the CUDA
jaxlib — see **Operational gotchas** below, along with the other traps
this setup has.

## TL;DR

Running JAX on GPU under MJX has four traps that compound. dexworld
addresses all of them:

1. **Process platform default**: pin to `cpu` (plus `cuda` when the CUDA
   plugin is installed) so MJX and other retargeters stay on CPU unless
   they opt in.
2. **MJX device kwargs**: pass `device=` explicitly to `mjx.put_model`
   and `mjx.make_data` — they ignore `jax.default_device(...)`.
3. **Closure-captured arrays**: a JIT'd function specializes to the
   device of arrays in its closure, and `jax.default_device(...)`
   doesn't move them. Rebuild the FK kernel on the target device.
4. **Per-frame data placement**: any persistent JAX state (qpos, RNG
   key, bounds) and any per-frame numpy → JAX inputs must be
   `jax.device_put`-ed to the target device, otherwise every step
   shuffles bytes across PCIe.

Plus a separate gotcha: warmup must use the **same input objects** the
real call uses, or the JIT cache miss on frame 1 costs ~50 s on a fresh
cold cache.

## How the bugs surfaced (and what each fix solved)

### Layer 1 — JAX process default flipped to GPU after `pip install jax[cuda12]`

After installing CUDA-jax, `jax.default_backend()` becomes `gpu` for the
whole process. MJX's `mjx.put_model(...)` and `mjx.make_data(...)`
honor this default for any unspecified `device=` arg, so the hand model
silently lands on GPU. MuJoCo logs this:

```
[INFO] [_resolve_impl_and_device]: Using JAX default device: cuda:0.
```

Even retargeters configured for `device: cpu` triggered this log
because MJX builds the model before the retargeter runs.

**Fix** — pin platform priority so CPU is the process default and CUDA
is still discoverable for retargeters that opt in:

- `dexworld/__init__.py` does
  `os.environ.setdefault("JAX_PLATFORMS", "cpu,cuda" if _HAS_CUDA_PLUGIN else "cpu")`,
  where `_HAS_CUDA_PLUGIN` tests whether `jax_cuda12_plugin` is importable.
  The `cuda` entry is conditional because JAX treats every platform named in
  `JAX_PLATFORMS` as required: on a CPU-only install, listing `cuda` makes the
  first `jax.devices()` call (e.g. from `mjx.put_model`) raise
  `RuntimeError: Unable to initialize backend 'cuda'` rather than fall back to
  CPU, which would break every entry script for users without a GPU.
- Entry scripts (`run_offline_retargeting.py`, `compute_hand_retargeter_pair_metrics.py`,
  `precompute_workspace.py`) `import dexworld` at the very top —
  **before** `import jax`.

The order matters: JAX caches the platform priority during config-object
construction, which runs on `import jax`. Setting the env var after
that point is too late. `dexworld/__init__.py` is intentionally tiny
(no `import jax`) so importing it can't trigger backend init.

Override at the shell with `JAX_PLATFORMS=cuda` (etc.) to force GPU
process-wide.

### Layer 2 — Stale CPU-compiled artifact in JAX's persistent cache

Once the process default was CPU, the hand model's MJX FK was built on
CPU. `SamplingBasedRetargeter` wrapped its CEM step in
`with jax.default_device(cuda):`, but **`jax.default_device(...)` does
not re-place arrays captured by closure**. The JIT'd FK
(`_jax_fk_batch_ctrl`) closed over a CPU-resident `mjx_model` and a
batch of CPU-pinned helper arrays, so it kept running on CPU.

Worse, the JAX persistent cache (under
`~/.cache/dexworld/jax_compilation`) had previously saved a CPU-compiled
binary from earlier runs. JAX cheerfully loaded it. Symptom: warmup
"completed in 2.95 s" — way too fast for a real GPU compile, because
the cache was hitting a CPU-targeted artifact and reusing it. The robot
moved at CPU speed but every diagnostic looked GPU-shaped.

**Fix** — let consumers rebuild MJX FK on the device they'll invoke it
from:

- `RobotHandModel.create_mjx_kinematic_model(device=None)` now takes an
  optional `jax.Device`. When provided, it wraps the
  JAX-array-creation block in `jax.default_device(device)` and passes
  `device=device` explicitly to **both** `mjx.put_model(...)` and
  `mjx.make_data(...)` — MJX has its own
  `_resolve_impl_and_device` path that bypasses
  `jax.default_device(...)`, so the explicit kwargs are required.
- `SamplingBasedRetargeter.__init__` calls
  `to_model.create_mjx_kinematic_model(device=self._device)` when
  `self._device.platform != "cpu"`. The MJX kernel + closures are now
  resident on the target device.

Verification: after this change, warmup with a cleared cache took
~55 s — a real cold compile on GPU. Subsequent runs hit the (now
GPU-targeted) cache and warmup drops to ~3 s.

### Layer 3 — Persistent state and per-frame inputs were on the wrong device

The FK kernel was on GPU, but several JAX arrays the JIT'd step
accepted as inputs were created on CPU before the retargeter knew its
device:

- `_qpos_prev` (initialized in `BaseOnlineRetargeter`)
- `_lower_bounds`, `_upper_bounds` (built on CPU during `_init_bounds`)
- `_q_neutral` (created via `jnp.asarray`)
- `_jax_rng_key` (created via `jax.random.PRNGKey`)

And per frame, `_compute_dexpilot_loss_terms` returned `targets` and
`weights` placed on the JAX default (CPU). Every per-frame call to the
GPU step was bouncing CPU → GPU → CPU.

**Fix** — explicit placement:

- In `__init__`, `jax.device_put(...)` each of the five persistent
  arrays onto `self._device`.
- In `retarget()`, `jax.device_put(targets, self._device)` and
  `jax.device_put(weights, self._device)` per frame.
- Moved `jnp.clip(qpos_next_j, ...)` inside the `default_device`
  block so the output stays GPU-resident for the next frame.

### Layer 4 — Frame 1 paid a 50 s recompile

Even with everything pinned, frame 1 took ~50 s while frames 2+ took
~4 ms. JAX's JIT cache key includes input array "device commitment",
and the warmup's synthetic inputs (`jnp.asarray(self._q_neutral)`,
freshly built `jnp.zeros`/`jnp.ones`) didn't byte-match the real
call's pinned inputs. Frame 1 was a cache miss → recompile.

**Fix** — make warmup use the **same input objects** the real call
uses:

```python
# in _warmup_jax_compilation
targets_warm = jax.device_put(jnp.zeros(...), self._device)
weights_warm = jax.device_put(jnp.ones(...), self._device)
self._monte_carlo_step_fn(
    self._jax_rng_key,   # same array as retarget()
    self._qpos_prev,     # same array as retarget()
    targets_warm,
    weights_warm,
)
```

With this aligned, frame 1 dropped from ~50 s to ~35 ms.

## End-to-end timing on a 4090

Hand: `orca_v2_hand`, 17 actuated DoFs, 8192 samples × 20 update
cycles per frame.

| Phase                          | Cold cache | Warm cache |
|--------------------------------|-----------|-----------|
| Hand model load + GPU rebuild  | ~10 s     | ~10 s     |
| JAX warmup (compile)           | ~55 s     | ~3 s      |
| Frame 1                        | ~35 ms    | ~35 ms    |
| Frames 2+                      | ~4 ms     | ~4 ms     |
| Total time-to-first-motion     | ~65 s     | ~13 s     |

Per-frame throughput is ≈ 250 Hz; the bottleneck after warmup is the
data loader, not the optimizer.

## Key APIs

### `dexworld.utils.resolve_jax_device(device_str, logger=None)`
Resolve `"cpu"` / `"cuda"` / `"gpu"` to a `jax.Device`. Falls back to
CPU with a logger warning if a GPU was requested but no CUDA jaxlib is
available. Used by all four retargeters.

### `dexworld.utils.rebuild_mjx_fk_on_device(to_model, device, logger=None)`
No-op when `device.platform == "cpu"`. Otherwise calls
`to_model.create_mjx_kinematic_model(device=device)` to re-place the
MJX model + JIT'd FK functions on `device`. Required because closure-
captured MJX arrays don't follow `jax.default_device(...)`. Every
retargeter that runs a JIT'd loss closing over `compute_keyvectors_jax`
should call this right after device resolution.

Also no-op when the FK is already pinned to the requested `device`
(a second consumer on the same device shares the existing build). Raises
`RuntimeError` if the FK is already pinned to a *different* non-CPU
device — see [One hand model, one device](#one-hand-model-one-device).

### `dexworld.utils.device_put_attrs(obj, attr_names, device)`
In-place `jax.device_put` for a batch of named JAX attributes on
`obj`. Skips attributes whose value is `None`; raises
`AttributeError` on a typo. The standard call site is at the end of a
retargeter's `__init__` once all persistent JAX state is built:

```python
device_put_attrs(
    self,
    ("_lower_bounds", "_upper_bounds", "_qpos_prev", ...),
    self._device,
)
```

### `RobotHandModel.create_mjx_kinematic_model(device=None)`
Build (or rebuild) the MJX model + JIT'd FK functions on the given
device. Pass `device=jax.devices('cuda')[0]` when the FK kernel must be
GPU-resident. Default `None` honors the JAX process default. Most
callers don't invoke this directly — `rebuild_mjx_fk_on_device` does.

### Retargeter config

All four retargeters expose a `device` field, defaulting to `cpu`:

| Retargeter           | YAML location                  |
|----------------------|--------------------------------|
| `keyvector`          | top-level `config.device`      |
| `dexpilot`           | `config.solver_params.device`  |
| `ako`                | `config.solver_params.device`  |
| `sampling_based`     | `config.sampling_params.device`|

```yaml
# Example: sampling_based on GPU
sampling_params:
  device: cuda            # cpu | gpu | cuda
  jax_compilation_cache_enable: true
  jax_compilation_cache_dir: ~/.cache/dexworld/jax_compilation
  num_samples: 8192
  num_samples_elite: 128
  ...
```

You can mix devices across retargeters in a single process — JIT
artifacts are cached per device.

## One hand model, one device

`rebuild_mjx_fk_on_device` mutates the hand model in place — it
replaces `to_model._mjx_model` and the JIT'd FK closures attached to
the same `to_model` instance. The rule that falls out:

> **A given `to_model` instance can have its MJX FK pinned to at most
> one non-CPU device per process.**

Today every entry script constructs one retargeter per run from one
`to_model`, so this is invisible. It only matters if a future script
shares a hand model across retargeters.

**Allowed**:

- Two retargeters sharing one `to_model`, both on CPU.
- Two retargeters sharing one `to_model`, both on the same `cuda:N`
  (the second `rebuild_mjx_fk_on_device` is a no-op).
- Two `to_model` instances on different devices in the same process.
- The same retargeter class run on CPU in one process and GPU in
  another.

**Trips a `RuntimeError`** (one scenario): the same `to_model`
instance receiving `rebuild_mjx_fk_on_device` calls for two different
non-CPU devices, e.g. `cuda:0` then `cuda:1`. The error points at
the fix: build a separate `to_model` per device.

```python
# Don't:
hand = ShadowHand(...)
ret_a = SamplingBasedRetargeter(to_model=hand, device="cuda:0")
ret_b = SamplingBasedRetargeter(to_model=hand, device="cuda:1")  # raises

# Do:
hand_a = ShadowHand(...)
hand_b = ShadowHand(...)
ret_a = SamplingBasedRetargeter(to_model=hand_a, device="cuda:0")
ret_b = SamplingBasedRetargeter(to_model=hand_b, device="cuda:1")
```

**One subtlety not caught by the guard**: sharing a `to_model` between
a CPU retargeter constructed *first* and a GPU retargeter constructed
*second* is permitted (the GPU rebuild sees `current=None` or
`current.platform=="cpu"` and proceeds). The CPU retargeter's JIT'd
loss is safe — JAX froze the FK trace at compile time — but any
*non-JIT* code path on the CPU retargeter that later reaches into
`to_model._mjx_model` will see the GPU model. Not currently a problem
in this codebase; if you hit it, build two `to_model` instances.

## Operational gotchas

- **Stale JAX cache**: if you change device targets and see fishy
  warmup times (sub-second on fresh code, or "fast warmup but slow
  frames"), nuke `~/.cache/dexworld/jax_compilation` and the
  in-repo `.jax_cache` directory. JAX will rebuild a correctly-targeted
  artifact and cache it.
- **`uv sync` without `--extra gpu`** removes the CUDA jaxlib. After
  installing GPU deps, either always pass `--extra gpu` or set
  `UV_EXTRA=gpu` in your shell.
- **Sharing the venv across machines**: the JAX cache and the
  CUDA-jaxlib install are machine-specific. Don't copy them.
- **`dexworld/__init__.py` not running after a fresh `uv sync`**: if
  you add or modify side effects in the package init and they don't
  fire (env var unset, log filter not applied), the editable install's
  metadata may have been generated when the package had no
  `__init__.py` and is now treating dexworld as a namespace package
  (you'll see `dexworld.__file__ is None`). Fix:
  `uv sync --extra gpu --reinstall-package dexworld`.

## Adding a new GPU-capable retargeter

All four existing retargeters apply Layers 1–4 the same way through
the shared helpers. Mirror this pattern in any new retargeter that
runs a JIT'd JAX loss and you'll be GPU-correct out of the box.

### 1. In `__init__`, after resolving `self._device`

```python
from dexworld.utils import (
    device_put_attrs,
    rebuild_mjx_fk_on_device,
    resolve_jax_device,
)

# resolve the device knob from your YAML (defaults to CPU)
self._device = resolve_jax_device(device_str, logger=self._logger)

# Layer 2: re-place MJX FK on the target device. No-op for CPU.
rebuild_mjx_fk_on_device(self.to_model, self._device, logger=self._logger)

# ... build bounds, normalize keyvector cfg, allocate persistent JAX state ...

# Layer 3: pin every persistent JAX array we hand to the JIT'd loss.
device_put_attrs(
    self,
    (
        "_lower_bounds", "_upper_bounds", "_qpos_prev",
        # ... plus any retargeter-specific JAX arrays the loss closes over
    ),
    self._device,
)

# JIT-compile the loss/solver under the target device
with jax.default_device(self._device):
    self._loss_fn = self._init_loss_fn()
    self._solver = ...
```

### 2. In your per-frame `_optimize_controls`

```python
def _optimize_controls(self, qpos_init_guess, targets, weights):
    with jax.default_device(self._device):
        # Per-frame inputs arrive from numpy/CPU paths — push them.
        targets = jax.device_put(targets, self._device)
        weights = jax.device_put(weights, self._device)
        qpos_clipped = jnp.clip(
            qpos_init_guess, self._lower_bounds, self._upper_bounds,
        )
        result = self._solver.run(qpos_clipped, ..., targets=targets, weights=weights)
        # Keep the post-processing inside the block so the output stays
        # on-device for the next frame.
        qpos_optimized = jnp.clip(
            result.params, self._lower_bounds, self._upper_bounds,
        )
    return qpos_optimized
```

### 3. If you have a warmup

Use the **same input objects** the real call will use
(`self._qpos_prev`, `self._jax_rng_key`, persistent attrs that have
already been `device_put`-ed) plus `jax.device_put`-ed dummy
targets/weights. JAX's JIT cache key includes input device commitment,
so synthetic constants built fresh inside warmup miss the cache and
trigger a 50 s recompile on frame 1.

### Smoke checks

- CPU run: no `Rebuilding MJX FK kernel` log.
- GPU run: `Rebuilding MJX FK kernel on cuda:0 ...` fires once at
  init; `nvidia-smi` shows non-trivial GPU memory; first frame is
  fast (cache hit, not the 50 s recompile path).
