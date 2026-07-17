"""JAX device-placement helpers shared across retargeters.

Lets retargeter classes accept a string from a YAML config (e.g. ``"cpu"``,
``"cuda"``, ``"gpu"``) and pin their JIT compilation + per-frame dispatch
to that device via ``jax.default_device(...)``. JIT artifacts are cached
per device, so a single process can mix CPU and GPU retargeters.

The trio of helpers below is what every retargeter needs to be GPU-correct:

- :func:`resolve_jax_device` parses the YAML string.
- :func:`rebuild_mjx_fk_on_device` re-places the hand model's MJX kernel on
  the target device — required because closure-captured MJX arrays don't
  follow ``jax.default_device(...)``.
- :func:`device_put_attrs` moves a batch of persistent JAX attributes on a
  retargeter instance to the target device in one line, so per-frame calls
  don't host->device copy them every step.

See ``docs/jax_gpu_setup.md`` for the four-layer story behind these helpers.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import jax


def resolve_jax_device(
    device_str: str,
    logger: logging.Logger | None = None,
) -> jax.Device:
    """Resolve a string (``"cuda"``, ``"gpu"``, ``"cpu"``) to a JAX device.

    Falls back to CPU with a warning if a GPU was requested but no
    CUDA-enabled jaxlib is available (e.g. ``jax[cuda12]`` not installed).
    """
    target = device_str.lower()
    if target in ("cuda", "gpu"):
        for backend in ("cuda", "gpu"):
            try:
                devs = jax.devices(backend)
            except RuntimeError:
                continue
            if devs:
                return devs[0]
        if logger is not None:
            logger.warning(
                "GPU requested but no CUDA-enabled jaxlib detected; falling "
                "back to CPU. Install jax[cuda12] for GPU support."
            )
        return jax.devices("cpu")[0]
    if target == "cpu":
        return jax.devices("cpu")[0]
    raise ValueError(
        f"Unknown device {device_str!r}; expected 'cuda', 'gpu', or 'cpu'."
    )


def rebuild_mjx_fk_on_device(
    to_model: Any,
    device: jax.Device,
    logger: logging.Logger | None = None,
) -> None:
    """Re-place the hand model's MJX FK kernel on ``device`` if it isn't CPU.

    Required when a retargeter wraps its JIT'd loss in
    ``jax.default_device(device)`` but the hand model was constructed
    earlier (typically on the JAX process default, which mimic_retargeter_lab pins to
    CPU). Closure-captured MJX arrays don't follow
    ``jax.default_device(...)``, so without this rebuild the FK silently
    runs on the wrong device and per-frame batches shuffle across PCIe.

    No-op when ``device.platform == "cpu"`` — the hand model's original
    CPU build is already correct, *or* when the FK is already pinned to
    ``device`` (a second consumer on the same device shares the build).

    Raises ``RuntimeError`` if the FK is already pinned to a *different*
    non-CPU device: rebuilding would silently invalidate the prior
    consumer's closure-captured kernel. Use a separate ``to_model``
    instance per device instead.
    """
    if device.platform == "cpu":
        return
    current = getattr(to_model, "_mjx_fk_device", None)
    if current == device:
        return
    if current is not None and current.platform != "cpu":
        raise RuntimeError(
            f"Hand model's MJX FK is already pinned to {current}; cannot "
            f"rebuild on {device} without invalidating the prior consumer's "
            f"closure-captured kernel. Use a separate hand model instance "
            f"per device."
        )
    if logger is not None:
        logger.info(
            f"Rebuilding MJX FK kernel on {device} so per-frame FK stays on-device."
        )
    to_model.create_mjx_kinematic_model(device=device)


def device_put_attrs(
    obj: Any,
    attr_names: Iterable[str],
    device: jax.Device,
) -> None:
    """In-place: ``jax.device_put`` each named JAX attribute of ``obj``.

    Convenience for pinning a retargeter's persistent JAX state (bounds,
    qpos_prev, per-keyvector weight arrays, etc.) on a single target
    device. Otherwise these stay on the JAX process default, and every
    per-frame call to a JIT'd loss compiled for ``device`` triggers a
    host->device copy.

    Skips attributes whose value is ``None``. Raises ``AttributeError``
    if a name doesn't exist on ``obj`` — typos should be loud.
    """
    for name in attr_names:
        val = getattr(obj, name)
        if val is None:
            continue
        setattr(obj, name, jax.device_put(val, device))
