"""mimic_retargeter_lab package init.

Two side effects, both intentionally cheap (this module must NOT
``import jax``, ``import mujoco``, etc., since entry scripts rely on
importing it before those):

1. Pin JAX's process platform priority. CPU is always the default;
   ``cuda`` is appended only when the CUDA plugin is installed (the
   ``gpu`` extra), so retargeters can opt in via ``device: cuda``.
   Every platform named in ``JAX_PLATFORMS`` is *required* — JAX raises
   rather than falling back if one fails to initialize — so listing
   ``cuda`` on a CPU-only install would break every entry script.
   Override with ``JAX_PLATFORMS=cuda`` (etc.) in the shell when you
   need a different default.

2. Drop MJX's ``Using JAX default device: ...`` info log. It fires
   every time MJX places a model or builds an FK kernel and is
   misleading because it reports the JAX process default, not where
   the retargeter's actual work runs. Filtered by function name
   (``_resolve_impl_and_device``) so we don't accidentally swallow
   unrelated info logs.

Setting the env var must happen before ``import jax``: JAX caches the
platform priority during config-object construction. Entry scripts
that do ``import jax`` at top level should ``import mimic_retargeter_lab`` first.
"""

import importlib.util
import logging
import os

_HAS_CUDA_PLUGIN = importlib.util.find_spec("jax_cuda12_plugin") is not None
os.environ.setdefault("JAX_PLATFORMS", "cpu,cuda" if _HAS_CUDA_PLUGIN else "cpu")


class _MjxDeviceLogFilter(logging.Filter):
    """Drop MJX's ``Using JAX default device: ...`` info log.

    MJX calls ``logging.info(...)`` (root logger) from
    ``_resolve_impl_and_device`` whenever it needs to pick a default
    device — typically twice per process (model placement + FK kernel
    build). The message reports the JAX process default, which is now
    pinned to CPU even when a sampling retargeter is using GPU, so it
    just confuses readers of the log. Filtering by function name keeps
    the filter targeted.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return record.funcName != "_resolve_impl_and_device"


logging.getLogger().addFilter(_MjxDeviceLogFilter())
