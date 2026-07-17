from __future__ import annotations

import jax.numpy as jnp


def compute_keyvectors(frames: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    """Compute keyvectors from a dictionary of frames.

    Args:
        frames: A dictionary of frames. Frames are 4x4 transformation matrices.

    Returns:
        A dictionary of keyvectors.
    """
    keys = list(frames.keys())
    num_keys = len(keys)

    keyvectors: dict[str, jnp.ndarray] = {}
    for i in range(num_keys):
        for j in range(i + 1, num_keys):
            trans_i = frames[keys[i]][:, :3, 3]
            trans_j = frames[keys[j]][:, :3, 3]
            keyvectors[f"{keys[i]}_to_{keys[j]}"] = trans_j - trans_i

    return keyvectors
