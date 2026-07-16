import numcodecs
import numpy as np
import zarr
from pathlib import Path

# Datasets are written in the zarr v2 layout so they stay readable by both zarr
# 2.x and 3.x (pyproject allows either), and so they match the stores already
# committed under ``dataset/``.
_ZARR_FORMAT = 2


class HumanDataWriter:
    def __init__(self, base_dir: Path, dataset_name: str):
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.out_file = self.base_dir / f"{self.dataset_name}.zarr"

    def flatten_dict(self, d):
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                sub_dict = self.flatten_dict(value)
                for sub_key, sub_value in sub_dict.items():
                    result[f"{key}.{sub_key}"] = sub_value
            else:
                result[key] = value
        return result

    def write_recursive(self, root, data_dict):
        for key, value in data_dict.items():
            if value is None:
                continue
            if isinstance(value, dict):
                sub_root = root.create_group(key)
                self.write_recursive(sub_root, value)
            else:
                value = np.array(value)
                root.create_dataset(
                    name=key,
                    data=value,
                    # ``numcodecs.Blosc`` rather than ``zarr.Blosc``: the latter
                    # was removed in zarr 3.
                    compressor=numcodecs.Blosc(
                        cname="lz4", clevel=1, shuffle=numcodecs.Blosc.BITSHUFFLE
                    ),
                    shape=value.shape,
                    dtype=value.dtype,
                )

    def write_episode(self, episode_id, data_dict, verbose=False):
        if verbose:
            print(f"Writing episode {episode_id}...")
        # No context manager: in zarr 3 ``zarr.open`` returns a Group, which is
        # not a context manager (it was in zarr 2).
        root = zarr.open(self.out_file, mode="a", zarr_format=_ZARR_FORMAT)
        group = root.create_group(episode_id)
        self.write_recursive(group, data_dict)
