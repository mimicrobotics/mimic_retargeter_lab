# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import numpy as np
from pathlib import Path
import json
from geort.utils.path import get_package_root, get_human_data_output_path, get_data_root
from geort.utils.config_utils import get_config
from geort.export import load_model


# Logger for the module
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def save_human_data(human_data, tag):
    save_path = get_human_data_output_path(tag)
    np.save(save_path, human_data)
    return save_path


def save_data(
    human_data: np.ndarray, metadata: dict, tag: str, logger: logging.Logger = None
) -> tuple[Path, Path]:
    """
    Saves the numpy data and associated metadata into the geort data directory.
    """
    logger = logger if logger is not None else _logger

    # TODO: Group files into directory with tag name (easier grouping and scalable)
    # Saving files to data/ dir
    save_dir = get_data_root()

    dataset_data_filename = f"dataset_{tag}.npy"
    dataset_data_filepath = save_dir / dataset_data_filename

    dataset_metadata_filename = f"metadata_{tag}.json"
    dataset_metadata_filepath = save_dir / dataset_metadata_filename

    # Saving data to .npy file
    np.save(dataset_data_filepath, human_data)
    info_msg = f"Saving numpy data to {dataset_data_filepath}."
    logger.info(info_msg)

    # Saving metadata to .json
    with open(dataset_metadata_filepath, "w") as f:
        json.dump(metadata, f, indent=4)
    info_msg = f"Saving metadata to {dataset_metadata_filepath}."
    logger.info(info_msg)

    info_msg = f"Successfully exported data for tag: {tag} to directory: {save_dir}"
    logger.info(info_msg)
    return dataset_data_filepath, dataset_metadata_filepath
