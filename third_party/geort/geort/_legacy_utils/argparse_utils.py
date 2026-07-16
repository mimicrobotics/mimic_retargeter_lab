"""Argparse utility functions for the project.

Notes:
- This was generated with help from Gemini.

Author(s):
    - Robert Jomar Malate (robert.malate@mimicrobotics.com)
"""

# Standard
from __future__ import annotations
import argparse
from typing import TypedDict, Any, Sequence
from typing_extensions import NotRequired


class ArgSpec(TypedDict):
    """
    Type definition for argument specification dictionary.
    NotRequired allows us to omit keys (like 'choices' or 'default')
    when they aren't relevant for a specific arg.
    """

    # Mandatory fields
    flag: Sequence[str]  # e.g., ["-f", "--file"]
    help: str  # Help text

    # Optional fields
    type: NotRequired[type | Any]
    default: NotRequired[Any]
    required: NotRequired[bool]
    choices: NotRequired[Sequence[Any]]


STANDARD_ARGS: dict[str, ArgSpec] = {
    "dataset_filename": {
        "flag": ["--dataset_filename"],
        "type": str,
        "help": "Name of .npy dataset file to inspect (datasets that are stored in data/).",
        "required": True,
    },
    "hand": {
        "flag": ["--hand"],
        "type": str,
        "help": "Hand name as defined in GeoRT configs (e.g., 'p50', 'allegro_right')",
        "required": True,
        "default": "p50",
    },
}


def create_argparser(description: str) -> argparse.ArgumentParser:
    """Creates and returns an ArgumentParser with common arguments.

    Args:
        description: Description for the argument parser.
    Returns:
        An argparse.ArgumentParser instance.
    """
    return argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )


# --- Atomic arguments ---
def add_args(
    parser: argparse.ArgumentParser, *names: str, **overrides: ArgSpec
) -> None:
    """
    Adds standard arguments to the parser by name, with optional overrides.

    Args:
        parser: The ArgumentParser to update.
        *names: Keys from STANDARD_ARGS to add (e.g., "dataset_filename", "hand").
        **overrides: Key-value pairs to override specific settings.
    """
    for name in names:
        if name not in STANDARD_ARGS:
            raise ValueError(f"Argument '{name}' is not defined in STANDARD_ARGS.")

        # 1. Copy to avoid mutating the global standard
        # Casting to dict to avoid mypy complaining about modifying TypedDicts dynamically
        arg_spec = dict(STANDARD_ARGS[name]).copy()

        # 2. Apply overrides
        if name in overrides:
            arg_spec.update(overrides[name])  # type: ignore

        # 3. Extract flags (positional args for add_argument)
        # We know flags exists because it's mandatory in ArgSpec
        flags = arg_spec.pop("flag")  # type: ignore

        # 4. Add to parser
        parser.add_argument(*flags, **arg_spec)
