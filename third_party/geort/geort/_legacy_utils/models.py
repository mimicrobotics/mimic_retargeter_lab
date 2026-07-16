"""Helper functions related to datasets.

Author(s):
    - Robert Jomar Malate (robert.malate@mimicrobotics.com)
"""

# Standard
# Third-party
# Custom
from geort._legacy_utils.common import (
    HandTracker,
)


def extract_hand_tracker_from_checkpoint_name(
    checkpoint_name: str,
) -> str | None:
    """Extracts the hand tracker type from the checkpoint filename.

    Args:
        checkpoint_name: Name of the checkpoint file.
    Returns:
        Hand tracker type as a string or None if not found.
    """
    for hand_tracker in HandTracker:
        if hand_tracker.value in checkpoint_name.lower():
            return hand_tracker.value

    return None
