"""Common data types and constants used in the repo.

Author(s):
    - Robert Jomar Malate (robert.malate@mimicrobotics.com)
"""

# Standard
from dataclasses import dataclass
from enum import Enum

# Third-party
import numpy as np


# --- TYPES ---
class HandChirality(Enum):
    RIGHT = 0
    LEFT = 1

    def __str__(self) -> str:
        """String representation of HandChirality."""
        return self.name.lower()

    def __int__(self) -> int:
        """Integer representation of HandChirality."""
        return self.value


class FingerName(str, Enum):
    """Enum to represent finger identity."""

    THUMB = "thumb"
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    PINKY = "pinky"


class HandTracker(str, Enum):
    """Enum for hand tracker types."""

    MANUS = "manus"
    AVP = "avp"


@dataclass
class Point:
    """Class representing a 3D position."""

    x: float
    y: float
    z: float

    def __array__(self, dtype) -> np.ndarray:
        """Convert Point to numpy array."""
        return np.array([self.x, self.y, self.z], dtype=dtype)


@dataclass
class Quaternion:
    """Class representing an orientation in quaternion format."""

    x: float
    y: float
    z: float
    w: float

    def __array__(self, dtype) -> np.ndarray:
        """Convert Quaternion to numpy array.

        Returns in order [x, y, z, w]
        """
        return np.array([self.x, self.y, self.z, self.w], dtype=dtype)


@dataclass
class Pose:
    """Class representing a 3D pose with position and orientation."""

    position: Point
    orientation: Quaternion


@dataclass
class JointCmd:
    """Class representing joint command in different formats."""

    ros2_deg: np.ndarray
    urdf_rad: np.ndarray


@dataclass
class Finger:
    """Class representing a finger."""

    name: FingerName
    fingertip_pose: Pose


@dataclass
class Hand:
    """Class representing a hand with multiple fingers."""

    fingers: dict[FingerName, Finger]

    def get_fingertip_pose(self, finger_name: FingerName) -> Pose:
        """Get the pose of a specific fingertip."""
        return self.fingers[finger_name].fingertip_pose


@dataclass
class GraspPrimitive:
    """Class representing a grasp primitive."""

    name: str
    joint_commands: JointCmd
    ground_truth_pose: Hand


# --- CONSTANTS ---
FINGERTIP_INDICES = {
    HandTracker.MANUS: {
        "thumb": 24,
        "index": 5,
        "middle": 10,
        "ring": 20,
        "pinky": 15,
    },
    HandTracker.AVP: {
        "thumb": 4,
        "index": 9,
        "middle": 14,
        "ring": 19,
        "pinky": 24,
    },
}

FINGER_GROUPS = {
    HandTracker.MANUS: {
        "wrist": [0],
        "index": [1, 2, 3, 4, 5],
        "middle": [6, 7, 8, 9, 10],
        "ring": [16, 17, 18, 19, 20],
        "pinky": [11, 12, 13, 14, 15],
        "thumb": [21, 22, 23, 24],
    },
    HandTracker.AVP: {
        "wrist": [0],
        "thumb": [1, 2, 3, 4],
        "index": [5, 6, 7, 8, 9],
        "middle": [10, 11, 12, 13, 14],
        "ring": [15, 16, 17, 18, 19],
        "pinky": [20, 21, 22, 23, 24],
    },
}

# Angle offsets taken from p050.py in mimic_robotics
GC_ANGLE_OFFSETS = np.array(
    [
        30.0,  # thumb_cmc
        -20.0,  # thumb_mcp
        50.0,  # thumb_pp
        50.0,  # thumb_dp
        8.5,  # index_mcp_abd
        15.0,  # index_mcp_flex
        15.0,  # index_pip
        12.5,  # middle_mcp_abd
        15.0,  # middle_mcp_flex
        15.0,  # middle_pip
        17.5,  # ring_mcp_abd
        15.0,  # ring_mcp_flex
        15.0,  # ring_pip
        22.5,  # pinky_mcp_abd
        15.0,  # pinky_mcp_flex
        15.0,  # pinky_pip
    ]
)

GRASP_PRIMITIVES = {
    "idle": GraspPrimitive(
        name="idle",
        joint_commands=JointCmd(
            ros2_deg=np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),  # Fill with appropriate values
            urdf_rad=np.array(
                [
                    0.523599,
                    -0.349066,
                    0.872665,
                    0.872665,
                    0.148353,
                    0.261800,
                    0.261800,
                    0.218166,
                    0.261800,
                    0.261800,
                    0.305433,
                    0.261800,
                    0.261800,
                    0.392699,
                    0.261800,
                    0.261800,
                ]
            ),  # Fill with appropriate values
        ),
        ground_truth_pose=Hand(
            fingers={
                FingerName.THUMB: Finger(
                    name=FingerName.THUMB,
                    fingertip_pose=Pose(
                        position=Point(x=0.071, y=-0.031, z=0.224),
                        orientation=Quaternion(x=-0.474, y=0.419, z=0.624, w=-0.459),
                    ),
                ),
                FingerName.INDEX: Finger(
                    name=FingerName.INDEX,
                    fingertip_pose=Pose(
                        position=Point(x=0.023, y=0.043, z=0.293),
                        orientation=Quaternion(x=0.554, y=-0.052, z=-0.026, w=0.831),
                    ),
                ),
                FingerName.MIDDLE: Finger(
                    name=FingerName.MIDDLE,
                    fingertip_pose=Pose(
                        position=Point(x=-0.007, y=0.047, z=0.301),
                        orientation=Quaternion(x=0.551, y=-0.036, z=0.032, w=0.833),
                    ),
                ),
                FingerName.RING: Finger(
                    name=FingerName.RING,
                    fingertip_pose=Pose(
                        position=Point(x=-0.033, y=0.040, z=0.288),
                        orientation=Quaternion(x=0.556, y=-0.013, z=0.084, w=0.827),
                    ),
                ),
                FingerName.PINKY: Finger(
                    name=FingerName.PINKY,
                    fingertip_pose=Pose(
                        position=Point(x=-0.057, y=0.030, z=0.267),
                        orientation=Quaternion(x=0.558, y=-0.002, z=0.136, w=0.818),
                    ),
                ),
            }
        ),
    ),
    "pinch_index": GraspPrimitive(
        name="pinch_index",
        joint_commands=JointCmd(
            ros2_deg=np.array(
                [
                    52.0,
                    -19.0,
                    -17.0,
                    -45.0,
                    -8.0,
                    70.0,
                    25.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),  # Fill with appropriate values
            urdf_rad=np.array(
                [
                    1.431171,
                    -0.680679,
                    0.575959,
                    0.087267,
                    0.008727,
                    1.483531,
                    0.698132,
                    0.218166,
                    0.261800,
                    0.261800,
                    0.305433,
                    0.261800,
                    0.261800,
                    0.392699,
                    0.261800,
                    0.261800,
                ]
            ),  # Fill with appropriate values
        ),
        ground_truth_pose=Hand(
            fingers={
                FingerName.THUMB: Finger(
                    name=FingerName.THUMB,
                    fingertip_pose=Pose(
                        position=Point(x=0.014, y=-0.053, z=0.236),
                        orientation=Quaternion(x=-0.429, y=0.321, z=0.777, w=-0.330),
                    ),
                ),
                FingerName.INDEX: Finger(
                    name=FingerName.INDEX,
                    fingertip_pose=Pose(
                        position=Point(x=0.014, y=-0.056, z=0.228),
                        orientation=Quaternion(x=0.993, y=-0.035, z=0.100, w=-0.060),
                    ),
                ),
                FingerName.MIDDLE: Finger(
                    name=FingerName.MIDDLE,
                    fingertip_pose=Pose(
                        position=Point(x=-0.007, y=0.047, z=0.301),
                        orientation=Quaternion(x=0.551, y=-0.036, z=0.032, w=0.833),
                    ),
                ),
                FingerName.RING: Finger(
                    name=FingerName.RING,
                    fingertip_pose=Pose(
                        position=Point(x=-0.033, y=0.040, z=0.288),
                        orientation=Quaternion(x=0.556, y=-0.013, z=0.084, w=0.827),
                    ),
                ),
                FingerName.PINKY: Finger(
                    name=FingerName.PINKY,
                    fingertip_pose=Pose(
                        position=Point(x=-0.057, y=0.030, z=0.267),
                        orientation=Quaternion(x=0.558, y=-0.002, z=0.136, w=0.818),
                    ),
                ),
            }
        ),
    ),
    "pinch_middle": GraspPrimitive(
        name="pinch_middle",
        joint_commands=JointCmd(
            ros2_deg=np.array(
                [
                    52.0,
                    -30.0,
                    -10.0,
                    -50.0,
                    0.0,
                    0.0,
                    0.0,
                    -5.0,
                    75.0,
                    30.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            urdf_rad=np.array(
                [
                    1.431171,
                    -0.872665,
                    0.698132,
                    0.000000,
                    0.148353,
                    0.261800,
                    0.261800,
                    0.130900,
                    1.570797,
                    0.785399,
                    0.305433,
                    0.261800,
                    0.261800,
                    0.392699,
                    0.261800,
                    0.261800,
                ]
            ),
        ),
        ground_truth_pose=Hand(
            fingers={
                FingerName.THUMB: Finger(
                    name=FingerName.THUMB,
                    fingertip_pose=Pose(
                        position=Point(x=0.003, y=-0.064, z=0.227),
                        orientation=Quaternion(x=-0.488, y=0.283, z=0.783, w=-0.263),
                    ),
                ),
                FingerName.INDEX: Finger(
                    name=FingerName.INDEX,
                    fingertip_pose=Pose(
                        position=Point(x=0.023, y=0.043, z=0.293),
                        orientation=Quaternion(x=0.554, y=-0.052, z=-0.026, w=0.831),
                    ),
                ),
                FingerName.MIDDLE: Finger(
                    name=FingerName.MIDDLE,
                    fingertip_pose=Pose(
                        position=Point(x=0.001, y=-0.060, z=0.222),
                        orientation=Quaternion(x=0.978, y=0.030, z=0.087, w=-0.190),
                    ),
                ),
                FingerName.RING: Finger(
                    name=FingerName.RING,
                    fingertip_pose=Pose(
                        position=Point(x=-0.033, y=0.040, z=0.288),
                        orientation=Quaternion(x=0.556, y=-0.013, z=0.084, w=0.827),
                    ),
                ),
                FingerName.PINKY: Finger(
                    name=FingerName.PINKY,
                    fingertip_pose=Pose(
                        position=Point(x=-0.057, y=0.030, z=0.267),
                        orientation=Quaternion(x=0.558, y=-0.002, z=0.136, w=0.818),
                    ),
                ),
            }
        ),
    ),
    "pinch_ring": GraspPrimitive(
        name="pinch_ring",
        joint_commands=JointCmd(
            ros2_deg=np.array(
                [
                    55.0,
                    -32.0,
                    5.0,
                    -50.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -8.0,
                    70.0,
                    25.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            urdf_rad=np.array(
                [
                    1.483531,
                    -0.907572,
                    0.959932,
                    0.000000,
                    0.148353,
                    0.261800,
                    0.261800,
                    0.218166,
                    0.261800,
                    0.261800,
                    0.165806,
                    1.483531,
                    0.698132,
                    0.392699,
                    0.261800,
                    0.261800,
                ]
            ),
        ),
        ground_truth_pose=Hand(
            fingers={
                FingerName.THUMB: Finger(
                    name=FingerName.THUMB,
                    fingertip_pose=Pose(
                        position=Point(x=-0.016, y=-0.057, z=0.225),
                        orientation=Quaternion(x=-0.536, y=0.390, z=0.729, w=-0.171),
                    ),
                ),
                FingerName.INDEX: Finger(
                    name=FingerName.INDEX,
                    fingertip_pose=Pose(
                        position=Point(x=0.023, y=0.043, z=0.293),
                        orientation=Quaternion(x=0.554, y=-0.052, z=-0.026, w=0.831),
                    ),
                ),
                FingerName.MIDDLE: Finger(
                    name=FingerName.MIDDLE,
                    fingertip_pose=Pose(
                        position=Point(x=-0.007, y=0.047, z=0.301),
                        orientation=Quaternion(x=0.551, y=-0.036, z=0.032, w=0.833),
                    ),
                ),
                FingerName.RING: Finger(
                    name=FingerName.RING,
                    fingertip_pose=Pose(
                        position=Point(x=-0.016, y=-0.058, z=0.224),
                        orientation=Quaternion(x=0.987, y=0.079, z=0.122, w=-0.072),
                    ),
                ),
                FingerName.PINKY: Finger(
                    name=FingerName.PINKY,
                    fingertip_pose=Pose(
                        position=Point(x=-0.057, y=0.030, z=0.267),
                        orientation=Quaternion(x=0.558, y=-0.002, z=0.136, w=0.818),
                    ),
                ),
            }
        ),
    ),
    "pinch_pinky": GraspPrimitive(
        name="pinch_pinky",
        joint_commands=JointCmd(
            ros2_deg=np.array(
                [
                    55.0,
                    -45.0,
                    10.0,
                    -30.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -20.0,
                    75.0,
                    25.0,
                ]
            ),
            urdf_rad=np.array(
                [
                    1.483531,
                    -1.134465,
                    1.047198,
                    0.349066,
                    0.148353,
                    0.261800,
                    0.261800,
                    0.218166,
                    0.261800,
                    0.261800,
                    0.305433,
                    0.261800,
                    0.261800,
                    0.043633,
                    1.570797,
                    0.698132,
                ]
            ),
        ),
        ground_truth_pose=Hand(
            fingers={
                FingerName.THUMB: Finger(
                    name=FingerName.THUMB,
                    fingertip_pose=Pose(
                        position=Point(x=-0.035, y=-0.061, z=0.208),
                        orientation=Quaternion(x=-0.590, y=0.477, z=0.651, w=0.036),
                    ),
                ),
                FingerName.INDEX: Finger(
                    name=FingerName.INDEX,
                    fingertip_pose=Pose(
                        position=Point(x=0.023, y=0.043, z=0.293),
                        orientation=Quaternion(x=0.554, y=-0.052, z=-0.026, w=0.831),
                    ),
                ),
                FingerName.MIDDLE: Finger(
                    name=FingerName.MIDDLE,
                    fingertip_pose=Pose(
                        position=Point(x=-0.007, y=0.047, z=0.301),
                        orientation=Quaternion(x=0.551, y=-0.036, z=0.032, w=0.833),
                    ),
                ),
                FingerName.RING: Finger(
                    name=FingerName.RING,
                    fingertip_pose=Pose(
                        position=Point(x=-0.033, y=0.040, z=0.288),
                        orientation=Quaternion(x=0.556, y=-0.013, z=0.084, w=0.827),
                    ),
                ),
                FingerName.PINKY: Finger(
                    name=FingerName.PINKY,
                    fingertip_pose=Pose(
                        position=Point(x=-0.030, y=-0.059, z=0.200),
                        orientation=Quaternion(x=0.951, y=0.158, z=0.231, w=-0.131),
                    ),
                ),
            }
        ),
    ),
}
