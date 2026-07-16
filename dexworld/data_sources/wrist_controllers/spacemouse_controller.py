import logging
import time

import numpy as np
import omegaconf
import pyspacemouse
from scipy.spatial.transform import Rotation

from dexworld.data_sources.base_wrist_data_source import BaseWristDataSource

logger = logging.getLogger(__name__)


class SpaceMouseController(BaseWristDataSource):
    """Reads a 6DOF SpaceMouse via HID and integrates twist into an SE3 wrist pose.

    Uses ``pyspacemouse`` to read directly from the device (no daemon needed).
    Each iteration yields the current absolute SE3 wrist pose accumulated from
    velocity inputs, plus the raw twist and button state.
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        self.cfg = cfg
        self._linear_scale = float(cfg.get("linear_scale", 0.001))
        self._angular_scale = float(cfg.get("angular_scale", 0.005))
        self._dead_zone = float(cfg.get("dead_zone", 0.05))
        self._poll_rate_hz = float(cfg.get("poll_rate_hz", 200.0))
        self._device_name = cfg.get("device_name", None)
        self._translation_frame = str(cfg.get("translation_frame", "world"))
        self._rotation_frame = str(cfg.get("rotation_frame", "body"))

        # Current integrated pose (4x4 homogeneous).
        self._wrist_pose = np.eye(4, dtype=np.float64)
        self._device = None

    def reset_pose(self, pose=None):
        """Reset the integrated pose to identity or a given SE3 matrix."""
        if pose is None:
            self._wrist_pose = np.eye(4, dtype=np.float64)
        else:
            self._wrist_pose = np.array(pose, dtype=np.float64).copy()

    def _open_device(self):
        """Open the SpaceMouse HID device."""
        kwargs = {"nonblocking": True}
        if self._device_name:
            kwargs["device"] = str(self._device_name)
        self._device = pyspacemouse.open(**kwargs)
        logger.info(
            f"SpaceMouseController opened device"
            f"{' (' + self._device_name + ')' if self._device_name else ''}"
        )

    def _apply_dead_zone(self, value: float) -> float:
        """Zero out values below the dead-zone threshold."""
        if abs(value) < self._dead_zone:
            return 0.0
        return value

    def get_iter(self):
        self._open_device()
        dt = 1.0 / self._poll_rate_hz

        try:
            while True:
                state = self._device.read()

                # Extract and dead-zone raw axes (each in [-1, 1]).
                lin_x = self._apply_dead_zone(state.x)
                lin_y = self._apply_dead_zone(state.y)
                lin_z = self._apply_dead_zone(state.z)
                ang_roll = self._apply_dead_zone(state.roll)
                ang_pitch = self._apply_dead_zone(state.pitch)
                ang_yaw = self._apply_dead_zone(state.yaw)

                # Scale to physical units.
                dx = lin_x * self._linear_scale
                dy = lin_y * self._linear_scale
                dz = lin_z * self._linear_scale
                drx = ang_pitch * self._angular_scale
                dry = ang_roll * self._angular_scale
                drz = ang_yaw * self._angular_scale

                twist = np.array([dx, dy, dz, drx, dry, drz], dtype=np.float64)

                # Integrate translation.
                d_pos = np.array([dx, dy, dz])
                if self._translation_frame == "body":
                    self._wrist_pose[:3, 3] += self._wrist_pose[:3, :3] @ d_pos
                else:  # world
                    self._wrist_pose[:3, 3] += d_pos

                # Integrate rotation via rotation vector (exponential map).
                rotvec = np.array([drx, dry, drz])
                if np.linalg.norm(rotvec) > 1e-10:
                    dR = Rotation.from_rotvec(rotvec).as_matrix()
                    if self._rotation_frame == "body":
                        self._wrist_pose[:3, :3] = self._wrist_pose[:3, :3] @ dR
                    else:  # world
                        self._wrist_pose[:3, :3] = dR @ self._wrist_pose[:3, :3]

                buttons = {
                    "pressed": list(state.buttons) if state.buttons else [],
                }

                yield {
                    "wrist_pose": self._wrist_pose.copy(),
                    "twist": twist,
                    "buttons": buttons,
                }

                time.sleep(dt)

        except KeyboardInterrupt:
            logger.info("SpaceMouseController interrupted.")
        finally:
            self.close()

    def close(self):
        """Clean up the HID device."""
        if self._device is not None:
            self._device.__exit__(None, None, None)
            self._device = None
            logger.info("SpaceMouseController closed.")
