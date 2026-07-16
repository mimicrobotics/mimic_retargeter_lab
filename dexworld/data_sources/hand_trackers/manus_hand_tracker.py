import json
import logging

import numpy as np
import omegaconf
import zmq
from scipy.spatial.transform import Rotation

from dexworld.data_sources.base_hand_data_source import BaseHandDataSource
from dexworld.hand_models.manus_hand import ManusHandModel

logger = logging.getLogger(__name__)

# Number of nodes in the MANUS raw skeleton stream.
_MANUS_NUM_NODES = 25


def _quat_pos_to_4x4(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from position (3,) and quaternion (4,).

    Quaternion is expected in (x, y, z, w) order (scipy convention).
    """
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    T[:3, 3] = position
    return T


class ManusHandTracker(BaseHandDataSource):
    """Reads MANUS Metagloves Pro skeleton data via ZMQ from the ManusZmqBridge.

    The ManusZmqBridge binary publishes JSON skeleton frames on a ZMQ PUB
    socket with topic prefixes "RIGHT" and "LEFT".  This tracker subscribes
    to the topic matching the configured chirality and yields frames in the
    format the retargeting pipeline expects.
    """

    _NUM_JOINTS = _MANUS_NUM_NODES  # 25 nodes from MANUS

    def __init__(self, cfg: omegaconf.DictConfig, logger=logging.getLogger(__name__)):
        self.cfg = cfg
        self.logger = logger
        self.hand_model = ManusHandModel()

        self._host = cfg.get("host", "127.0.0.1")
        self._port = int(cfg.get("port", 8000))

        # Chirality determines which ZMQ topic to subscribe to.
        chirality = str(cfg.get("chirality", "right")).upper()
        self._topic = chirality  # "RIGHT" or "LEFT"

        self._ctx = None
        self._sock = None

    def _connect(self):
        """Open the ZMQ SUB socket and subscribe to the chirality topic."""
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.CONFLATE, True)
        self._sock.setsockopt_string(zmq.SUBSCRIBE, self._topic)
        endpoint = f"tcp://{self._host}:{self._port}"
        self._sock.connect(endpoint)
        self.logger.info(
            f"ManusHandTracker connected to {endpoint}, subscribed to '{self._topic}'"
        )

    def _parse_frame(self, raw: str) -> dict | None:
        """Parse a topic-prefixed JSON message.

        Format: "RIGHT {json...}" or "LEFT {json...}"
        Returns the parsed JSON dict, or None on failure.
        """
        # Strip topic prefix (everything before the first space)
        space_idx = raw.find(" ")
        if space_idx == -1:
            self.logger.warning("Malformed ZMQ message (no topic prefix). Skipping.")
            return None

        json_str = raw[space_idx + 1 :]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parse error: {e}. Skipping frame.")
            return None

    def get_iter(self):
        self._connect()

        try:
            while True:
                msg = self._sock.recv_string()
                frame = self._parse_frame(msg)
                if frame is None:
                    continue

                nodes = frame.get("nodes", [])
                if len(nodes) < _MANUS_NUM_NODES:
                    self.logger.warning(
                        f"Expected {_MANUS_NUM_NODES} nodes, got {len(nodes)}. "
                        "Skipping frame."
                    )
                    continue

                # Build arrays indexed by MANUS node ID (0..24).
                # The ManusHandModel.get_landmarks() indexes into these
                # by MANUS node ID directly.
                node_by_id = {n["id"]: n for n in nodes}

                joints = np.zeros((_MANUS_NUM_NODES, 3), dtype=np.float32)
                transforms = np.zeros((_MANUS_NUM_NODES, 4, 4), dtype=np.float32)

                for node_id in range(_MANUS_NUM_NODES):
                    node = node_by_id.get(node_id)
                    if node is None:
                        continue

                    pos = np.array(node["pos"], dtype=np.float32)
                    # MANUS SDK quaternion order is (w, x, y, z);
                    # scipy expects (x, y, z, w).
                    w, x, y, z = node["quat"]
                    quat_scipy = np.array([x, y, z, w], dtype=np.float32)

                    joints[node_id] = pos
                    transforms[node_id] = _quat_pos_to_4x4(pos, quat_scipy)

                yield {
                    "transforms": transforms,  # (25, 4, 4)
                    "joints": joints[np.newaxis],  # (1, 25, 3)
                    "links": [],
                }

        except KeyboardInterrupt:
            self.logger.info("ManusHandTracker interrupted.")
        finally:
            self.close()

    def close(self):
        """Clean up ZMQ resources."""
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None
        self.logger.info("ManusHandTracker closed.")
