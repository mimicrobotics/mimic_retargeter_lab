from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from dexworld.types import Chirality, HandLandmark


class BaseHandModel(ABC):
    """Minimal interface shared by both robot and human hand models."""

    def __init__(self, chirality: Chirality):
        self.chirality = chirality

    @abstractmethod
    def get_num_fingertips(self) -> int:
        pass

    @abstractmethod
    def get_landmarks(self, data: Any, **kwargs) -> dict[HandLandmark, np.ndarray]:
        pass

    def compute_keyvectors(self, data: Any, **kwargs) -> dict[str, np.ndarray]:
        """Universally compute pairwise keyvectors from the extracted landmarks.

        Emits two groups:
          - All pairs among {WRIST, *_TIP} (e.g. ``WRIST_to_THUMB_TIP``,
            ``THUMB_TIP_to_INDEX_TIP``) — the full clique used by
            keyvector / DexPilot / AKO wrist+pinch+finger losses.
          - Per-finger ``<FINGER>_DP_to_<FINGER>_TIP`` — a short vector
            along the distal phalanx used for fingertip-orientation losses
            (e.g. AKO's third objective). Only emitted for fingers where
            both the DP and TIP landmarks are available, and only as the
            intra-finger pair (cross-finger DP pairs aren't useful and
            would blow up the keyvector count).
        """
        landmarks = self.get_landmarks(data, **kwargs)

        # Gather all available tracking points
        available_lms = []
        if HandLandmark.WRIST in landmarks:
            available_lms.append(HandLandmark.WRIST)

        canonical_tips = [
            HandLandmark.THUMB_TIP,
            HandLandmark.INDEX_TIP,
            HandLandmark.MIDDLE_TIP,
            HandLandmark.RING_TIP,
            HandLandmark.PINKY_TIP,
        ]
        for tip in canonical_tips:
            if tip in landmarks:
                available_lms.append(tip)

        # Calculate pairwise vectors and assign to Universal string keys
        out: dict[str, np.ndarray] = {}
        num = len(available_lms)
        for i in range(num):
            for j in range(i + 1, num):
                lm_i = available_lms[i]
                lm_j = available_lms[j]

                vec = landmarks[lm_j] - landmarks[lm_i]

                # e.g., "WRIST_to_THUMB_TIP"
                universal_key = f"{lm_i.name}_to_{lm_j.name}"
                out[universal_key] = vec

        # Per-finger DP->TIP orientation vectors.
        finger_dp_tip_pairs = [
            (HandLandmark.THUMB_DP, HandLandmark.THUMB_TIP),
            (HandLandmark.INDEX_DP, HandLandmark.INDEX_TIP),
            (HandLandmark.MIDDLE_DP, HandLandmark.MIDDLE_TIP),
            (HandLandmark.RING_DP, HandLandmark.RING_TIP),
            (HandLandmark.PINKY_DP, HandLandmark.PINKY_TIP),
        ]
        for dp, tip in finger_dp_tip_pairs:
            if dp in landmarks and tip in landmarks:
                out[f"{dp.name}_to_{tip.name}"] = landmarks[tip] - landmarks[dp]

        return out
