# Per-call retargeter latency over real episode data.
#
# The retargeter is exercised exactly once per (data_path, episode_id)
# during the cache populate in RetargetCache: phase 1 runs untimed
# warmup retargets to compile JAX, phase 2 times every frame in the
# stream with `jax.block_until_ready` after each call. This metric is a
# thin reader of those timings — it doesn't iterate the dataset itself.
# Latencies are therefore order-independent: equivalent whether Latency
# triggers the populate or rides on a populate kicked off earlier in the
# pipeline.

import hydra
from omegaconf import DictConfig

from dexworld.hand_models import HumanHandModel, RobotHandModel
from dexworld.retargeting.online import BaseOnlineRetargeter
from dexworld.utils import RetargetCache, get_logger

from ._stats import summarize_array
from .base_metric import BaseMetric


_NOISY_P99_THRESHOLD = 1000


class LatencyMetric(BaseMetric):
    def __init__(
        self,
        config,
        human_hand_model: HumanHandModel,
        robot_hand_model: RobotHandModel,
        retargeter: BaseOnlineRetargeter,
        data_source_cfg: DictConfig,
        retarget_cache: RetargetCache | None = None,
    ):
        self.display_name = config.display_name

        self.retargeter = retargeter
        self.retarget_cache = retarget_cache
        self.human_hand_model = human_hand_model
        self.robot_hand_model = robot_hand_model

        self._logger = get_logger(__name__)

        self.data_source = hydra.utils.instantiate(data_source_cfg)

    def compute(self):
        if self.retarget_cache is None:
            raise RuntimeError(
                "LatencyMetric now reads timings from RetargetCache; the cache "
                "is required."
            )

        episode_metrics = {}
        for episode_data in self.data_source.get_episode_iter():
            cache_key = (
                str(self.data_source.data_path),
                str(episode_data["episode_id"]),
            )
            # Ensure populated. No-op if any earlier metric already ran on
            # this (data_path, episode_id); otherwise this triggers the
            # timed populate (warmup → reset → full pass).
            self.retarget_cache.get(cache_key, episode_data["joints"])
            timings = self.retarget_cache.timings(cache_key)

            # All T frames are timed and reported — no slice.
            latencies_ms = list(timings["latencies_ms"])
            stats = _summarize(latencies_ms)
            stats["device"] = timings["device"]
            stats["device_str"] = timings["device"]
            stats["num_warmup"] = timings["warmup_frames"]
            stats["num_timed"] = len(latencies_ms)

            if 0 < len(latencies_ms) < _NOISY_P99_THRESHOLD:
                self._logger.warning(
                    f"Latency: episode {episode_data['episode_id']} has "
                    f"{len(latencies_ms)} timed frames (<{_NOISY_P99_THRESHOLD}); "
                    f"p99 estimate is noisy."
                )

            episode_metrics[episode_data["episode_id"]] = stats

        return episode_metrics


def _summarize(latencies_ms: list[float]) -> dict:
    """Emit the canonical 12-stat block plus the legacy ``*_ms`` aliases.

    The bare ``STAT_KEYS`` names (``n``/``mean``/``median``/``std``/``min``/
    ``max``/``p1``/``p5``/``p25``/``p75``/``p95``/``p99``) match every other
    metric's stat block so consumers can iterate uniformly. The ``*_ms``
    aliases stay so the dashboard's existing reads keep working.
    """
    stats = summarize_array(latencies_ms)
    return {
        "latencies_ms": list(latencies_ms),
        # Canonical 12-stat block (same names as every other metric).
        **stats,
        # Legacy `*_ms` aliases — same values, retained for the dashboard.
        "mean_ms": stats["mean"],
        "median_ms": stats["median"],
        "stdev_ms": stats["std"],
        "min_ms": stats["min"],
        "max_ms": stats["max"],
        "p1_ms": stats["p1"],
        "p5_ms": stats["p5"],
        "p25_ms": stats["p25"],
        "p75_ms": stats["p75"],
        "p95_ms": stats["p95"],
        "p99_ms": stats["p99"],
    }
