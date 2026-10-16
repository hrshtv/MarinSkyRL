from collections import defaultdict
from typing import Dict, List
import re
import numpy as np

MAX_DOMAIN_REWARD_METRICS = 32


def domain_metric_source_key(source: str | None) -> str:
    """Encode one source as a distinct, tracker-safe metric path segment.

    Lowercase ASCII names stay readable. ``_missing`` denotes absent metadata;
    other names use fixed-width UTF-8 byte escapes under ``_source_``.
    """
    if source is None:
        return "_missing"
    if re.fullmatch(r"[a-z_][a-z0-9_]*", source) and source != "_missing" and not source.startswith("_source_"):
        return source
    encoded = "".join(
        chr(byte) if byte in b"abcdefghijklmnopqrstuvwxyz0123456789" else f"_{byte:02x}"
        for byte in source.encode("utf-8")
    )
    return f"_source_{encoded}"


def domain_reward_metrics(data_sources: List[str | None], rewards: List[float]) -> Dict[str, float]:
    """Return bounded per-source means with distinct, stable metric names."""
    if len(data_sources) != len(rewards):
        raise ValueError(
            f"Expected one data source per reward, got {len(data_sources)} sources and {len(rewards)} rewards"
        )

    rewards_by_source: Dict[str, List[float]] = defaultdict(list)
    for source, reward in zip(data_sources, rewards, strict=True):
        rewards_by_source[domain_metric_source_key(source)].append(reward)

    sources = sorted(rewards_by_source)
    metrics = {
        f"reward/domain/{source}/avg_raw_reward": float(np.mean(rewards_by_source[source]))
        for source in sources[:MAX_DOMAIN_REWARD_METRICS]
    }
    if len(sources) > MAX_DOMAIN_REWARD_METRICS:
        overflow = [reward for source in sources[MAX_DOMAIN_REWARD_METRICS:] for reward in rewards_by_source[source]]
        metrics["reward/domain_overflow/avg_raw_reward"] = float(np.mean(overflow))
    return metrics
