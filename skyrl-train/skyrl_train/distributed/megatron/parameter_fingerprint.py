"""Layout-independent fingerprint of a parameter update.

Every per-group statistic is a sum over elements of a function of that element's value before and after the
update, so shards and ranks combine by addition and the result does not depend on how a layout shards, permutes,
or interleaves elements (TP slices, fused GLU or QKV layouts, local expert numbering). The random-projection
sketches draw each element's sign from a hash of the element's *pre-update value bits* rather than its position,
which keeps them layout-independent while still detecting an update applied to the wrong element: the sign no
longer pairs with the same delta.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import re

import torch

# Odd 32-bit salts select independent hash streams for the sign sketches.
SKETCH_SALTS = (0x9E3779B9, 0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F)
_MASK32 = 0xFFFFFFFF
_HASH_MULTIPLIER = 0x45D9F3B
# Bounds the float64 temporaries of one statistics pass.
STATISTICS_CHUNK_ELEMENTS = 1 << 24

_MODULE_PREFIX = re.compile(r"^(?:module\.)+")
_LAYER_MODULE = re.compile(r"^(?:module\.)*decoder\.layers\.(\d+)$")
_LOCAL_LAYER = re.compile(r"^decoder\.layers\.(\d+)\.")
_LOCAL_EXPERT = re.compile(r"\.local_experts\.\d+\.")
_GROUPED_EXPERT_WEIGHT = re.compile(r"(\.mlp\.experts\..*\.(?:weight|bias))\d+$")


@dataclass(frozen=True)
class ParameterGroupStatistics:
    """Additive per-group sums over parameter elements; shards and ranks combine by addition."""

    numel: float
    delta_square_sum: float
    before_square_sum: float
    sketches: tuple[float, ...]

    def __add__(self, other: ParameterGroupStatistics) -> ParameterGroupStatistics:
        return ParameterGroupStatistics(
            numel=self.numel + other.numel,
            delta_square_sum=self.delta_square_sum + other.delta_square_sum,
            before_square_sum=self.before_square_sum + other.before_square_sum,
            sketches=tuple(a + b for a, b in zip(self.sketches, other.sketches, strict=True)),
        )


def mix32(values: torch.Tensor) -> torch.Tensor:
    """Bijective 32-bit integer mix of int64 values in ``[0, 2**32)``; products stay below ``2**59``."""
    values = (((values >> 16) ^ values) * _HASH_MULTIPLIER) & _MASK32
    values = (((values >> 16) ^ values) * _HASH_MULTIPLIER) & _MASK32
    return (values >> 16) ^ values


def _value_bits(values: torch.Tensor) -> torch.Tensor:
    return values.to(torch.float32).view(torch.int32).to(torch.int64) & _MASK32


def _value_signs(values: torch.Tensor, salt: int) -> torch.Tensor:
    """Pseudo-random +-1 per element, a function of the element's value bits only."""
    return 1.0 - 2.0 * (mix32(_value_bits(values) ^ salt) & 1).to(torch.float64)


@torch.no_grad()
def parameter_delta_statistics(before: torch.Tensor, after: torch.Tensor) -> ParameterGroupStatistics:
    """Sum the fingerprint statistics over one parameter shard.

    ``before`` may live on another device than ``after``; it is moved one chunk at a time.
    """
    if before.shape != after.shape:
        raise ValueError(f"Parameter shape changed: {tuple(before.shape)} -> {tuple(after.shape)}")
    flat_before = before.detach().reshape(-1)
    flat_after = after.detach().reshape(-1)
    totals = torch.zeros(2 + len(SKETCH_SALTS), dtype=torch.float64, device=flat_after.device)
    for start in range(0, flat_after.numel(), STATISTICS_CHUNK_ELEMENTS):
        stop = start + STATISTICS_CHUNK_ELEMENTS
        old = flat_before[start:stop].to(flat_after.device)
        old64 = old.to(torch.float64)
        delta = flat_after[start:stop].to(torch.float64) - old64
        parts = [delta.square().sum(), old64.square().sum()]
        parts.extend((_value_signs(old, salt) * delta).sum() for salt in SKETCH_SALTS)
        totals += torch.stack(parts)
    values = totals.tolist()
    return ParameterGroupStatistics(
        numel=float(flat_after.numel()),
        delta_square_sum=values[0],
        before_square_sum=values[1],
        sketches=tuple(values[2:]),
    )


def canonical_parameter_group(parameter_name: str, global_layer_by_local: Mapping[int, int]) -> str:
    """Map a rank-local Megatron parameter name to a name that every layout shares.

    Wrapper prefixes are dropped, the pipeline-local decoder layer index becomes the global one, and the
    rank-local expert index of sequential (``local_experts.<i>``) and grouped (``weight<i>``) experts is
    removed so a layer's routed experts form one group across expert-parallel ranks.
    """
    name = _MODULE_PREFIX.sub("", parameter_name)
    layer = _LOCAL_LAYER.match(name)
    if layer is not None:
        global_layer = global_layer_by_local[int(layer.group(1))]
        name = f"decoder.layers.{global_layer}.{name[layer.end() :]}"
    name = _LOCAL_EXPERT.sub(".local_experts.", name)
    return _GROUPED_EXPERT_WEIGHT.sub(r"\1", name)


def global_layer_indices(named_modules: Iterable[tuple[str, object]]) -> dict[int, int]:
    """Map pipeline-local decoder layer indices to 0-based global ones via Megatron's ``layer_number``."""
    indices = {}
    for name, module in named_modules:
        match = _LAYER_MODULE.match(name)
        if match is not None:
            indices[int(match.group(1))] = int(module.layer_number) - 1
    return indices


def merge_parameter_statistics(
    per_rank: Sequence[Mapping[str, Sequence[float]]],
) -> dict[str, ParameterGroupStatistics]:
    """Sum the per-rank wire format ``{group: [numel, delta_sq, before_sq, *sketches]}`` over ranks."""
    merged: dict[str, ParameterGroupStatistics] = {}
    for rank_statistics in per_rank:
        for group, values in rank_statistics.items():
            statistics = statistics_from_wire(values)
            merged[group] = merged[group] + statistics if group in merged else statistics
    return merged


def statistics_to_wire(statistics: ParameterGroupStatistics) -> list[float]:
    return [statistics.numel, statistics.delta_square_sum, statistics.before_square_sum, *statistics.sketches]


def statistics_from_wire(values: Sequence[float]) -> ParameterGroupStatistics:
    if len(values) != 3 + len(SKETCH_SALTS):
        raise ValueError(f"Expected {3 + len(SKETCH_SALTS)} statistics, got {len(values)}")
    return ParameterGroupStatistics(
        numel=float(values[0]),
        delta_square_sum=float(values[1]),
        before_square_sum=float(values[2]),
        sketches=tuple(float(value) for value in values[3:]),
    )
