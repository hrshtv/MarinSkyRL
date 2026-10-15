"""Post-update numerics check for learner optimizations.

A learner optimization (kernel, layout, overlap, precision plumbing) must not change what one PPO update
computes beyond the bf16 reduction-order noise that any layout change introduces. This module records a
fixed set of signals for a fixed batch, checkpoint, and seed, and compares a candidate's signals against a
stored, calibrated reference:

* ``train_status/<key>``: scalars that ``ppo_train`` returns (loss, gradient norm, clip fractions, log-ratio
  statistics, update counts), one set per update.
* ``logprobs/{pre,post}_update/...``: scoring log-probs on the batch before the first update and after the
  last one, summarized as the global masked mean, the per-sequence masked means, and a deterministic sample
  of token log-probs.
* ``params/<group>/...``: a parameter-delta fingerprint over all updates, per layout-independent parameter
  group (see :func:`canonical_parameter_group`).

The parameter fingerprint is computed on rank-local shards and reduced by summation, so no full weight moves.
Every per-group statistic is a sum over elements of a function of that element's value before and after the
update, so it does not depend on how a layout shards, permutes, or interleaves elements (TP slices, fused GLU
or QKV layouts, local expert numbering). The random-projection sketches draw each element's sign from a hash
of the element's *pre-update value bits* rather than its position, which keeps them layout-independent while
still detecting an update applied to the wrong element: the sign no longer pairs with the same delta.

Calibration procedure:

1. Record the reference at least twice with :func:`record_numerics`, preferably under different layouts
   (TP/EP/PP/CP or micro-batch size), because a layout change reorders bf16 reductions and that noise is
   what a candidate legitimately adds. Runs on one layout can be bitwise identical and underestimate noise.
2. :func:`calibrate_reference` sets each signal's tolerance to ``multiplier`` times the largest pairwise
   deviation observed, but never below the floor for the signal's kind. Store it with
   :func:`write_reference`.
3. Run the must-fail controls (:class:`ControlPerturbation`) through the same driver with the reference
   provenance and check that :func:`compare_to_reference` rejects them. ``NEGATE_ADVANTAGES`` must fail the
   loss, sketch, and post-update log-prob signals. ``DOUBLE_ADVANTAGES`` must fail ``policy_loss`` and
   ``raw_grad_norm``; scale-invariant optimizers (Adam, MuonH) can leave the parameter delta nearly unchanged,
   which is why the train-status scalars are part of the check. A control that passes means the
   tolerances are too loose for that signal.
4. Compare each candidate with :func:`compare_to_reference`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
import hashlib
import itertools
import json
import math
import re

from omegaconf import DictConfig, OmegaConf
import ray
import torch

from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.io import io
from skyrl_train.training_batch import TrainingInputBatch

SCHEMA_VERSION = 1

# Config subtrees that change what an update computes and so must match between reference and candidate.
# Layout and kernel choices are deliberately absent: they are what candidates vary.
NUMERICS_CONFIG_KEYS = (
    "trainer.seed",
    "trainer.algorithm",
    "trainer.policy.optimizer_config",
    "trainer.policy_mini_batch_size",
    "trainer.update_epochs_per_batch",
    "generator.sampling_params.temperature",
)

# ppo_train scalars compared per update. Losses and fractions can be zero, so they use absolute deviations.
TRAIN_STATUS_SIGNAL_KINDS = {
    "policy_loss": "absolute",
    "final_loss": "absolute",
    "policy_kl": "absolute",
    "policy_entropy": "relative",
    "raw_grad_norm": "relative",
    "ppo_clip_ratio": "absolute",
    "ppo_clip_ratio_low": "absolute",
    "ppo_clip_ratio_high": "absolute",
    "log_ratio_mean": "absolute",
    "log_ratio_abs_mean": "absolute",
    "log_ratio_abs_max": "absolute",
    "skipped_steps": "absolute",
    "policy_update_steps": "absolute",
}
REQUIRED_TRAIN_STATUS_KEYS = ("policy_loss", "raw_grad_norm", "policy_update_steps")

LOGPROB_SAMPLE_SIZE = 4096
# Odd 32-bit salts select independent hash streams for the sign sketches and the log-prob sample.
SKETCH_SALTS = (0x9E3779B9, 0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F)
LOGPROB_SAMPLE_SALT = 0x165667B1
_MASK32 = 0xFFFFFFFF
_HASH_MULTIPLIER = 0x45D9F3B
# Bounds the float64 temporaries of one statistics pass.
STATISTICS_CHUNK_ELEMENTS = 1 << 24

_MODULE_PREFIX = re.compile(r"^(?:module\.)+")
_LAYER_MODULE = re.compile(r"^(?:module\.)*decoder\.layers\.(\d+)$")
_LOCAL_LAYER = re.compile(r"^decoder\.layers\.(\d+)\.")
_LOCAL_EXPERT = re.compile(r"\.local_experts\.\d+\.")
_GROUPED_EXPERT_WEIGHT = re.compile(r"(\.mlp\.experts\..*\.(?:weight|bias))\d+$")


class SignalKind(StrEnum):
    """How a signal's deviation from the reference is measured.

    RELATIVE: ``|c - r| / |r|`` for one scalar. ABSOLUTE: ``max_i |c_i - r_i|``. RELATIVE_L2:
    ``||c - r|| / ||r||`` for a vector.
    """

    RELATIVE = "relative"
    ABSOLUTE = "absolute"
    RELATIVE_L2 = "relative_l2"


class ControlPerturbation(StrEnum):
    """Batch perturbations that a calibrated reference must reject."""

    NEGATE_ADVANTAGES = "negate_advantages"
    DOUBLE_ADVANTAGES = "double_advantages"


@dataclass(frozen=True)
class Signal:
    kind: SignalKind
    values: tuple[float, ...]


@dataclass(frozen=True)
class NumericsProvenance:
    """Facts that must match between a reference and a candidate, plus informational layout facts.

    Attributes:
        checkpoint: Model checkpoint identity, including the layer count of a truncated variant.
        batch_sha256: Hash of the batch tensors and the routes they carry.
        config_fingerprint: :func:`numerics_config_fingerprint` of the run's config.
        num_updates: ``ppo_train`` calls between the pre- and post-update scoring passes.
        layout: Informational parallel layout description, e.g. ``"tp1-ep8-pp1-cp1-mbs1"``.
        source_revision: Informational harness revision.
    """

    checkpoint: str
    batch_sha256: str
    config_fingerprint: str
    num_updates: int
    layout: str
    source_revision: str

    def matching_fields(self) -> tuple[str, str, str, int]:
        return (self.checkpoint, self.batch_sha256, self.config_fingerprint, self.num_updates)


@dataclass(frozen=True)
class NumericsRecord:
    provenance: NumericsProvenance
    signals: dict[str, Signal]


@dataclass(frozen=True)
class CalibrationPolicy:
    """Tolerance = max(multiplier x observed spread, floors[kind])."""

    multiplier: float
    floors: dict[SignalKind, float]


# A starting point for Hero-scale bf16 runs; check it with the must-fail controls before trusting it.
RECOMMENDED_CALIBRATION = CalibrationPolicy(
    multiplier=4.0,
    floors={SignalKind.RELATIVE: 1e-3, SignalKind.ABSOLUTE: 1e-4, SignalKind.RELATIVE_L2: 1e-3},
)


@dataclass(frozen=True)
class NumericsReference:
    """A reference record with calibrated tolerances.

    Attributes:
        record: The first calibration run; candidates are compared against its signal values.
        tolerances: Maximum accepted deviation per signal.
        spreads: Largest pairwise deviation per signal across the calibration runs.
        calibration: The policy that turned spreads into tolerances.
        calibration_layouts: Layouts of the calibration runs, in order.
    """

    schema_version: int
    record: NumericsRecord
    tolerances: dict[str, float]
    spreads: dict[str, float]
    calibration: CalibrationPolicy
    calibration_layouts: tuple[str, ...]


@dataclass(frozen=True)
class SignalCheck:
    name: str
    deviation: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.deviation <= self.tolerance


@dataclass(frozen=True)
class NumericsComparison:
    checks: tuple[SignalCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[SignalCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


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


# ---------------------------------------------------------------------------------------------------------------
# Hashing and the parameter-delta fingerprint


def _mix32(values: torch.Tensor) -> torch.Tensor:
    """Bijective 32-bit integer mix of int64 values in ``[0, 2**32)``; products stay below ``2**59``."""
    values = (((values >> 16) ^ values) * _HASH_MULTIPLIER) & _MASK32
    values = (((values >> 16) ^ values) * _HASH_MULTIPLIER) & _MASK32
    return (values >> 16) ^ values


def _value_bits(values: torch.Tensor) -> torch.Tensor:
    return values.to(torch.float32).view(torch.int32).to(torch.int64) & _MASK32


def _value_signs(values: torch.Tensor, salt: int) -> torch.Tensor:
    """Pseudo-random +-1 per element, a function of the element's value bits only."""
    return 1.0 - 2.0 * (_mix32(_value_bits(values) ^ salt) & 1).to(torch.float64)


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


def parameter_signals(statistics: Mapping[str, ParameterGroupStatistics]) -> dict[str, Signal]:
    signals = {}
    for group, group_statistics in sorted(statistics.items()):
        signals[f"params/{group}/numel"] = Signal(SignalKind.ABSOLUTE, (group_statistics.numel,))
        signals[f"params/{group}/delta_norm"] = Signal(
            SignalKind.RELATIVE, (math.sqrt(group_statistics.delta_square_sum),)
        )
        signals[f"params/{group}/sketch"] = Signal(SignalKind.RELATIVE_L2, group_statistics.sketches)
    total = sum(group_statistics.delta_square_sum for group_statistics in statistics.values())
    signals["params/total_delta_norm"] = Signal(SignalKind.RELATIVE, (math.sqrt(total),))
    return signals


# ---------------------------------------------------------------------------------------------------------------
# Log-prob and train-status signals


def logprob_sample_positions(mask: torch.Tensor, sample_size: int) -> torch.Tensor:
    """Flat positions of up to ``sample_size`` masked tokens, chosen by a position hash and sorted."""
    positions = torch.nonzero(mask.reshape(-1), as_tuple=True)[0]
    if positions.numel() <= sample_size:
        return positions
    keys = _mix32(positions.to(torch.int64) ^ LOGPROB_SAMPLE_SALT)
    chosen = positions[torch.topk(keys, sample_size, largest=False).indices]
    return torch.sort(chosen).values


def logprob_signals(prefix: str, logprobs: torch.Tensor, mask: torch.Tensor) -> dict[str, Signal]:
    """Summarize response log-probs ``[batch, response_length]`` over the loss mask."""
    if logprobs.shape != mask.shape:
        raise ValueError(f"Log-prob shape {tuple(logprobs.shape)} does not match mask {tuple(mask.shape)}")
    values = logprobs.to(torch.float64)
    weights = mask.to(torch.float64)
    token_count = weights.sum()
    if token_count == 0:
        raise ValueError("Log-prob mask selects no tokens")
    per_sequence_count = weights.sum(dim=1).clamp_min(1.0)
    positions = logprob_sample_positions(mask.bool(), LOGPROB_SAMPLE_SIZE)
    return {
        f"{prefix}/token_mean": Signal(SignalKind.ABSOLUTE, (float((values * weights).sum() / token_count),)),
        f"{prefix}/sequence_mean": Signal(
            SignalKind.ABSOLUTE, tuple(((values * weights).sum(dim=1) / per_sequence_count).tolist())
        ),
        f"{prefix}/sampled_tokens": Signal(SignalKind.ABSOLUTE, tuple(values.reshape(-1)[positions].tolist())),
    }


def train_status_signals(update_index: int, train_status: Mapping[str, float]) -> dict[str, Signal]:
    missing = [key for key in REQUIRED_TRAIN_STATUS_KEYS if key not in train_status]
    if missing:
        raise KeyError(f"train_status lacks required keys {missing}")
    return {
        f"train_status/update{update_index}/{key}": Signal(SignalKind(kind), (float(train_status[key]),))
        for key, kind in TRAIN_STATUS_SIGNAL_KINDS.items()
        if key in train_status
    }


def numerics_record(
    provenance: NumericsProvenance,
    train_statuses: Sequence[Mapping[str, float]],
    pre_update_logprobs: torch.Tensor,
    post_update_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    parameter_statistics: Mapping[str, ParameterGroupStatistics],
) -> NumericsRecord:
    if len(train_statuses) != provenance.num_updates:
        raise ValueError(f"Expected {provenance.num_updates} train statuses, got {len(train_statuses)}")
    signals: dict[str, Signal] = {}
    for index, status in enumerate(train_statuses):
        signals.update(train_status_signals(index, status))
    signals.update(logprob_signals("logprobs/pre_update", pre_update_logprobs, loss_mask))
    signals.update(logprob_signals("logprobs/post_update", post_update_logprobs, loss_mask))
    signals.update(parameter_signals(parameter_statistics))
    return NumericsRecord(provenance=provenance, signals=signals)


def numerics_config_fingerprint(cfg: DictConfig) -> str:
    """Hash the config subtrees in :data:`NUMERICS_CONFIG_KEYS`; a missing key is an error."""
    selected = {}
    for key in NUMERICS_CONFIG_KEYS:
        value = OmegaConf.select(cfg, key, throw_on_missing=True)
        if value is None:
            raise KeyError(f"Config lacks numerics key {key}")
        selected[key] = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    payload = json.dumps(selected, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def perturbed_batch(batch: TrainingInputBatch, control: ControlPerturbation) -> TrainingInputBatch:
    """Return a copy of ``batch`` with the control's advantage perturbation applied."""
    scale = {ControlPerturbation.NEGATE_ADVANTAGES: -1.0, ControlPerturbation.DOUBLE_ADVANTAGES: 2.0}[control]
    tensors = dict(batch)
    tensors["advantages"] = batch["advantages"] * scale
    perturbed = TrainingInputBatch(tensors, routed_expert_rows=batch.routed_expert_rows)
    perturbed.metadata = dict(batch.metadata)
    return perturbed


# ---------------------------------------------------------------------------------------------------------------
# Comparison and calibration


def signal_deviation(reference: Signal, candidate: Signal) -> float:
    """Deviation of ``candidate`` from ``reference``; ``inf`` for a structural mismatch or non-finite value."""
    if reference.kind != candidate.kind or len(reference.values) != len(candidate.values):
        return math.inf
    if not all(math.isfinite(value) for value in candidate.values):
        return math.inf
    if reference.kind == SignalKind.ABSOLUTE:
        return max((abs(c - r) for r, c in zip(reference.values, candidate.values)), default=0.0)
    if reference.kind == SignalKind.RELATIVE:
        (r,), (c,) = reference.values, candidate.values
        return _relative(abs(c - r), abs(r))
    difference = math.sqrt(sum((c - r) ** 2 for r, c in zip(reference.values, candidate.values)))
    return _relative(difference, math.sqrt(sum(r * r for r in reference.values)))


def _relative(difference: float, scale: float) -> float:
    if scale == 0.0:
        return 0.0 if difference == 0.0 else math.inf
    return difference / scale


def _check_provenance(reference: NumericsProvenance, candidate: NumericsProvenance) -> None:
    if reference.matching_fields() != candidate.matching_fields():
        raise ValueError(
            "Numerics provenance differs (checkpoint, batch_sha256, config_fingerprint, num_updates): "
            f"{reference.matching_fields()} != {candidate.matching_fields()}"
        )


def calibrate_reference(runs: Sequence[NumericsRecord], policy: CalibrationPolicy) -> NumericsReference:
    """Build a reference from repeated runs of one configuration, ideally on different layouts."""
    if len(runs) < 2:
        raise ValueError("Calibration needs at least two runs")
    first = runs[0]
    for run in runs[1:]:
        _check_provenance(first.provenance, run.provenance)
        if run.signals.keys() != first.signals.keys():
            raise ValueError(
                f"Calibration runs disagree on signals: {sorted(run.signals.keys() ^ first.signals.keys())}"
            )
    spreads = {
        name: max(signal_deviation(a.signals[name], b.signals[name]) for a, b in itertools.permutations(runs, 2))
        for name in first.signals
    }
    unstable = sorted(name for name, spread in spreads.items() if not math.isfinite(spread))
    if unstable:
        raise ValueError(f"Calibration runs disagree structurally or are non-finite for {unstable}")
    tolerances = {
        name: max(policy.multiplier * spread, policy.floors[first.signals[name].kind])
        for name, spread in spreads.items()
    }
    return NumericsReference(
        schema_version=SCHEMA_VERSION,
        record=first,
        tolerances=tolerances,
        spreads=spreads,
        calibration=policy,
        calibration_layouts=tuple(run.provenance.layout for run in runs),
    )


def compare_to_reference(reference: NumericsReference, candidate: NumericsRecord) -> NumericsComparison:
    """Check every reference signal; a signal missing from or added by the candidate fails."""
    _check_provenance(reference.record.provenance, candidate.provenance)
    checks = []
    for name, expected in sorted(reference.record.signals.items()):
        observed = candidate.signals.get(name)
        deviation = math.inf if observed is None else signal_deviation(expected, observed)
        checks.append(SignalCheck(name=name, deviation=deviation, tolerance=reference.tolerances[name]))
    checks.extend(
        SignalCheck(name=name, deviation=math.inf, tolerance=0.0)
        for name in sorted(candidate.signals.keys() - reference.record.signals.keys())
    )
    return NumericsComparison(checks=tuple(checks))


# ---------------------------------------------------------------------------------------------------------------
# JSON storage


def _record_payload(record: NumericsRecord) -> dict:
    return {
        "provenance": asdict(record.provenance),
        "signals": {
            name: {"kind": str(signal.kind), "values": list(signal.values)} for name, signal in record.signals.items()
        },
    }


def _record_from_payload(payload: Mapping) -> NumericsRecord:
    return NumericsRecord(
        provenance=NumericsProvenance(**payload["provenance"]),
        signals={
            name: Signal(SignalKind(signal["kind"]), tuple(float(value) for value in signal["values"]))
            for name, signal in payload["signals"].items()
        },
    )


def reference_json(reference: NumericsReference) -> str:
    payload = {
        "schema_version": reference.schema_version,
        "record": _record_payload(reference.record),
        "tolerances": reference.tolerances,
        "spreads": reference.spreads,
        "calibration": {
            "multiplier": reference.calibration.multiplier,
            "floors": {str(kind): floor for kind, floor in reference.calibration.floors.items()},
        },
        "calibration_layouts": list(reference.calibration_layouts),
    }
    return json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"


def parse_reference(text: str) -> NumericsReference:
    payload = json.loads(text)
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported numerics reference schema {payload['schema_version']}")
    calibration = payload["calibration"]
    return NumericsReference(
        schema_version=payload["schema_version"],
        record=_record_from_payload(payload["record"]),
        tolerances={name: float(value) for name, value in payload["tolerances"].items()},
        spreads={name: float(value) for name, value in payload["spreads"].items()},
        calibration=CalibrationPolicy(
            multiplier=float(calibration["multiplier"]),
            floors={SignalKind(kind): float(floor) for kind, floor in calibration["floors"].items()},
        ),
        calibration_layouts=tuple(payload["calibration_layouts"]),
    )


def write_reference(path: str, reference: NumericsReference) -> None:
    io.write_bytes_atomic(path, reference_json(reference).encode())


def read_reference(path: str) -> NumericsReference:
    return parse_reference(io.read_bytes(path).decode())


def record_json(record: NumericsRecord) -> str:
    """Serialize one run, e.g. a calibration run kept for later re-calibration."""
    return json.dumps({"schema_version": SCHEMA_VERSION, **_record_payload(record)}, indent=1, allow_nan=False) + "\n"


def parse_record(text: str) -> NumericsRecord:
    payload = json.loads(text)
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported numerics record schema {payload['schema_version']}")
    return _record_from_payload(payload)


# ---------------------------------------------------------------------------------------------------------------
# Driver (GPU; runs through the production policy worker)


def _response_logprobs(policy, batch: TrainingInputBatch) -> torch.Tensor:
    outputs = ray.get(policy.async_run_ray_method("mesh", "forward", data=batch))
    return concatenate_outputs_after_mesh_dispatch(policy.actor_infos, outputs)["output"].float()


def record_numerics(policy, batch: TrainingInputBatch, provenance: NumericsProvenance) -> NumericsRecord:
    """Score, update ``provenance.num_updates`` times, fingerprint the parameter delta, and score again.

    ``policy`` is the Megatron policy actor group. The batch is used as given; set its ``action_log_probs``
    to the pre-update scores beforehand if the update should be on-policy.
    """
    pre_update = _response_logprobs(policy, batch)
    ray.get(policy.async_run_ray_method("pass_through", "capture_numerics_baseline"))
    statuses = []
    for _ in range(provenance.num_updates):
        outputs = ray.get(policy.async_run_ray_method("mesh", "ppo_train", batch))
        statuses.append(outputs[0].metadata["train_status"])
    per_rank = ray.get(policy.async_run_ray_method("pass_through", "numerics_parameter_statistics"))
    post_update = _response_logprobs(policy, batch)
    return numerics_record(
        provenance,
        statuses,
        pre_update,
        post_update,
        batch["loss_mask"],
        merge_parameter_statistics(per_rank),
    )
