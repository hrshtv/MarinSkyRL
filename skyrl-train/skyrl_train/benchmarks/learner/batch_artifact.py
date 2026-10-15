"""Versioned on-disk ensemble of learner benchmark batches.

An ensemble batch is one post-advantage ``TrainingInputBatch`` taken from an RL run's
``trainer.dump_data_batch`` output, stored without pickle so it can be replayed through the production policy
worker. Layout under an ensemble root (a local path or a cloud URI understood by ``skyrl_train.io``)::

    <root>/<ensemble_version>/<profile>/batch_<i>/manifest.json
    <root>/<ensemble_version>/<profile>/batch_<i>/tensors.safetensors
    <root>/<ensemble_version>/<profile>/batch_<i>/routes/<checkpoint_id>.safetensors

Routes belong to the model that is benchmarked, not to the model that generated the text, so the converter drops
the dump's routes and a separate routing pass writes one routes file per checkpoint. Route rows are
response-aligned: row ``t`` of a sequence holds the experts chosen at the position that predicts response token
``t``, beginning at the final prompt token (see ``dense_replay_targets``).

The stored batch has the generating run's data-parallel padding rows removed; a driver re-pads it to its own
data-parallel size and sets ``metadata["global_step"]`` before calling ``ppo_train``.
"""

import argparse
import dataclasses
import hashlib
import json
import pickle
import re
import struct
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import safetensors.numpy
import safetensors.torch
import torch
from loguru import logger

from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.io import io
from skyrl_train.training_batch import ROUTED_EXPERTS_KEY, TrainingInputBatch

SCHEMA_VERSION = 1
MANIFEST_FILE = "manifest.json"
TENSORS_FILE = "tensors.safetensors"
ROUTES_DIR = "routes"
BATCH_DIR_PATTERN = re.compile(r"^batch_(\d+)$")
DUMP_FILE_PATTERN = re.compile(r"global_step_(\d+)_training_input\.pkl$")
CHECKPOINT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
NONE_KEYS_METADATA = "none_keys"
ROUTE_ROWS_KEY = "rows"
ROUTE_OFFSETS_KEY = "offsets"

# ``TrainingBatchIterator`` indexes these keys directly, so they must be present (possibly as ``None``).
REQUIRED_KEYS = (
    "sequences",
    "attention_mask",
    "response_mask",
    "loss_mask",
    "advantages",
    "returns",
    "action_log_probs",
    "base_action_log_probs",
    "values",
)


class LengthProfile(StrEnum):
    SHORT = "short"
    LONG_REASONING = "long_reasoning"
    AGENTIC = "agentic"


@dataclass(frozen=True)
class EnsembleBatchManifest:
    """Provenance and length statistics for one stored batch.

    Attributes:
        schema_version: On-disk format version; readers reject other versions.
        ensemble_version: Ensemble release, e.g. ``"v0"``.
        profile: Length profile the batch belongs to.
        batch_index: Index of the batch within its profile.
        source: Generating run facts: run name, global step, source revision, generating model, dump path.
        tokenizer: Tokenizer HF id and revision.
        num_sequences: Rows in the stored batch (data-parallel padding excluded).
        response_length: Padded response width, ``metadata["response_length"]``.
        total_tokens: Attended tokens, ``attention_mask.sum()``.
        response_tokens: Response tokens, ``response_mask.sum()``.
        tensors_sha256: SHA-256 of the ``tensors.safetensors`` bytes.
    """

    schema_version: int
    ensemble_version: str
    profile: LengthProfile
    batch_index: int
    source: dict[str, Any]
    tokenizer: str
    num_sequences: int
    response_length: int
    total_tokens: int
    response_tokens: int
    tensors_sha256: str


def ensemble_batch_path(root: str, ensemble_version: str, profile: LengthProfile, batch_index: int) -> str:
    return f"{root.rstrip('/')}/{ensemble_version}/{profile}/batch_{batch_index}"


def _routes_path(batch_path: str, checkpoint_id: str) -> str:
    if not CHECKPOINT_ID_PATTERN.match(checkpoint_id):
        raise ValueError(f"checkpoint_id {checkpoint_id!r} must match {CHECKPOINT_ID_PATTERN.pattern}")
    return f"{batch_path.rstrip('/')}/{ROUTES_DIR}/{checkpoint_id}.safetensors"


def _tensors_payload(batch: TrainingInputBatch) -> bytes:
    present = {key: value.contiguous() for key, value in batch.items() if value is not None}
    none_keys = sorted(key for key, value in batch.items() if value is None)
    return safetensors.torch.save(present, metadata={NONE_KEYS_METADATA: json.dumps(none_keys)})


def _safetensors_metadata(payload: bytes) -> dict[str, str]:
    (header_length,) = struct.unpack("<Q", payload[:8])
    header = json.loads(payload[8 : 8 + header_length])
    return header.get("__metadata__", {})


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def benchmark_batch(batch: TrainingInputBatch) -> TrainingInputBatch:
    """Return the replayable part of a dumped training batch.

    Removes the generating run's data-parallel padding rows, its route targets (compact rows and any dense
    tensor), and all metadata except ``response_length``. Every other tensor key, including ``None`` values, is
    kept so the policy loss sees what the production learner saw.
    """
    missing = [key for key in REQUIRED_KEYS if key not in batch]
    if missing:
        raise ValueError(f"dumped batch lacks keys required by ppo_train: {missing}")
    real_rows = batch.batch_size - int(batch.metadata["pad_size"])
    tensors = {
        key: None if value is None else value[:real_rows].contiguous()
        for key, value in batch.items()
        if key != ROUTED_EXPERTS_KEY
    }
    result = TrainingInputBatch(tensors)
    result.metadata = {"response_length": int(batch.metadata["response_length"])}
    return result


def batch_manifest(
    batch: TrainingInputBatch,
    *,
    ensemble_version: str,
    profile: LengthProfile,
    batch_index: int,
    source: dict[str, Any],
    tokenizer: str,
) -> EnsembleBatchManifest:
    """Build the manifest for a replayable batch, computing its length statistics and tensor digest."""
    response_length = int(batch.metadata["response_length"])
    if batch["response_mask"].shape[1] != response_length:
        raise ValueError(f"response_mask width {batch['response_mask'].shape[1]} != response_length {response_length}")
    return EnsembleBatchManifest(
        schema_version=SCHEMA_VERSION,
        ensemble_version=ensemble_version,
        profile=LengthProfile(profile),
        batch_index=batch_index,
        source=source,
        tokenizer=tokenizer,
        num_sequences=batch.batch_size,
        response_length=response_length,
        total_tokens=int(batch["attention_mask"].sum()),
        response_tokens=int(batch["response_mask"].sum()),
        tensors_sha256=_sha256(_tensors_payload(batch)),
    )


def convert_dump(
    dump_path: str,
    *,
    ensemble_version: str,
    profile: LengthProfile,
    batch_index: int,
    source: dict[str, Any],
    tokenizer: str,
) -> tuple[TrainingInputBatch, EnsembleBatchManifest]:
    """Load one ``dump_data`` pickle and convert it to an ensemble batch.

    ``dump_path`` must be a ``global_step_<N>_training_input.pkl`` file written by ``RayPPOTrainer.dump_data``;
    the dump is a trusted pickle produced by our own trainer. ``source`` records the generating run (run name,
    source revision, generating model); the dump path and global step are added to it.
    """
    match = DUMP_FILE_PATTERN.search(dump_path)
    if match is None:
        raise ValueError(f"{dump_path} is not a trainer dump_data file")
    dumped = pickle.loads(io.read_bytes(dump_path))
    if not isinstance(dumped, TrainingInputBatch):
        raise ValueError(f"{dump_path} holds {type(dumped).__name__}, not a TrainingInputBatch")
    if dumped.routed_expert_rows is not None or dumped.get(ROUTED_EXPERTS_KEY) is not None:
        logger.info("Dropping routes of the generating model from {}", dump_path)
    batch = benchmark_batch(dumped)
    manifest = batch_manifest(
        batch,
        ensemble_version=ensemble_version,
        profile=profile,
        batch_index=batch_index,
        source={**source, "dump_path": dump_path, "global_step": int(match.group(1))},
        tokenizer=tokenizer,
    )
    return batch, manifest


def write_ensemble_batch(path: str, batch: TrainingInputBatch, manifest: EnsembleBatchManifest) -> None:
    """Write a batch and its manifest to a batch directory; the manifest is written last.

    Raises:
        ValueError: if the batch carries routes, or its statistics or digest disagree with the manifest.
    """
    if batch.routed_expert_rows is not None or batch.get(ROUTED_EXPERTS_KEY) is not None:
        raise ValueError("ensemble batches are stored without routes; write routes with write_routes")
    expected = batch_manifest(
        batch,
        ensemble_version=manifest.ensemble_version,
        profile=manifest.profile,
        batch_index=manifest.batch_index,
        source=manifest.source,
        tokenizer=manifest.tokenizer,
    )
    if expected != manifest:
        raise ValueError(f"manifest does not describe the batch: expected {expected}, got {manifest}")
    path = path.rstrip("/")
    io.write_bytes_atomic(f"{path}/{TENSORS_FILE}", _tensors_payload(batch))
    io.write_bytes_atomic(f"{path}/{MANIFEST_FILE}", json.dumps(dataclasses.asdict(manifest), indent=2).encode())


def read_manifest(path: str) -> EnsembleBatchManifest:
    fields = json.loads(io.read_bytes(f"{path.rstrip('/')}/{MANIFEST_FILE}"))
    if fields.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported ensemble schema_version {fields.get('schema_version')} at {path}")
    fields["profile"] = LengthProfile(fields["profile"])
    return EnsembleBatchManifest(**fields)


def read_ensemble_batch(
    path: str, *, routes_checkpoint: str | None
) -> tuple[TrainingInputBatch, EnsembleBatchManifest]:
    """Read a batch directory, verifying its tensor digest.

    Args:
        path: Batch directory, ``<root>/<ensemble_version>/<profile>/batch_<i>``.
        routes_checkpoint: Checkpoint id whose routes to attach as ``routed_expert_rows``, or ``None`` for a batch
            without routes.

    Raises:
        ValueError: on a schema or profile mismatch, a digest mismatch, or routes that do not align with the
            batch's response tokens.
    """
    path = path.rstrip("/")
    manifest = read_manifest(path)
    payload = io.read_bytes(f"{path}/{TENSORS_FILE}")
    digest = _sha256(payload)
    if digest != manifest.tensors_sha256:
        raise ValueError(f"{path}/{TENSORS_FILE} sha256 {digest} != manifest {manifest.tensors_sha256}")
    tensors: dict[str, torch.Tensor | None] = dict(safetensors.torch.load(payload))
    for key in json.loads(_safetensors_metadata(payload)[NONE_KEYS_METADATA]):
        tensors[key] = None
    routes = None
    if routes_checkpoint is not None:
        routes = read_routes(path, routes_checkpoint)
        _check_routes_align(routes, tensors["response_mask"], manifest.response_length)
    batch = TrainingInputBatch(tensors, routed_expert_rows=routes)
    batch.metadata = {"response_length": manifest.response_length}
    return batch, manifest


def _check_routes_align(routes: RoutedExpertRows, response_mask: torch.Tensor, response_length: int) -> None:
    if routes.response_len != response_length:
        raise ValueError(f"routes response_len {routes.response_len} != batch response_length {response_length}")
    if len(routes) != len(response_mask):
        raise ValueError(f"routes cover {len(routes)} sequences, batch has {len(response_mask)}")
    response_tokens = response_mask.sum(dim=1).tolist()
    for index, (row, tokens) in enumerate(zip(routes.rows, response_tokens, strict=True)):
        if len(row) != tokens:
            raise ValueError(f"sequence {index}: {len(row)} route rows for {tokens} response tokens")


def write_routes(path: str, checkpoint_id: str, routes: RoutedExpertRows) -> None:
    """Write response-aligned routes for a batch directory as ragged ``rows`` plus ``offsets``.

    Rows are stored in the dtype ``RoutedExpertRows`` uses for ``num_experts`` (uint8 up to 256 experts, int16
    up to 32768).
    """
    if not routes.rows:
        raise ValueError("routes must cover at least one sequence")
    layers, top_k = routes.rows[0].shape[1:]
    if any(row.shape[1:] != (layers, top_k) for row in routes.rows):
        raise ValueError("every route row must share the same [layers, top_k] shape")
    rows = np.concatenate(routes.rows)
    if rows.size and (rows.min() < 0 or rows.max() >= routes.num_experts):
        raise ValueError(f"expert ids must lie in [0, {routes.num_experts})")
    numpy_dtype = torch.empty(0, dtype=routes.dtype).numpy().dtype
    offsets = np.zeros(len(routes) + 1, dtype=np.int64)
    np.cumsum([len(row) for row in routes.rows], out=offsets[1:])
    metadata = {
        "checkpoint_id": checkpoint_id,
        "num_experts": str(routes.num_experts),
        "layers": str(layers),
        "top_k": str(top_k),
        "response_len": str(routes.response_len),
    }
    payload = safetensors.numpy.save(
        {ROUTE_ROWS_KEY: rows.astype(numpy_dtype), ROUTE_OFFSETS_KEY: offsets}, metadata=metadata
    )
    io.write_bytes_atomic(_routes_path(path, checkpoint_id), payload)


def read_routes(path: str, checkpoint_id: str) -> RoutedExpertRows:
    payload = io.read_bytes(_routes_path(path, checkpoint_id))
    metadata = _safetensors_metadata(payload)
    if metadata["checkpoint_id"] != checkpoint_id:
        raise ValueError(f"routes file names checkpoint {metadata['checkpoint_id']}, expected {checkpoint_id}")
    arrays = safetensors.numpy.load(payload)
    rows, offsets = arrays[ROUTE_ROWS_KEY], arrays[ROUTE_OFFSETS_KEY]
    if offsets[0] != 0 or offsets[-1] != len(rows) or np.any(np.diff(offsets) < 0):
        raise ValueError(f"routes offsets for {checkpoint_id} do not partition {len(rows)} rows")
    return RoutedExpertRows(
        tuple(rows[start:stop] for start, stop in zip(offsets[:-1], offsets[1:], strict=True)),
        int(metadata["response_len"]),
        int(metadata["num_experts"]),
    )


def list_ensemble(root: str, ensemble_version: str) -> list[str]:
    """Return batch directories of an ensemble, ordered by (profile, batch index)."""
    batches: list[tuple[int, int, str]] = []
    for profile_order, profile in enumerate(LengthProfile):
        profile_path = f"{root.rstrip('/')}/{ensemble_version}/{profile}"
        if not io.exists(profile_path):
            continue
        for entry in io.list_dir(profile_path):
            match = BATCH_DIR_PATTERN.match(entry.rstrip("/").rsplit("/", 1)[-1])
            if match is not None:
                index = int(match.group(1))
                batches.append((profile_order, index, ensemble_batch_path(root, ensemble_version, profile, index)))
    return [path for _, _, path in sorted(batches)]


def _convert_command(args: argparse.Namespace) -> None:
    profile = LengthProfile(args.profile)
    batch, manifest = convert_dump(
        args.dump,
        ensemble_version=args.ensemble_version,
        profile=profile,
        batch_index=args.batch_index,
        source={"run": args.run, "revision": args.revision, "model": args.model},
        tokenizer=args.tokenizer,
    )
    path = ensemble_batch_path(args.out, args.ensemble_version, profile, args.batch_index)
    write_ensemble_batch(path, batch, manifest)
    print(json.dumps({"path": path, **dataclasses.asdict(manifest)}, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m skyrl_train.benchmarks.learner.batch_artifact")
    commands = parser.add_subparsers(required=True)
    convert = commands.add_parser("convert", help="convert one trainer dump_data pickle into an ensemble batch")
    convert.add_argument("--dump", required=True, help="global_step_<N>_training_input.pkl path or URI")
    convert.add_argument("--out", required=True, help="ensemble root path or URI")
    convert.add_argument("--profile", required=True, choices=[profile.value for profile in LengthProfile])
    convert.add_argument("--ensemble-version", required=True)
    convert.add_argument("--batch-index", required=True, type=int)
    convert.add_argument("--tokenizer", required=True, help="HF tokenizer id and revision")
    convert.add_argument("--run", required=True, help="generating run name")
    convert.add_argument("--revision", required=True, help="source revision of the generating run")
    convert.add_argument("--model", required=True, help="model that generated the text")
    convert.set_defaults(handler=_convert_command)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
