"""Build a depth-truncated ("shallow") Grug checkpoint from a full Hugging Face export.

The truncated model keeps a contiguous prefix of decoder layers, so routes recorded for the
full model remain exact replay targets for every kept layer. Every non-layer tensor (embedding,
embedding and final norms, LM head) and every config field other than the layer count is
unchanged.

The copy streams safetensors bytes between object stores without staging weights on local disk.
A source shard is read only when it holds a kept tensor: a shard of only kept tensors is copied
byte for byte, and a shard that mixes kept and dropped tensors is rewritten from range reads of
its kept tensors. The output carries a fresh weight index, the source's non-weight files, a
provenance record and a model manifest. The manifest is written last and marks completion.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import posixpath
import re
from typing import Any, BinaryIO

import numpy as np
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
from loguru import logger

from marinskyrl.model_manifest import (
    HF_WEIGHT_INDEX_FILENAME,
    MODEL_MANIFEST_FILENAME,
    SAFETENSORS_LENGTH_PREFIX_BYTES,
    ModelManifest,
    ModelManifestFile,
    build_model_manifest,
    build_safetensors_weight_index,
    read_safetensors_header,
)
from marinskyrl.remote_io import filesystem_and_path, open_output_stream
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.models.grug_moe import GRUG_MOE_MODEL_TYPE, GrugMoeConfig

CONFIG_FILENAME = "config.json"
TRUNCATION_PROVENANCE_FILENAME = "truncation_provenance.json"
LAYER_COUNT_CONFIG_KEYS = ("num_hidden_layers", "num_layers")
# Config keys that describe a per-layer schedule in common HF MoE configs. Grug does not use
# them; their presence means the checkpoint has semantics that a plain prefix cut would change.
PER_LAYER_CONFIG_KEYS = frozenset(
    {
        "layer_types",
        "mlp_only_layers",
        "moe_layer_freq",
        "decoder_sparse_step",
        "first_k_dense_replace",
        "no_rope_layers",
        "sliding_window_pattern",
    }
)
# Grug config fields whose list values are not per-layer.
NON_LAYER_LIST_CONFIG_KEYS = frozenset({"architectures", "sconv_sites"})
WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
_LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_HEADER_ALIGNMENT_BYTES = 8
_COPY_CHUNK_BYTES = 8 * 2**20
_OUTPUT_SHARD_NAME = "model-{index:05d}-of-{count:05d}.safetensors"


@dataclass(frozen=True)
class TruncationProvenance:
    """Source of a truncated checkpoint, stored next to its weights."""

    source_path: str
    source_config_sha256: str
    source_manifest_identity: str | None
    source_num_layers: int
    num_layers: int


@dataclass(frozen=True)
class ShardPlan:
    """How one source shard contributes to the truncated checkpoint."""

    source_shard: str
    output_shard: str
    kept_keys: tuple[str, ...]
    rewrite: bool


def truncated_config(config: dict[str, Any], num_layers: int) -> dict[str, Any]:
    """Return a Grug config dict for the first ``num_layers`` layers of ``config``.

    Raises:
        ValueError: if the config is not a Grug config, ``num_layers`` is out of range, the
            final kept layer was not a global-attention layer in the full model, or the config
            carries a per-layer schedule.
    """
    if config.get("model_type") != GRUG_MOE_MODEL_TYPE:
        raise ValueError(f"expected a {GRUG_MOE_MODEL_TYPE} config, got model_type={config.get('model_type')!r}")
    source = GrugMoeConfig.from_dict(config)
    source_num_layers = source.num_hidden_layers
    if not 0 < num_layers <= source_num_layers:
        raise ValueError(f"num_layers must be in [1, {source_num_layers}], got {num_layers}")
    # Grug forces the final layer to global attention. A prefix ending on a layer that was
    # sliding-window in the full model would change that layer's attention.
    if num_layers % source.global_every:
        raise ValueError(
            f"num_layers={num_layers} must be a multiple of global_every={source.global_every} so the truncated "
            "model's final (forced-global) layer is also global in the full model"
        )
    per_layer = sorted(
        key
        for key, value in config.items()
        if key in PER_LAYER_CONFIG_KEYS
        or (isinstance(value, list) and len(value) == source_num_layers and key not in NON_LAYER_LIST_CONFIG_KEYS)
    )
    if per_layer:
        raise ValueError(f"config fields {per_layer} encode a per-layer schedule that truncation does not slice")
    result = dict(config)
    for key in LAYER_COUNT_CONFIG_KEYS:
        if key in result:
            result[key] = num_layers
    GrugMoeConfig.from_dict(result)
    return result


def plan_truncation(
    weight_map: dict[str, str],
    *,
    source_num_layers: int,
    num_layers: int,
) -> tuple[ShardPlan, ...]:
    """Assign kept tensors to output shards, skipping shards with no kept tensor.

    Raises:
        ValueError: if the layer tensors do not cover exactly ``source_num_layers`` layers with
            identical parameter names in every layer.
    """
    suffixes_by_layer: dict[int, set[str]] = defaultdict(set)
    kept_by_shard: dict[str, list[str]] = defaultdict(list)
    dropped_shards: set[str] = set()
    for key, shard in weight_map.items():
        match = _LAYER_KEY.match(key)
        if match is None:
            kept_by_shard[shard].append(key)
            continue
        layer = int(match.group(1))
        suffixes_by_layer[layer].add(match.group(2))
        if layer < num_layers:
            kept_by_shard[shard].append(key)
        else:
            dropped_shards.add(shard)
    if sorted(suffixes_by_layer) != list(range(source_num_layers)):
        raise ValueError(
            f"checkpoint layer indices {sorted(suffixes_by_layer)} do not match num_hidden_layers={source_num_layers}"
        )
    first_layer = suffixes_by_layer[0]
    uneven = sorted(layer for layer, suffixes in suffixes_by_layer.items() if suffixes != first_layer)
    if uneven:
        raise ValueError(f"layers {uneven} have different parameter names from layer 0")

    shards = sorted(kept_by_shard)
    return tuple(
        ShardPlan(
            source_shard=shard,
            output_shard=_OUTPUT_SHARD_NAME.format(index=index, count=len(shards)),
            kept_keys=tuple(sorted(kept_by_shard[shard])),
            rewrite=shard in dropped_shards,
        )
        for index, shard in enumerate(shards, start=1)
    )


def slice_routes(routes: RoutedExpertRows, num_layers: int) -> RoutedExpertRows:
    """Keep the routes of the first ``num_layers`` layers.

    Every Grug decoder layer is a MoE layer, so the route layer axis indexes decoder layers.
    """
    short = [row.shape[1] for row in routes.rows if row.shape[1] < num_layers]
    if short:
        raise ValueError(f"routes carry {min(short)} layers, fewer than num_layers={num_layers}")
    return RoutedExpertRows(
        tuple(np.ascontiguousarray(row[:, :num_layers]) for row in routes.rows),
        routes.response_len,
        routes.num_experts,
    )


def _sliced_safetensors_header(
    header: dict[str, Any], kept_keys: tuple[str, ...]
) -> tuple[bytes, list[tuple[int, int]]]:
    """Return a header for ``kept_keys`` and the coalesced source byte ranges to copy after it."""
    kept = sorted(kept_keys, key=lambda key: header[key]["data_offsets"][0])
    sliced: dict[str, Any] = {}
    if "__metadata__" in header:
        sliced["__metadata__"] = header["__metadata__"]
    ranges: list[tuple[int, int]] = []
    offset = 0
    for key in kept:
        start, end = header[key]["data_offsets"]
        sliced[key] = {**header[key], "data_offsets": [offset, offset + end - start]}
        offset += end - start
        if ranges and ranges[-1][1] == start:
            ranges[-1] = (ranges[-1][0], end)
        else:
            ranges.append((start, end))
    header_json = json.dumps(sliced, separators=(",", ":")).encode()
    header_json += b" " * (-len(header_json) % _HEADER_ALIGNMENT_BYTES)
    return len(header_json).to_bytes(SAFETENSORS_LENGTH_PREFIX_BYTES, "little") + header_json, ranges


def _copy_exact(source: BinaryIO, destination, size: int, digest, source_name: str) -> None:
    remaining = size
    while remaining:
        chunk = source.read(min(remaining, _COPY_CHUNK_BYTES))
        if not chunk:
            raise ValueError(f"truncated read from {source_name}")
        destination.write(chunk)
        digest.update(chunk)
        remaining -= len(chunk)


@dataclass(frozen=True)
class _Store:
    filesystem: AbstractFileSystem
    root: str

    @classmethod
    def at(cls, path: str) -> "_Store":
        filesystem, root = filesystem_and_path(path)
        return cls(filesystem, root.rstrip("/"))

    def path(self, relative: str) -> str:
        return posixpath.join(self.root, relative)

    def read_bytes(self, relative: str) -> bytes:
        with self.filesystem.open(self.path(relative), "rb") as source:
            return source.read()

    def open_output(self, relative: str):
        path = self.path(relative)
        # Object stores have no directories; a local destination needs its parents.
        if isinstance(self.filesystem, LocalFileSystem):
            self.filesystem.makedirs(posixpath.dirname(path), exist_ok=True)
        return open_output_stream(self.filesystem, path)

    def write_bytes(self, relative: str, payload: bytes) -> ModelManifestFile:
        with self.open_output(relative) as destination:
            destination.write(payload)
        return ModelManifestFile(path=relative, size=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def _write_shard(source: _Store, output: _Store, plan: ShardPlan) -> tuple[ModelManifestFile, tuple[str, ...]]:
    source_name = source.path(plan.source_shard)
    digest = hashlib.sha256()
    with source.filesystem.open(source_name, "rb") as reader:
        header_bytes, keys = read_safetensors_header(reader, source_name)
        data_start = len(header_bytes)
        with output.open_output(plan.output_shard) as writer:
            if not plan.rewrite:
                if set(keys) != set(plan.kept_keys):
                    raise ValueError(f"{source_name} holds {keys}, but the index assigns it {plan.kept_keys}")
                writer.write(header_bytes)
                digest.update(header_bytes)
                size = len(header_bytes) + _copy_rest(reader, writer, digest)
            else:
                header = json.loads(header_bytes[SAFETENSORS_LENGTH_PREFIX_BYTES:])
                missing = sorted(set(plan.kept_keys) - set(keys))
                if missing:
                    raise ValueError(f"{source_name} is missing indexed tensors {missing}")
                sliced_header, ranges = _sliced_safetensors_header(header, plan.kept_keys)
                writer.write(sliced_header)
                digest.update(sliced_header)
                size = len(sliced_header)
                for start, end in ranges:
                    reader.seek(data_start + start)
                    _copy_exact(reader, writer, end - start, digest, source_name)
                    size += end - start
    logger.info(
        f"{'Rewrote' if plan.rewrite else 'Copied'} {plan.source_shard} -> {plan.output_shard} "
        f"({len(plan.kept_keys)} tensors, {size} bytes)"
    )
    return ModelManifestFile(path=plan.output_shard, size=size, sha256=digest.hexdigest()), plan.kept_keys


def _copy_rest(reader: BinaryIO, writer, digest) -> int:
    copied = 0
    while chunk := reader.read(_COPY_CHUNK_BYTES):
        writer.write(chunk)
        digest.update(chunk)
        copied += len(chunk)
    return copied


def _copied_file_names(source: _Store) -> list[str]:
    """Non-weight files copied unchanged: tokenizer, generation config, chat template and the like."""
    replaced = {CONFIG_FILENAME, HF_WEIGHT_INDEX_FILENAME, MODEL_MANIFEST_FILENAME, TRUNCATION_PROVENANCE_FILENAME}
    names = []
    for path in source.filesystem.find(source.root, withdirs=False):
        relative = posixpath.relpath(path, source.root)
        if relative in replaced or relative.endswith(WEIGHT_FILE_SUFFIXES) or relative.split("/")[0] == ".cache":
            continue
        names.append(relative)
    return sorted(names)


def _source_manifest(source: _Store) -> ModelManifest | None:
    if not source.filesystem.exists(source.path(MODEL_MANIFEST_FILENAME)):
        return None
    marker = source.path(MODEL_MANIFEST_FILENAME)
    return ModelManifest.from_mapping(json.loads(source.read_bytes(MODEL_MANIFEST_FILENAME)), marker)


def _verify_against_manifest(manifest: ModelManifest | None, plans: tuple[ShardPlan, ...], files) -> None:
    """Check byte-for-byte shard copies against the source manifest's digests."""
    if manifest is None:
        return
    expected = {entry.path: entry for entry in manifest.files}
    for plan, (entry, _keys) in zip(plans, files, strict=True):
        if plan.rewrite:
            continue
        source_entry = expected[plan.source_shard]
        if (entry.size, entry.sha256) != (source_entry.size, source_entry.sha256):
            raise ValueError(f"copied shard {plan.source_shard} does not match the source manifest digest")


def truncate_checkpoint(
    source_path: str,
    output_path: str,
    num_layers: int,
    *,
    max_concurrent_shards: int,
) -> TruncationProvenance:
    """Write the first ``num_layers`` layers of a Grug HF checkpoint to ``output_path``.

    Args:
        source_path: Local directory or object-store prefix holding ``config.json``, a
            ``model.safetensors.index.json`` and its shards, and tokenizer files.
        output_path: Local directory or object-store prefix. It must not already hold files.
        num_layers: Number of leading decoder layers to keep; a multiple of ``global_every``.
        max_concurrent_shards: Shards streamed at once.

    Returns:
        The provenance record also written to ``truncation_provenance.json``.
    """
    if max_concurrent_shards < 1:
        raise ValueError(f"max_concurrent_shards must be positive, got {max_concurrent_shards}")
    source = _Store.at(source_path)
    output = _Store.at(output_path)
    if output.filesystem.exists(output.root) and output.filesystem.find(output.root, withdirs=False):
        raise FileExistsError(f"truncated checkpoint output already holds files: {output_path}")

    config_bytes = source.read_bytes(CONFIG_FILENAME)
    config = json.loads(config_bytes)
    new_config = truncated_config(config, num_layers)
    source_num_layers = GrugMoeConfig.from_dict(config).num_hidden_layers
    weight_map = json.loads(source.read_bytes(HF_WEIGHT_INDEX_FILENAME))["weight_map"]
    plans = plan_truncation(weight_map, source_num_layers=source_num_layers, num_layers=num_layers)
    source_manifest = _source_manifest(source)
    provenance = TruncationProvenance(
        source_path=source_path,
        source_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        source_manifest_identity=source_manifest.identity if source_manifest else None,
        source_num_layers=source_num_layers,
        num_layers=num_layers,
    )
    logger.info(
        f"Truncating {source_path} from {source_num_layers} to {num_layers} layers: {len(plans)} shards "
        f"({sum(plan.rewrite for plan in plans)} rewritten) -> {output_path}"
    )

    with ThreadPoolExecutor(max_workers=max_concurrent_shards, thread_name_prefix="truncate-shard") as executor:
        shard_files = list(executor.map(lambda plan: _write_shard(source, output, plan), plans))
    _verify_against_manifest(source_manifest, plans, shard_files)

    entries = [entry for entry, _keys in shard_files]
    for name in _copied_file_names(source):
        entries.append(output.write_bytes(name, source.read_bytes(name)))
    entries.append(
        output.write_bytes(CONFIG_FILENAME, (json.dumps(new_config, indent=2, sort_keys=True) + "\n").encode())
    )
    index = build_safetensors_weight_index(
        {entry.path: (entry.size, keys) for entry, keys in shard_files}, output.path(HF_WEIGHT_INDEX_FILENAME)
    )
    entries.append(output.write_bytes(HF_WEIGHT_INDEX_FILENAME, index))
    entries.append(
        output.write_bytes(
            TRUNCATION_PROVENANCE_FILENAME, (json.dumps(asdict(provenance), indent=2, sort_keys=True) + "\n").encode()
        )
    )
    manifest = build_model_manifest(tuple(sorted(entries, key=lambda entry: entry.path)), None, None)
    output.write_bytes(
        MODEL_MANIFEST_FILENAME,
        (json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode(),
    )
    logger.info(f"Wrote truncated checkpoint {output_path} (identity={manifest.identity})")
    return provenance


def read_truncation_provenance(checkpoint_path: str) -> TruncationProvenance:
    store = _Store.at(checkpoint_path)
    return TruncationProvenance(**json.loads(store.read_bytes(TRUNCATION_PROVENANCE_FILENAME)))
