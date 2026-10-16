"""Bounded ownership summaries for live Python CUDA tensor storage."""

import gc
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping

import torch
from loguru import logger


def tensor_ownership_report(*, device: torch.device, known_tensors: Mapping[str, Iterable[torch.Tensor]]) -> dict:
    """Summarize storage owners without copying values or collecting garbage.

    The census covers Python Tensor and Parameter objects on the current device.
    Allocations with no such Python object remain outside its coverage. Shared
    storage counts once, including when several named owners refer to it.
    """
    known_owners: dict[int, set[str]] = defaultdict(set)
    known_storage_bytes: dict[int, int] = {}
    known_cpu_bytes: Counter[str] = Counter()
    for category, tensors in known_tensors.items():
        for tensor in tensors:
            if tensor.device.type == "cpu":
                known_cpu_bytes[category] += tensor.numel() * tensor.element_size()
            if tensor.device == device:
                storage = tensor.untyped_storage()
                if storage.nbytes():
                    pointer = storage.data_ptr()
                    known_owners[pointer].add(category)
                    known_storage_bytes[pointer] = storage.nbytes()

    objects = gc.get_objects()
    storage_bytes = dict(known_storage_bytes)
    examples: dict[int, torch.Tensor] = {}
    shapes: dict[int, set[str]] = defaultdict(set)
    tensor_count: Counter[int] = Counter()
    for value in objects:
        if type(value) not in (torch.Tensor, torch.nn.Parameter):
            continue
        if value.device != device:
            continue
        storage = value.untyped_storage()
        if not storage.nbytes():
            continue
        pointer = storage.data_ptr()
        storage_bytes[pointer] = storage.nbytes()
        tensor_count[pointer] += 1
        if len(shapes[pointer]) < 3:
            shapes[pointer].add(f"{value.dtype}:{tuple(value.shape)}")
        if pointer not in known_owners:
            examples[pointer] = value

    owner_bytes: Counter[str] = Counter()
    owner_storages: Counter[str] = Counter()
    for pointer, size in storage_bytes.items():
        category = "+".join(sorted(known_owners[pointer])) if pointer in known_owners else "unattributed_python_tensor"
        owner_bytes[category] += size
        owner_storages[category] += 1
    largest_unknown = []
    for pointer in sorted(examples, key=storage_bytes.__getitem__, reverse=True)[:8]:
        tensor = examples[pointer]
        referrers = []
        for referrer in gc.get_referrers(tensor):
            if referrer is objects or referrer is examples:
                continue
            description = {"type": f"{type(referrer).__module__}.{type(referrer).__qualname__}"}
            if isinstance(referrer, dict):
                description["keys_referring_to_tensor"] = [
                    str(key)[:120] if isinstance(key, (str, int)) else type(key).__qualname__
                    for key, value in referrer.items()
                    if value is tensor
                ][:8]
                description["object_owners"] = sorted(
                    {
                        f"{type(parent).__module__}.{type(parent).__qualname__}"
                        for parent in gc.get_referrers(referrer)
                        if type(parent) not in (dict, list, tuple)
                    }
                )[:8]
            referrers.append(description)
            if len(referrers) == 8:
                break
        largest_unknown.append(
            {
                "storage_bytes": storage_bytes[pointer],
                "python_tensor_views": tensor_count[pointer],
                "tensor_examples": sorted(shapes[pointer]),
                "referrers": referrers,
            }
        )
    return {
        "unique_python_storage_bytes": sum(storage_bytes.values()),
        "owner_storage_bytes": dict(owner_bytes),
        "owner_storage_counts": dict(owner_storages),
        "known_cpu_tensor_bytes": dict(known_cpu_bytes),
        "largest_unattributed_storages": largest_unknown,
        "coverage": "Python Tensor/Parameter storage on selected device; allocator allocations without a Python tensor are not attributed. CPU bytes count tensor views and may alias. No tensor values copied and no explicit garbage collection.",
    }


def record_cuda_tensor_ownership(*, rank: int, phase: str, known_tensors: Mapping[str, Iterable[torch.Tensor]]) -> None:
    """Log a storage census alongside CUDA allocator coverage."""
    device = torch.cuda.current_device()
    allocated_before = torch.cuda.memory_allocated(device)
    report = tensor_ownership_report(device=torch.device("cuda", device), known_tensors=known_tensors)
    report.update(
        rank=rank,
        phase=phase,
        cuda_device=device,
        allocated_before_bytes=allocated_before,
        allocated_after_bytes=torch.cuda.memory_allocated(device),
    )
    logger.info("CUDA tensor ownership: {}", json.dumps(report, sort_keys=True))
