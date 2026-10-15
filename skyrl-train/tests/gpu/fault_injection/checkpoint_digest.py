from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import json

import numpy as np
import torch


CHUNK_BYTES = 64 * 1024 * 1024


def digest_value(value):
    """Hash exact tensor values and tree structure with bounded CPU copies."""
    digest = hashlib.sha256()
    tensor_count = tensor_bytes = 0

    def record(tag, payload):
        data = payload.encode() if isinstance(payload, str) else payload
        digest.update(tag.encode())
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)

    def visit(item):
        nonlocal tensor_count, tensor_bytes
        if isinstance(item, torch.Tensor):
            tensor_count += 1
            record("tensor", json.dumps([str(item.dtype), list(item.shape)]))
            raw = item.detach().contiguous().reshape(-1).view(torch.uint8)
            tensor_bytes += raw.numel()
            record("data_length", str(raw.numel()))
            for offset in range(0, raw.numel(), CHUNK_BYTES):
                digest.update(raw[offset : offset + CHUNK_BYTES].cpu().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            visit(torch.from_numpy(np.ascontiguousarray(item)))
        elif isinstance(item, np.generic):
            visit(item.item())
        elif (
            type(item).__module__.startswith("megatron.core.dist_checkpointing.")
            and type(item).__name__ == "LocalNonpersistentObject"
        ):
            record("nonpersistent", "")
        elif type(item).__module__.startswith("megatron.core.dist_checkpointing.") and hasattr(item, "data"):
            visit(item.data)
        elif isinstance(item, Mapping):
            record("mapping", str(len(item)))
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            record(type(item).__name__, str(len(item)))
            for child in item:
                visit(child)
        elif is_dataclass(item) and not isinstance(item, type):
            visit({field.name: getattr(item, field.name) for field in fields(item)})
        elif isinstance(item, (str, bytes, int, float, bool, type(None), torch.dtype, torch.device, np.dtype)):
            record(type(item).__name__, repr(item))
        else:
            raise TypeError(f"Unsupported parity state {type(item).__module__}.{type(item).__name__}")

    visit(value)
    return {"sha256": digest.hexdigest(), "tensor_count": tensor_count, "tensor_bytes": tensor_bytes}
