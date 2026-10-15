"""Recompute a model's MoE expert routes for a learner-benchmark ensemble with a vLLM prefill pass.

The learner replays routes only at response-predicting positions
(``megatron_router_replay.dense_replay_targets``): route row ``j`` of a sequence belongs to the position
that predicts response token ``j``, starting with the final prompt token. For a sequence with ``P`` prompt
tokens and ``R`` response tokens those are absolute positions ``P - 1 .. P + R - 2``. The final response
token predicts nothing and gets no row.

The pass sends each full sequence (prompt + response, padding stripped) to vLLM as a ``/completions``
prompt with ``max_tokens=1`` and ``routed_experts_prompt_start=P - 1``. When prefill completes, the pinned
vLLM fork's scheduler reads the routing of prompt positions ``[routed_experts_prompt_start, P + R)`` from its
per-KV-slot buffer; the single sampled token is never forwarded, so it adds no row. The completion choice
carries that ``[R + 1, layers, top_k]`` array as a base64 ``.npy`` string, and this module drops its last row
(the final response token).

GPU layout: the CLI starts one TP=1, EP=DP vLLM engine per ``--data-parallel-size`` GPUs through
``create_ray_wrapped_inference_engines`` (the same factory the trainer and local teachers use) and needs a
running Ray cluster with that many GPUs. Hero BF16 serving needs about 32 H100s or 16 GB200s.
"""

import argparse
import asyncio
import json
import logging
import posixpath
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import ray
import torch
from transformers import AutoConfig

from skyrl_train.benchmarks.learner.batch_artifact import list_ensemble, read_ensemble_batch, write_routes
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.inference_engines.base import InferenceEngineInterface
from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
from skyrl_train.inference_engines.vllm_teacher_oracle import close_owned_inference_engine
from skyrl_train.io import io
from skyrl_train.tokenizer import create_tokenizer
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trajectory_runners.routed_experts import decode_routed_experts

logger = logging.getLogger(__name__)

ROUTES_DIR = "routes"
ROUTES_SUFFIX = ".safetensors"
# The same attribute names the trainer resolves the routed-expert count from.
NUM_EXPERTS_CONFIG_ATTRS = ("num_experts", "num_local_experts", "n_routed_experts")
NUMPY_ROUTE_DTYPES = {torch.uint8: np.uint8, torch.int16: np.int16, torch.int64: np.int64}


@dataclass(frozen=True)
class RouteRequest:
    """One sequence's attended token ids, split at the prompt/response boundary."""

    token_ids: tuple[int, ...]
    prompt_length: int

    def __post_init__(self) -> None:
        if not 1 <= self.prompt_length <= len(self.token_ids):
            raise ValueError(f"prompt_length={self.prompt_length} must be in [1, {len(self.token_ids)}]")

    @property
    def response_length(self) -> int:
        return len(self.token_ids) - self.prompt_length

    @property
    def capture_start(self) -> int:
        """First captured position: the final prompt token, which predicts response token 0."""
        return self.prompt_length - 1

    @property
    def captured_rows(self) -> int:
        """Rows vLLM returns for positions ``capture_start .. len(token_ids) - 1``."""
        return len(self.token_ids) - self.capture_start


def _contiguous_span(mask: torch.Tensor) -> tuple[int, int]:
    """Return ``[start, end)`` of the ones in a 0/1 row, rejecting holes."""
    ones = torch.nonzero(mask, as_tuple=False).flatten()
    if ones.numel() == 0:
        return 0, 0
    start, end = int(ones[0]), int(ones[-1]) + 1
    if end - start != ones.numel():
        raise ValueError("attention mask has a hole inside a sequence")
    return start, end


def route_requests(batch: TrainingInputBatch) -> list[RouteRequest]:
    """Strip padding from each ``TrainingInputBatch`` row to recover the exact token ids the learner sees.

    The layout is the trainer's: a left-padded prompt window followed by a right-padded response window of
    ``metadata["response_length"]`` columns. The learner's replay placement assumes the final prompt token
    sits in the last prompt column, so a prompt that does not end there is rejected.
    """
    sequences = batch["sequences"]
    attention_mask = batch["attention_mask"]
    response_mask = batch["response_mask"]
    response_width = int(batch.metadata["response_length"])
    prompt_width = sequences.shape[1] - response_width
    if response_mask.shape[1] != response_width or prompt_width < 1:
        raise ValueError(
            f"response window {response_width} does not fit sequences {tuple(sequences.shape)} "
            f"and response_mask {tuple(response_mask.shape)}"
        )
    requests = []
    for row in range(sequences.shape[0]):
        prompt_start, prompt_end = _contiguous_span(attention_mask[row, :prompt_width])
        if prompt_end != prompt_width or prompt_start == prompt_end:
            raise ValueError(f"row {row}: prompt must be non-empty and left-padded against the response window")
        response_start, response_end = _contiguous_span(attention_mask[row, prompt_width:])
        if response_start != 0:
            raise ValueError(f"row {row}: response must start at the response window")
        if not torch.equal(response_mask[row].bool(), attention_mask[row, prompt_width:].bool()):
            raise ValueError(f"row {row}: response_mask disagrees with the attended response tokens")
        token_ids = sequences[row, prompt_start : prompt_width + response_end].tolist()
        requests.append(RouteRequest(tuple(token_ids), prompt_width - prompt_start))
    return requests


def completion_payload(request: RouteRequest, served_model_name: str, request_id: str) -> dict[str, Any]:
    """Build the ``/completions`` payload that prefills one sequence and returns its prediction routes."""
    return {
        "json": {
            "model": served_model_name,
            "prompt": list(request.token_ids),
            "max_tokens": 1,
            "routed_experts_prompt_start": request.capture_start,
        },
        "headers": {"x-request-id": request_id},
    }


def captured_routes_from_completion(response: dict[str, Any], request: RouteRequest) -> np.ndarray:
    """Decode the ``[captured_rows, layers, top_k]`` route array from a completion response."""
    choices = response.get("choices")
    if not choices:
        raise ValueError(f"vLLM completion failed: {response.get('error', response)}")
    if len(choices) != 1:
        raise ValueError(f"expected one completion choice, got {len(choices)}")
    routes = choices[0].get("routed_experts")
    if routes is None:
        raise ValueError("vLLM returned no routed_experts; start the engine with enable_return_routed_experts")
    return decode_routed_experts(routes, request.captured_rows)


def response_prediction_rows(captured: np.ndarray, request: RouteRequest) -> np.ndarray:
    """Select the rows that predict each response token from a capture starting at ``capture_start``.

    ``captured[j]`` is the route at absolute position ``capture_start + j``; rows ``0 .. R - 1`` predict
    response tokens ``0 .. R - 1``. The last row (the final response token) predicts nothing.
    """
    if captured.ndim != 3 or captured.shape[0] != request.captured_rows:
        raise ValueError(f"captured routes {captured.shape} do not cover {request.captured_rows} positions")
    return captured[: request.response_length]


def routed_expert_rows(
    rows: Sequence[np.ndarray], response_width: int, num_experts: int, *, layers: int, top_k: int
) -> RoutedExpertRows:
    """Validate per-sequence route rows and narrow them to the learner's ``num_experts`` dtype."""
    if num_experts < 1:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    dtype = NUMPY_ROUTE_DTYPES[RoutedExpertRows((), response_width, num_experts).dtype]
    narrowed = []
    for index, sequence_rows in enumerate(rows):
        if sequence_rows.ndim != 3 or sequence_rows.shape[1:] != (layers, top_k):
            raise ValueError(f"sequence {index}: routes {sequence_rows.shape} are not [tokens, {layers}, {top_k}]")
        if not np.issubdtype(sequence_rows.dtype, np.integer):
            raise ValueError(f"sequence {index}: routes must be integer expert ids")
        if sequence_rows.size and (sequence_rows.min() < 0 or sequence_rows.max() >= num_experts):
            raise ValueError(f"sequence {index}: expert id outside [0, {num_experts})")
        narrowed.append(np.ascontiguousarray(sequence_rows, dtype=dtype))
    return RoutedExpertRows(tuple(narrowed), response_width, num_experts)


def engine_assignments(token_counts: Sequence[int], num_engines: int) -> list[int]:
    """Assign requests to engines, longest first, each to the engine with the fewest assigned tokens."""
    if num_engines < 1:
        raise ValueError(f"num_engines must be positive, got {num_engines}")
    loads = [0] * num_engines
    assignment = [0] * len(token_counts)
    for index in sorted(range(len(token_counts)), key=lambda i: -token_counts[i]):
        engine = min(range(num_engines), key=lambda e: loads[e])
        assignment[index] = engine
        loads[engine] += token_counts[index]
    return assignment


class CompletionEngine(Protocol):
    async def completion(self, request_payload: dict[str, Any]) -> dict[str, Any]: ...


async def prediction_routes(
    engines: Sequence[CompletionEngine],
    requests: Sequence[RouteRequest],
    *,
    served_model_name: str,
    max_in_flight_per_engine: int,
    request_id_prefix: str,
) -> list[np.ndarray]:
    """Prefill every request on the engines and return each sequence's ``[R, layers, top_k]`` routes."""
    assignment = engine_assignments([len(request.token_ids) for request in requests], len(engines))
    limits = [asyncio.Semaphore(max_in_flight_per_engine) for _ in engines]

    async def one(index: int) -> np.ndarray:
        request = requests[index]
        engine = assignment[index]
        async with limits[engine]:
            response = await engines[engine].completion(
                completion_payload(request, served_model_name, f"{request_id_prefix}-{index}")
            )
        return response_prediction_rows(captured_routes_from_completion(response, request), request)

    return list(await asyncio.gather(*(one(index) for index in range(len(requests)))))


async def batch_routes(
    engines: Sequence[CompletionEngine],
    batch: TrainingInputBatch,
    *,
    served_model_name: str,
    num_experts: int,
    max_in_flight_per_engine: int,
    request_id_prefix: str,
) -> RoutedExpertRows:
    """Compute the response-aligned routes for one ensemble batch."""
    requests = route_requests(batch)
    rows = await prediction_routes(
        engines,
        requests,
        served_model_name=served_model_name,
        max_in_flight_per_engine=max_in_flight_per_engine,
        request_id_prefix=request_id_prefix,
    )
    _, layers, top_k = rows[0].shape
    return routed_expert_rows(rows, int(batch.metadata["response_length"]), num_experts, layers=layers, top_k=top_k)


def routes_path(batch_dir: str, checkpoint_id: str) -> str:
    return posixpath.join(batch_dir, ROUTES_DIR, f"{checkpoint_id}{ROUTES_SUFFIX}")


def num_experts_from_config(model: str, revision: str | None) -> int:
    config = AutoConfig.from_pretrained(model, revision=revision, trust_remote_code=True)
    for attr in NUM_EXPERTS_CONFIG_ATTRS:
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    raise ValueError(f"{model} config has none of {NUM_EXPERTS_CONFIG_ATTRS}")


@dataclass(frozen=True)
class RouteEngineConfig:
    """vLLM geometry and limits for the routing pass."""

    model: str
    revision: str | None
    num_engines: int
    data_parallel_size: int
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    gpu_memory_utilization: float
    enable_prefix_caching: bool
    enforce_eager: bool
    engine_init_timeout_seconds: float
    engine_init_kwargs: dict[str, Any]


def create_route_engines(config: RouteEngineConfig) -> list[InferenceEngineInterface]:
    """Start TP=1, EP=DP vLLM engines with routed-expert capture; returns one handle per DP rank."""
    tokenizer = create_tokenizer(config.model, disable_fast_tokenizer=False, revision=config.revision)
    engine_init_kwargs = {
        **config.engine_init_kwargs,
        "enable_return_routed_experts": True,
        "max_model_len": config.max_model_len,
        "served_model_name": config.model,
    }
    if config.revision is not None:
        engine_init_kwargs["revision"] = config.revision
    return create_ray_wrapped_inference_engines(
        num_inference_engines=config.num_engines,
        tensor_parallel_size=1,
        data_parallel_size=config.data_parallel_size,
        expert_parallel_size=config.data_parallel_size,
        model_dtype="bfloat16",
        pretrain=config.model,
        seed=0,
        vllm_v1_disable_multiproc=True,
        enable_prefix_caching=config.enable_prefix_caching,
        enforce_eager=config.enforce_eager,
        engine_init_timeout_seconds=config.engine_init_timeout_seconds,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_num_batched_tokens=config.max_num_batched_tokens,
        max_num_seqs=config.max_num_seqs,
        tokenizer=tokenizer,
        backend="vllm",
        engine_init_kwargs=engine_init_kwargs,
    )


async def run_route_pass(
    config: RouteEngineConfig,
    *,
    ensemble_root: str,
    ensemble_version: str,
    checkpoint_id: str,
    max_in_flight_per_engine: int,
    overwrite: bool,
) -> None:
    """Write ``routes/<checkpoint_id>.safetensors`` for every batch in the ensemble version."""
    num_experts = num_experts_from_config(config.model, config.revision)
    batch_dirs = list_ensemble(ensemble_root, ensemble_version)
    pending = [path for path in batch_dirs if overwrite or not io.exists(routes_path(path, checkpoint_id))]
    logger.info("routing %d of %d batches with %s", len(pending), len(batch_dirs), config.model)
    if not pending:
        return
    engines = create_route_engines(config)
    try:
        for batch_dir in pending:
            batch, manifest = read_ensemble_batch(batch_dir, routes_checkpoint=None)
            rows = await batch_routes(
                engines,
                batch,
                served_model_name=config.model,
                num_experts=num_experts,
                max_in_flight_per_engine=max_in_flight_per_engine,
                request_id_prefix=f"{manifest.profile}-{manifest.batch_index}",
            )
            write_routes(batch_dir, checkpoint_id, rows)
            logger.info("wrote %s routes %s for %s", checkpoint_id, rows.shape, batch_dir)
    finally:
        await asyncio.gather(*(close_owned_inference_engine(engine) for engine in engines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-root", required=True)
    parser.add_argument("--ensemble-version", required=True)
    parser.add_argument("--model", required=True, help="HF checkpoint path or id served by vLLM")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--checkpoint-id", required=True, help="name of the routes file written per batch")
    parser.add_argument("--num-engines", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, required=True, help="GPUs per engine; EP = DP, TP = 1")
    parser.add_argument("--max-model-len", type=int, required=True, help="must exceed the longest sequence")
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--engine-init-timeout-seconds", type=float, default=1800)
    parser.add_argument(
        "--engine-init-kwargs", type=json.loads, default={}, help='extra vLLM kwargs as JSON, e.g. {"load_format": …}'
    )
    parser.add_argument("--max-in-flight-per-engine", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true", help="recompute batches that already have routes")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    config = RouteEngineConfig(
        model=args.model,
        revision=args.revision,
        num_engines=args.num_engines,
        data_parallel_size=args.data_parallel_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
        enforce_eager=args.enforce_eager,
        engine_init_timeout_seconds=args.engine_init_timeout_seconds,
        engine_init_kwargs=args.engine_init_kwargs,
    )
    ray.init()
    asyncio.run(
        run_route_pass(
            config,
            ensemble_root=args.ensemble_root,
            ensemble_version=args.ensemble_version,
            checkpoint_id=args.checkpoint_id,
            max_in_flight_per_engine=args.max_in_flight_per_engine,
            overwrite=args.overwrite,
        )
    )


if __name__ == "__main__":
    main()
