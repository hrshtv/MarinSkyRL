"""Score an ensemble batch's response tokens with vLLM, as an independent reference for the Megatron learner.

Each sequence is prefilled as a ``/completions`` prompt with ``echo`` and ``logprobs=0``, which returns the
log-prob of every prompt token given its prefix. The response tokens' entries are written as a
``[rows, response_length]`` float32 tensor (zeros outside the response mask), aligned with the learner's
``forward`` output.
"""

import argparse
import asyncio
import logging

import ray
import safetensors.torch
import torch

from skyrl_train.benchmarks.learner.batch_artifact import read_ensemble_batch
from skyrl_train.benchmarks.learner.route_pass import RouteEngineConfig, create_route_engines, route_requests
from skyrl_train.inference_engines.vllm_teacher_oracle import close_owned_inference_engine

logger = logging.getLogger(__name__)


async def response_logprobs(engines, batch, served_model_name: str) -> torch.Tensor:
    requests = route_requests(batch)
    width = int(batch.metadata["response_length"])

    async def one(index: int) -> list[float]:
        request = requests[index]
        response = await engines[index % len(engines)].completion(
            {
                "json": {
                    "model": served_model_name,
                    "prompt": list(request.token_ids),
                    "max_tokens": 1,
                    "echo": True,
                    "logprobs": 0,
                    "temperature": 0.0,
                },
                "headers": {"x-request-id": f"reference-{index}"},
            }
        )
        choices = response.get("choices")
        if not choices:
            raise ValueError(f"vLLM completion failed: {response.get('error', response)}")
        scores = choices[0]["logprobs"]["token_logprobs"]
        if len(scores) != len(request.token_ids) + 1:
            raise ValueError(f"row {index}: {len(scores)} echoed log-probs for {len(request.token_ids)} prompt tokens")
        return [float(score) for score in scores[request.prompt_length : len(request.token_ids)]]

    rows = await asyncio.gather(*(one(index) for index in range(len(requests))))
    result = torch.zeros(len(rows), width, dtype=torch.float32)
    for index, row in enumerate(rows):
        result[index, : len(row)] = torch.tensor(row)
    return result


async def run(args: argparse.Namespace) -> None:
    batch, _ = read_ensemble_batch(args.batch, routes_checkpoint=None)
    config = RouteEngineConfig(
        model=args.model,
        revision=None,
        num_engines=1,
        data_parallel_size=args.data_parallel_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=8192,
        max_num_seqs=256,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=False,
        enforce_eager=True,
        engine_init_timeout_seconds=1800,
        # Hero serving settings from the Hero RL launcher: FlashInfer autotune's dummy runs fail on Grug short
        # convolutions, and the Triton MoE backend is the qualified one.
        engine_init_kwargs={"enable_flashinfer_autotune": False, "kernel_config": {"moe_backend": "triton"}},
    )
    engines = create_route_engines(config)
    try:
        scores = await response_logprobs(engines, batch, args.model)
    finally:
        await asyncio.gather(*(close_owned_inference_engine(engine) for engine in engines))
    mask = batch["response_mask"].bool()
    logger.info("vLLM reference: %d response tokens, mean NLL %.4f", int(mask.sum()), float(-scores[mask].mean()))
    safetensors.torch.save_file({"response_logprobs": scores.contiguous()}, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", required=True, help="ensemble batch directory")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-parallel-size", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--output", required=True, help="local safetensors path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    ray.init()
    try:
        asyncio.run(run(args))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
