"""GPU smoke: run the route pass over an ensemble twice on one set of engines, write pass-1 routes, verify them."""

import argparse
import asyncio
import json
import logging
import time

import numpy as np
import ray
import torch

from skyrl_train.benchmarks.learner.batch_artifact import list_ensemble, read_ensemble_batch, write_routes
from skyrl_train.benchmarks.learner.route_pass import (
    RouteEngineConfig,
    batch_routes,
    create_route_engines,
    num_experts_from_config,
)
from skyrl_train.inference_engines.vllm_teacher_oracle import close_owned_inference_engine
from skyrl_train.io import io

logger = logging.getLogger("route_smoke")


async def run(args) -> dict:
    config = RouteEngineConfig(
        model=args.model,
        revision=None,
        model_source_uri=None,
        distributed_streaming=False,
        tokenizer=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        num_engines=1,
        data_parallel_size=args.data_parallel_size,
        max_model_len=4096,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enable_chunked_prefill=args.enable_chunked_prefill,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=args.enforce_eager,
        engine_init_timeout_seconds=args.engine_init_timeout_seconds,
        engine_init_kwargs=args.engine_init_kwargs,
    )
    num_experts = num_experts_from_config(config.model, None)
    batch_dirs = list_ensemble(args.ensemble_root, args.ensemble_version)
    summary: dict = {"batch_dirs": batch_dirs, "num_experts": num_experts, "gpus": args.data_parallel_size}
    started = time.perf_counter()
    engines = create_route_engines(config)
    summary["engine_start_seconds"] = time.perf_counter() - started
    logger.info("engines up in %.1fs", summary["engine_start_seconds"])
    try:
        passes = []
        # Pass 2 uses a different in-flight limit so prefill batches are composed differently.
        for pass_index, in_flight in enumerate((64, 4)):
            pass_started = time.perf_counter()
            routes = []
            for batch_dir in batch_dirs:
                batch, manifest = read_ensemble_batch(batch_dir, routes_checkpoint=None)
                batch_started = time.perf_counter()
                rows = await batch_routes(
                    engines,
                    batch,
                    served_model_name=config.model,
                    num_experts=num_experts,
                    max_in_flight_per_engine=in_flight,
                    request_id_prefix=f"p{pass_index}-{manifest.profile}-{manifest.batch_index}",
                )
                logger.info(
                    "pass %d %s: %d response tokens in %.1fs",
                    pass_index,
                    batch_dir,
                    manifest.response_tokens,
                    time.perf_counter() - batch_started,
                )
                routes.append(rows)
            passes.append({"in_flight": in_flight, "seconds": time.perf_counter() - pass_started, "routes": routes})
            if pass_index == 0:
                for batch_dir, rows in zip(batch_dirs, routes, strict=True):
                    write_routes(batch_dir, args.checkpoint_id, rows)
    finally:
        await asyncio.gather(*(close_owned_inference_engine(engine) for engine in engines))

    summary["pass_seconds"] = [p["seconds"] for p in passes]
    summary["pass_in_flight"] = [p["in_flight"] for p in passes]
    batches = []
    for batch_index, batch_dir in enumerate(batch_dirs):
        batch, manifest = read_ensemble_batch(batch_dir, routes_checkpoint=args.checkpoint_id)
        stored = batch.routed_expert_rows
        response_tokens = batch["response_mask"].sum(dim=1).tolist()
        flat = np.concatenate(stored.rows)
        first = passes[0]["routes"][batch_index].rows
        second = passes[1]["routes"][batch_index].rows
        a, b = np.concatenate(first), np.concatenate(second)
        same_set = np.sort(a, axis=-1) == np.sort(b, axis=-1)
        token_layer_set_equal = same_set.all(axis=-1)
        token_equal = token_layer_set_equal.all(axis=-1)
        sequence_equal = [bool(np.array_equal(np.sort(x, -1), np.sort(y, -1))) for x, y in zip(first, second)]
        batches.append(
            {
                "batch_dir": batch_dir,
                "rows_shape": list(flat.shape),
                "rows_dtype": str(flat.dtype),
                "torch_dtype": str(stored.dtype),
                "expert_min": int(flat.min()),
                "expert_max": int(flat.max()),
                "offsets_match_response_mask": [len(r) for r in stored.rows] == response_tokens,
                "response_tokens": int(sum(response_tokens)),
                "manifest_response_tokens": manifest.response_tokens,
                "distinct_experts_per_row_ok": bool(
                    (np.diff(np.sort(flat.astype(np.int64), axis=-1), axis=-1) > 0).all()
                ),
                "experts_used": int(len(np.unique(flat))),
                "rerun_element_mismatch_rate": float((a != b).mean()),
                "rerun_token_layer_set_mismatch_rate": float(1 - token_layer_set_equal.mean()),
                "rerun_token_mismatch_rate": float(1 - token_equal.mean()),
                "rerun_sequences_identical": int(sum(sequence_equal)),
                "num_sequences": len(first),
            }
        )
    summary["batches"] = batches
    summary["torch_int16"] = all(b["torch_dtype"] == str(torch.int16) for b in batches)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-root", required=True)
    parser.add_argument("--ensemble-version", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--data-parallel-size", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--engine-init-timeout-seconds", type=float, required=True)
    parser.add_argument("--engine-init-kwargs", type=json.loads, required=True)
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    ray.init()
    started = time.perf_counter()
    summary = asyncio.run(run(args))
    summary["total_seconds"] = time.perf_counter() - started
    payload = json.dumps(summary, indent=2)
    print("ROUTE_SMOKE_SUMMARY " + json.dumps(summary), flush=True)
    io.write_bytes_atomic(f"{args.output.rstrip('/')}/summary.json", payload.encode())


if __name__ == "__main__":
    main()
