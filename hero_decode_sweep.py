"""Hero serving sweep: one data/expert-parallel engine, fixed-length request waves, engine counters per point.

Each point submits concurrency x serving GPUs requests at once, all with the same prompt length and exactly
--decode-length generated tokens. The driver reads every engine's vLLM counters every --sample-seconds, so the
report separates the prefill phase (prompt counter still rising) from the pure decode phase that follows.
The engine uses the hero RL serving settings from hero_async.py with dummy weights, as the RL serving engine
starts before its first weight sync.
"""

import argparse
import asyncio
import dataclasses
import json
import os
import random
import time
import traceback

import ray
from omegaconf import OmegaConf, open_dict
from skyrl_train.config.utils import get_default_config

from hero_qualification import s3_client, s3_location

# Fallback per-GPU concurrency limits from the KV-capacity estimate when the engine reports no cache usage.
ESTIMATED_MAX = {8192: 24, 16384: 15, 32768: 9}


def config(args):
    serving_world = args.serving_nodes * 8
    cfg = get_default_config()
    cfg.trainer.policy.model.path = args.model
    cfg.trainer.policy.model.tokenizer_path = args.model
    cfg.trainer.critic.model.path = None
    cfg.trainer.strategy = "megatron"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.logger = "console"
    cfg.trainer.distributed.placement_group_timeout_seconds = 1800
    cfg.generator.backend = "vllm"
    cfg.generator.run_engines_locally = True
    cfg.generator.enable_http_endpoint = False
    cfg.generator.weight_sync_backend = "nccl"
    cfg.generator.sampling_params.max_generate_length = args.decode_length
    cfg.generator.sampling_params.temperature = 1.0
    cfg.generator.sampling_params.top_p = 1.0
    cfg.generator.sampling_params.top_k = -1
    cfg.generator.sampling_params.logprobs = 0
    cfg.generator.sampling_params.stop = None
    with open_dict(cfg.generator.sampling_params):
        # Every request generates exactly decode_length tokens.
        cfg.generator.sampling_params.min_tokens = args.decode_length
        cfg.generator.sampling_params.stop_token_ids = []
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.inference_engine_data_parallel_size = serving_world
    cfg.generator.inference_engine_expert_parallel_size = serving_world
    cfg.generator.inference_engine_pipeline_parallel_size = 1
    cfg.generator.num_inference_engines = 1
    cfg.generator.max_num_batched_tokens = args.max_num_batched_tokens
    cfg.generator.max_num_seqs = args.max_num_seqs
    cfg.generator.gpu_memory_utilization = args.gpu_memory_utilization
    cfg.generator.enable_prefix_caching = False
    cfg.generator.enforce_eager = os.environ.get("HERO_SERVING_EAGER", "0") == "1"
    cfg.generator.engine_init_timeout_seconds = 4200
    cfg.generator.engine_init_kwargs = {
        "load_format": "dummy",
        "enable_return_routed_experts": args.route_capture,
        "max_model_len": args.max_model_len,
        "enable_flashinfer_autotune": False,
        "kernel_config": {"moe_backend": "triton"},
        "hf_overrides": {"max_seq_len": args.max_model_len, "max_position_embeddings": args.max_model_len},
    }
    return cfg


def save_report(output, report):
    bucket, prefix = s3_location(output)
    s3_client().put_object(Bucket=bucket, Key=prefix + "/report.json", Body=json.dumps(report, default=str).encode())


def engine_rows(snapshot, histograms=False):
    """Per-engine current and cumulative counters; histogram sums only when asked."""
    rows = []
    for engine in snapshot.engines:
        row = {"engine": engine.engine_id, "t": engine.timestamp,
               "current": dataclasses.asdict(engine.current), "cumulative": dataclasses.asdict(engine.cumulative)}
        if histograms:
            row["histograms"] = [
                {"name": h.name, "attributes": dict(h.attributes), "count": h.count, "total": h.total}
                for h in engine.histograms
            ]
        rows.append(row)
    return rows


def totals(rows):
    return {
        "prompt_tokens": sum(r["cumulative"]["prompt_tokens"] for r in rows),
        "generation_tokens": sum(r["cumulative"]["generation_tokens"] for r in rows),
        "preemptions": sum(r["cumulative"]["preemptions"] for r in rows),
        "running": sum(r["current"]["running_requests"] for r in rows),
        "waiting": sum(r["current"]["waiting_capacity"] + r["current"]["waiting_deferred"] for r in rows),
        "kv_usage_max": max((r["current"]["kv_cache_usage"] for r in rows), default=0.0),
        "kv_usage_mean": sum(r["current"]["kv_cache_usage"] for r in rows) / max(len(rows), 1),
        "engines": len(rows),
    }


def summarize(samples, start, gpus, requests, prompt_length, decode_length):
    """Prefill phase ends when the prompt counter reaches the wave's prompt tokens; decode is measured after it."""
    base = samples[0]["totals"]
    wanted = requests * prompt_length
    prefill_end = next((s["t"] for s in samples if s["totals"]["prompt_tokens"] - base["prompt_tokens"] >= 0.999 * wanted), None)
    out = {"prompt_tokens_counted": samples[-1]["totals"]["prompt_tokens"] - base["prompt_tokens"],
           "generation_tokens_counted": samples[-1]["totals"]["generation_tokens"] - base["generation_tokens"],
           "preemptions": samples[-1]["totals"]["preemptions"] - base["preemptions"],
           "kv_usage_max": max(s["totals"]["kv_usage_max"] for s in samples),
           "prefill_end_seconds": None if prefill_end is None else prefill_end - start}
    if prefill_end is None:
        return out
    out["prefill_tokens_per_second_per_gpu"] = wanted / max(prefill_end - start, 1e-9) / gpus
    decode = [s for s in samples if s["t"] >= prefill_end and s["totals"]["running"] > 0]
    if len(decode) >= 5:
        span = decode[-1]["t"] - decode[0]["t"]
        window = [s for s in decode if decode[0]["t"] + 0.2 * span <= s["t"] <= decode[0]["t"] + 0.8 * span]
        if len(window) >= 2:
            dt = window[-1]["t"] - window[0]["t"]
            dg = window[-1]["totals"]["generation_tokens"] - window[0]["totals"]["generation_tokens"]
            running = sum(s["totals"]["running"] for s in window) / len(window)
            out.update(decode_window_seconds=dt, decode_tokens_per_second_per_gpu=dg / dt / gpus,
                       decode_running_per_gpu=running / gpus,
                       decode_step_ms=1000.0 * (running / gpus) / max(dg / dt / gpus, 1e-9),
                       decode_mean_context=prompt_length + decode_length / 2)
    return out


async def run_point(client, sampling, gpus, prompt_length, per_gpu, decode_length, sample_seconds, timeout, seed):
    from skyrl_train.inference_engines.vllm.stats import IntervalReadMode

    requests = per_gpu * gpus
    rng = random.Random(seed)
    # Token ids from the ordinary vocabulary range; the content does not matter for timing.
    prompts = [[rng.randrange(1000, 120000) for _ in range(prompt_length)] for _ in range(requests)]
    samples = []
    first = await client.get_stats(IntervalReadMode.PEEK)
    start = time.time()
    samples.append({"t": start, "totals": totals(engine_rows(first))})
    task = asyncio.create_task(client.generate({"prompt_token_ids": prompts, "sampling_params": sampling}))
    while not task.done():
        if time.time() - start > timeout:
            task.cancel()
            raise TimeoutError(f"point exceeded {timeout} s")
        await asyncio.wait([task], timeout=sample_seconds)
        rows = engine_rows(await client.get_stats(IntervalReadMode.PEEK))
        samples.append({"t": time.time(), "totals": totals(rows)})
        print("HERO_SWEEP_SAMPLE " + json.dumps({"L": prompt_length, "c": per_gpu, "s": round(samples[-1]["t"] - start, 1),
                                                  **samples[-1]["totals"]}), flush=True)
    output = task.result()
    finished = time.time()
    last = await client.get_stats(IntervalReadMode.PEEK)
    lengths = [len(r) for r in output["response_ids"]]
    point = {
        "prompt_length": prompt_length, "per_gpu": per_gpu, "requests": requests, "decode_length": decode_length,
        "wall_seconds": finished - start, "response_lengths": {"min": min(lengths), "max": max(lengths)},
        "stop_reasons": sorted(set(output["stop_reasons"])),
        "end_engines": engine_rows(last, histograms=True), "start_engines": engine_rows(first, histograms=True),
        "samples": samples,
    }
    point["summary"] = summarize(samples, start, gpus, requests, prompt_length, decode_length)
    return point


@ray.remote(num_cpus=1, max_retries=0)
def sweep(cfg, args):
    from skyrl_train.entrypoints.main_base import create_ray_wrapped_inference_engines_from_config
    from skyrl_train.config.trajectory_runner_capabilities import EntrypointOperation
    from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
    from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
    from skyrl_train.tokenizer import tokenizer_from_config

    gpus = args.serving_nodes * 8
    report = {"status": "starting", "args": vars(args), "config": OmegaConf.to_container(cfg, resolve=True),
              "vllm_batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT") == "1", "points": [], "errors": []}
    save_report(args.output, report)
    tokenizer = tokenizer_from_config(cfg)
    started = time.monotonic()
    engines = create_ray_wrapped_inference_engines_from_config(cfg, None, tokenizer, operation=EntrypointOperation.GENERATE)
    client = InferenceEngineClient(engines, tokenizer, cfg)
    report.update(status="running", engine_init_seconds=time.monotonic() - started, engine_count=len(engines))
    report["kernel_runtime"] = ray.get([engine.inference_engine_actor.report_engine_kernel_runtime.remote() for engine in engines[:1]])
    save_report(args.output, report)
    print("HERO_SWEEP_ENGINE " + json.dumps({"init_seconds": report["engine_init_seconds"], "engines": len(engines)}), flush=True)
    sampling = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)

    async def main():
        warm = await run_point(client, {**sampling, "max_tokens": 32, "min_tokens": 32}, gpus, 1024, 1, 32,
                               args.sample_seconds, args.point_timeout, seed=0)
        report["warmup_seconds"] = warm["wall_seconds"]
        for prompt_length in args.lengths:
            cap = None
            for per_gpu in args.concurrencies:
                if per_gpu == 0:
                    if cap is None:
                        continue
                    per_gpu = cap
                if cap is not None and per_gpu > cap:
                    report["errors"].append({"prompt_length": prompt_length, "per_gpu": per_gpu, "skipped": f"above estimated limit {cap}"})
                    continue
                if any(p["prompt_length"] == prompt_length and p["per_gpu"] == per_gpu for p in report["points"]):
                    continue
                try:
                    point = await run_point(client, sampling, gpus, prompt_length, per_gpu, args.decode_length,
                                            args.sample_seconds, args.point_timeout, seed=prompt_length * 1000 + per_gpu)
                except Exception:
                    report["errors"].append({"prompt_length": prompt_length, "per_gpu": per_gpu, "error": traceback.format_exc()})
                    save_report(args.output, report)
                    print("HERO_SWEEP_ERROR " + json.dumps(report["errors"][-1])[:2000], flush=True)
                    continue
                report["points"].append(point)
                save_report(args.output, report)
                print("HERO_SWEEP_POINT " + json.dumps({"L": prompt_length, "c": per_gpu, "wall": point["wall_seconds"],
                                                        **point["summary"]}), flush=True)
                if per_gpu == 1 and cap is None:
                    usage = point["summary"]["kv_usage_max"]
                    # One sequence per GPU at full length uses `usage` of the cache; keep 10% headroom.
                    cap = min(args.max_num_seqs, int(0.9 / usage)) if usage > 0 else ESTIMATED_MAX.get(prompt_length, 8)
                    report.setdefault("per_gpu_limits", {})[str(prompt_length)] = {"limit": cap, "kv_usage_one_per_gpu": usage}
                    save_report(args.output, report)

    asyncio.run(main())
    report["status"] = "passed" if report["points"] else "failed"
    report["finished_at"] = time.time()
    save_report(args.output, report)
    return report["status"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="/tmp/hero-model")
    parser.add_argument("--serving-nodes", type=int, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[8192, 16384, 32768])
    # Per-GPU concurrencies in order; 0 means the largest that fits, measured from the one-per-GPU point.
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4, 0, 2, 8])
    parser.add_argument("--decode-length", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=34816)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--route-capture", action="store_true")
    parser.add_argument("--sample-seconds", type=float, default=2.0)
    parser.add_argument("--point-timeout", type=float, default=2400)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = config(args)
    if args.preflight:
        print(OmegaConf.to_yaml(cfg.generator))
        return
    from skyrl_train.utils.utils import initialize_ray

    initialize_ray(cfg)
    status = ray.get(sweep.options(runtime_env={"env_vars": {"VLLM_BATCH_INVARIANT": "1"}}).remote(cfg, args))
    print("HERO_SWEEP_STATUS " + status, flush=True)
    raise SystemExit(0 if status == "passed" else 1)


if __name__ == "__main__":
    main()
