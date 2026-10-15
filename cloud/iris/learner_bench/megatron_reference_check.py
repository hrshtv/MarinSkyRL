"""Compare the Megatron learner's response log-probs on one ensemble batch with a vLLM reference.

Takes the learner driver's own arguments (so the policy worker is built exactly as the benchmark builds it) plus
``--reference`` (the output of ``vllm_reference_logprobs.py``) and ``--report``. Prints and writes a JSON report
with the per-token absolute difference, both models' mean NLL, and the layer schedule (global vs sliding) the
truncated config implies next to the full model's.
"""

import argparse
import json
import sys

import ray
import safetensors.torch
import torch
from transformers import AutoConfig

from skyrl_train.benchmarks.learner import driver
from skyrl_train.benchmarks.learner.batch_artifact import read_ensemble_batch
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.models.grug_moe import grug_long_layer_flags
from skyrl_train.training_batch import pad_training_input
from skyrl_train.utils.utils import initialize_ray


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", required=True)
    parser.add_argument("--report", required=True)
    args, driver_argv = parser.parse_known_args()
    spec = driver.parse_spec(driver_argv)

    entries = driver.ensemble_entries(spec)
    entry = min(entries, key=lambda e: (e.profile, e.batch_index))
    rows = driver.rows_per_update([entry], spec.layout, spec.micro_batch_size_per_gpu)
    cfg = driver.build_policy_cfg(spec, rows, driver.compose_base_config())
    model_config = AutoConfig.from_pretrained(spec.model_path)
    full_layers = driver.source_num_layers(spec.model_path, model_config.num_hidden_layers)
    truncated_flags = grug_long_layer_flags(model_config.num_hidden_layers, model_config.global_every)
    full_flags = grug_long_layer_flags(full_layers, model_config.global_every)[: model_config.num_hidden_layers]

    batch, _ = read_ensemble_batch(entry.path, routes_checkpoint=None)
    reference = safetensors.torch.load_file(args.reference)["response_logprobs"]
    padded = pad_training_input(batch, spec.layout.data_parallel * spec.micro_batch_size_per_gpu)
    initialize_ray(cfg)
    try:
        policy = driver._init_policy_group(cfg, spec.layout)
        outputs = ray.get(policy.async_run_ray_method("mesh", "forward", data=padded))
        megatron = concatenate_outputs_after_mesh_dispatch(policy.actor_infos, outputs)["output"].float()
    finally:
        ray.shutdown()

    megatron = megatron[: batch.batch_size]
    mask = batch["response_mask"].bool()
    actual, expected = megatron[mask], reference[mask]
    diff = (actual - expected).abs()
    report = {
        "batch": entry.path,
        "response_tokens": int(mask.sum()),
        "megatron_finite": bool(torch.isfinite(actual).all()),
        "megatron_mean_nll": float(-actual.mean()),
        "vllm_mean_nll": float(-expected.mean()),
        "abs_diff_max": float(diff.max()),
        "abs_diff_mean": float(diff.mean()),
        "abs_diff_p99": float(torch.quantile(diff, 0.99)),
        "pearson": float(torch.corrcoef(torch.stack([actual, expected]))[0, 1]),
        "layers": model_config.num_hidden_layers,
        "source_layers": full_layers,
        "global_layers_truncated": [index for index, flag in enumerate(truncated_flags) if flag],
        "global_layers_full_prefix": [index for index, flag in enumerate(full_flags) if flag],
        "layer_schedule_matches_full_prefix": truncated_flags == full_flags,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text, flush=True)
    with open(args.report, "w") as handle:
        handle.write(text + "\n")
    if not report["megatron_finite"] or not report["layer_schedule_matches_full_prefix"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
