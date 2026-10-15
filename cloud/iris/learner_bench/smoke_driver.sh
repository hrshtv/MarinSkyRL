#!/usr/bin/env bash
# Rank-0 driver of the learner-benchmark smoke (run by ray_gang.py): optional vLLM-vs-Megatron log-prob check on the
# first ensemble batch, then the learner benchmark with native routing.
set -euo pipefail

py=/tmp/lb-venv/bin/python

driver_args=(
  --run-name "$LB_RUN_NAME" --harness-revision "$LB_REVISION" --model-path "$LB_MODEL_PATH"
  --tp "$LB_TP" --pp "$LB_PP" --ep "$LB_EP" --cp 1 --etp 1 --num-nodes "$LB_NODES" --gpus-per-node 8
  --optimizer MuonH --lr 1e-6 --adam-lr 1e-6 --offload-momentum
  --recompute full_layers --micro-batch-size-per-gpu "$LB_MICRO" --sample-packing --no-flash-attn
  --ensemble-root "$LB_ENSEMBLE_ROOT" --ensemble-version "$LB_ENSEMBLE_VERSION" --profiles short
  --warmup-updates "$LB_WARMUP" --measured-updates "$LB_MEASURED" --score-old-logprobs
  --override trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32=true
  --override trainer.policy.megatron_config.ddp_config.overlap_grad_reduce=true
  --override trainer.policy.megatron_config.ddp_config.overlap_param_gather=false
)

if [[ "$LB_REFERENCE_CHECK" == 1 ]]; then
  first_batch="$("$py" -c "import sys; from skyrl_train.benchmarks.learner.batch_artifact import list_ensemble; print(list_ensemble(sys.argv[1], sys.argv[2])[0])" "$LB_ENSEMBLE_ROOT" "$LB_ENSEMBLE_VERSION")"
  "$py" cloud/iris/learner_bench/vllm_reference_logprobs.py --batch "$first_batch" --model "$LB_MODEL_PATH" \
    --data-parallel-size 8 --max-model-len 4096 --output /tmp/lb-vllm-reference.safetensors
  "$py" cloud/iris/learner_bench/megatron_reference_check.py --reference /tmp/lb-vllm-reference.safetensors \
    --report /tmp/lb-reference-check.json "${driver_args[@]}" --output "$LB_OUTPUT/unused.json"
  "$py" -c "import sys; from skyrl_train.io import io; io.write_bytes_atomic(sys.argv[2], open(sys.argv[1], 'rb').read())" \
    /tmp/lb-reference-check.json "$LB_OUTPUT/reference_check.json"
fi

# Hero learner memory is tight on H100; expandable segments avoid fragmentation OOMs in the optimizer step.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$py" -m skyrl_train.benchmarks.learner.driver "${driver_args[@]}" --output "$LB_OUTPUT/result.json"
