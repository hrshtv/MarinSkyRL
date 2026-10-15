#!/usr/bin/env bash
# One-node learner-benchmark smoke: stage a (shallow) Hero checkpoint, check Megatron log-probs against vLLM,
# then run the learner driver with native routing. Configured by LB_* environment variables (see submit_smoke.py).
set -euo pipefail

root="$PWD"
bash cloud/iris/bootstrap_runtime.sh "$root" /tmp/lb-venv /tmp/lb-runtime.env megatron development
source /tmp/lb-runtime.env
export PYTHONPATH="$root/skyrl-train:$root/skyrl-gym:$root"
export SKYRL_HOME="$root"
py=/tmp/lb-venv/bin/python
tokenizer=(--tokenizer marin-community/marin-tokenizer --tokenizer-revision a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2)
model=/tmp/lb-model
nvidia-smi --query-gpu=index,name,memory.total --format=csv

"$py" cloud/iris/learner_bench/stage_model.py --source "$LB_MODEL" --destination "$model" --workers 12

ensemble_root="${LB_ENSEMBLE_ROOT:-}"
if [[ -z "$ensemble_root" ]]; then
  ensemble_root=/tmp/lb-ensemble
  "$py" cloud/iris/learner_bench/make_smoke_batches.py "${tokenizer[@]}" --output-root "$ensemble_root" \
    --ensemble-version "$LB_ENSEMBLE_VERSION" --num-batches 2 --rows-per-batch "$LB_ROWS"
fi
first_batch="$("$py" -c "import sys; from skyrl_train.benchmarks.learner.batch_artifact import list_ensemble; print(list_ensemble(sys.argv[1], sys.argv[2])[0])" "$ensemble_root" "$LB_ENSEMBLE_VERSION")"

driver_args=(
  --run-name "$LB_RUN_NAME" --harness-revision "$LB_REVISION" --model-path "$model"
  --tp 1 --pp 1 --ep "$LB_EP" --cp 1 --etp 1 --num-nodes 1 --gpus-per-node 8
  --optimizer MuonH --lr 1e-6 --adam-lr 1e-6 --offload-momentum
  --recompute full_layers --micro-batch-size-per-gpu "$LB_MICRO" --sample-packing --no-flash-attn
  --ensemble-root "$ensemble_root" --ensemble-version "$LB_ENSEMBLE_VERSION" --profiles short
  --warmup-updates "$LB_WARMUP" --measured-updates "$LB_MEASURED" --score-old-logprobs
  --override trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32=true
  --override trainer.policy.megatron_config.ddp_config.overlap_grad_reduce=true
  --override trainer.policy.megatron_config.ddp_config.overlap_param_gather=false
)

if [[ "${LB_REFERENCE_CHECK:-1}" == 1 ]]; then
  "$py" cloud/iris/learner_bench/vllm_reference_logprobs.py --batch "$first_batch" --model "$model" \
    --data-parallel-size 8 --max-model-len 4096 --output /tmp/lb-vllm-reference.safetensors
  "$py" cloud/iris/learner_bench/megatron_reference_check.py --reference /tmp/lb-vllm-reference.safetensors \
    --report /tmp/lb-reference-check.json "${driver_args[@]}" --output "$LB_OUTPUT/unused.json"
  "$py" -c "import sys; from skyrl_train.io import io; io.write_bytes_atomic(sys.argv[2], open(sys.argv[1], 'rb').read())" \
    /tmp/lb-reference-check.json "$LB_OUTPUT/reference_check.json"
fi

"$py" -m skyrl_train.benchmarks.learner.driver "${driver_args[@]}" --output "$LB_OUTPUT/result.json"
