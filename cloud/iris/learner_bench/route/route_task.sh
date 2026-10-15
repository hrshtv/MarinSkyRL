#!/usr/bin/env bash
set -euo pipefail
root="$PWD"
model_dir=/tmp/hero-model
cleanup() { rm -rf "$model_dir"; echo "STAGE_CLEANUP node=$(hostname) removed=$model_dir"; }
trap cleanup EXIT
trap 'exit 143' TERM INT
bash cloud/iris/bootstrap_runtime.sh "$root" /tmp/route-venv /tmp/route-runtime.env megatron development
source /tmp/route-runtime.env
export PYTHONPATH="$root/skyrl-train:$root/skyrl-gym:$root"
export SKYRL_HOME="$root"
echo "NODE task=${IRIS_TASK_ID:-?} host=$(hostname) ip=${IRIS_ADVERTISE_HOST:-?}"
df -B1 /tmp
/tmp/route-venv/bin/python cloud/iris/learner_bench/route/stage_model.py --source "$ROUTE_MODEL_SOURCE" --dest "$model_dir" \
  --min-free-bytes 1200000000000 --parallel-files 8 --parts-per-file 4
/tmp/route-venv/bin/python cloud/iris/learner_bench/route/task_runtime_wrapper.py \
  --rendezvous-dir "$ROUTE_OUTPUT/rendezvous" --ray-log-dir "$ROUTE_OUTPUT/ray-logs" \
  -- /tmp/route-venv/bin/python cloud/iris/learner_bench/route/route_smoke.py \
  --ensemble-root "$ROUTE_ENSEMBLE_ROOT" --ensemble-version v0 --model "$model_dir" \
  --tokenizer marin-community/marin-tokenizer --tokenizer-revision a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2 \
  --checkpoint-id "$ROUTE_CHECKPOINT_ID" --data-parallel-size "$ROUTE_DP" --output "$ROUTE_OUTPUT" \
  --max-num-batched-tokens 512 --max-num-seqs 16 --enable-chunked-prefill \
  --gpu-memory-utilization 0.9 --no-enforce-eager --engine-init-timeout-seconds 4200 \
  --engine-init-kwargs '{"enable_flashinfer_autotune": false, "kernel_config": {"moe_backend": "triton"}}'
