#!/usr/bin/env bash
# Rank-0 driver (run by ray_gang.py): replay-vs-native learner benchmark on one layout, then a numerics
# calibration on two layouts, a held-out unperturbed layout and the must-fail controls. Needs LB_NODES=4.
set -euo pipefail

py=/tmp/lb-venv/bin/python
common=(
  --run-name "$LB_RUN_NAME" --harness-revision "$LB_REVISION" --model-path "$LB_MODEL_PATH"
  --cp 1 --etp 1 --num-nodes "$LB_NODES" --gpus-per-node 8
  --optimizer MuonH --lr 1e-6 --adam-lr 1e-6 --offload-momentum
  --recompute full_layers --micro-batch-size-per-gpu "$LB_MICRO" --sample-packing --no-flash-attn
  --ensemble-root "$LB_ENSEMBLE_ROOT" --ensemble-version "$LB_ENSEMBLE_VERSION" --profiles short
  --warmup-updates "$LB_WARMUP" --measured-updates "$LB_MEASURED" --score-old-logprobs
  --override trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32=true
  --override trainer.policy.megatron_config.ddp_config.overlap_grad_reduce=true
  --override trainer.policy.megatron_config.ddp_config.overlap_param_gather=false
)
replay=(--routes-checkpoint "$LB_ROUTES_CHECKPOINT")
bench=("$py" -m skyrl_train.benchmarks.learner.driver)
numerics=("$py" -m skyrl_train.benchmarks.learner.numerics)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Steps whose output already exists are skipped, so a rerun into the same LB_OUTPUT resumes.
exists() { "$py" -c "import sys; from skyrl_train.io import io; sys.exit(0 if io.exists(sys.argv[1]) else 1)" "$1"; }

exists "$LB_OUTPUT/result_replay.json" || "${bench[@]}" "${common[@]}" "${replay[@]}" --tp 1 --pp 2 --ep 16 \
  --memory-breakdown --output "$LB_OUTPUT/result_replay.json"
exists "$LB_OUTPUT/result_native.json" || "${bench[@]}" "${common[@]}" --tp 1 --pp 2 --ep 16 \
  --output "$LB_OUTPUT/result_native.json"

record() {
  local name="$1"
  shift
  exists "$LB_OUTPUT/numerics/$name.json" && return
  "${numerics[@]}" record --record-output "$LB_OUTPUT/numerics/$name.json" --num-updates 2 \
    "${common[@]}" "${replay[@]}" --tp 1 --output "$LB_OUTPUT/unused.json" "$@"
}
record pp2-ep16 --pp 2 --ep 16
record pp4-ep8 --pp 4 --ep 8
# Held out from calibration: the PP4/EP8 layout with two-row micro-batches (a different reduction order).
record pp4-ep8-mbs2 --pp 4 --ep 8 --micro-batch-size-per-gpu 2
record pp2-ep16-negate --control negate_advantages --pp 2 --ep 16
record pp2-ep16-double --control double_advantages --pp 2 --ep 16

"${numerics[@]}" calibrate --records "$LB_OUTPUT/numerics/pp2-ep16.json" "$LB_OUTPUT/numerics/pp4-ep8.json" \
  --reference "$LB_OUTPUT/numerics/reference.json"
for candidate in pp4-ep8-mbs2 pp2-ep16-negate pp2-ep16-double; do
  status=0
  "${numerics[@]}" compare --reference "$LB_OUTPUT/numerics/reference.json" \
    --record "$LB_OUTPUT/numerics/$candidate.json" --output "$LB_OUTPUT/numerics/compare-$candidate.json" || status=$?
  echo "numerics compare $candidate exit=$status"
done
