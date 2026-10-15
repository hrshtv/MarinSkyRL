#!/usr/bin/env bash
# Learner-benchmark smoke on an Iris H100 gang of LB_NODES nodes: every node bootstraps the runtime and stages the
# (shallow) Hero checkpoint locally; rank 0 then runs smoke_driver.sh on a Ray cluster spanning the gang.
# Configured by LB_* environment variables (see submit_smoke.sh).
set -euo pipefail

root="$PWD"
bash cloud/iris/bootstrap_runtime.sh "$root" /tmp/lb-venv /tmp/lb-runtime.env megatron development
source /tmp/lb-runtime.env
export PYTHONPATH="$root/skyrl-train:$root/skyrl-gym:$root"
export SKYRL_HOME="$root"
py=/tmp/lb-venv/bin/python
nvidia-smi --query-gpu=index,name,memory.total --format=csv

export LB_MODEL_PATH=/tmp/lb-model
"$py" cloud/iris/learner_bench/stage_model.py --source "$LB_MODEL" --destination "$LB_MODEL_PATH" --workers 12

if [[ -z "${LB_ENSEMBLE_ROOT:-}" ]]; then
  export LB_ENSEMBLE_ROOT=/tmp/lb-ensemble
  "$py" cloud/iris/learner_bench/make_smoke_batches.py \
    --tokenizer marin-community/marin-tokenizer --tokenizer-revision a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2 \
    --output-root "$LB_ENSEMBLE_ROOT" --ensemble-version "$LB_ENSEMBLE_VERSION" --num-batches 2 \
    --rows-per-batch "$LB_ROWS"
fi

exec "$py" cloud/iris/learner_bench/ray_gang.py --rendezvous-dir "$LB_OUTPUT/rendezvous" \
  --ray-log-dir "$LB_OUTPUT/ray-logs" --rendezvous-timeout 3600 --cluster-join-timeout 3600 \
  --driver-liveness-timeout 3600 -- bash "cloud/iris/learner_bench/${LB_DRIVER_SCRIPT:-smoke_driver.sh}"
