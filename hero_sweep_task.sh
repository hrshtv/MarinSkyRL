#!/usr/bin/env bash
set -euo pipefail

root="$PWD"
bash cloud/iris/bootstrap_runtime.sh "$root" /tmp/hero-venv /tmp/hero-runtime.env megatron development
source /tmp/hero-runtime.env
export PYTHONPATH="$root/skyrl-train:$root/skyrl-gym:$root"
export SKYRL_HOME="$root"

started=$(date +%s)
/tmp/hero-venv/bin/python hero_qualification.py --metadata-only --source "$HERO_SOURCE" --model /tmp/hero-model \
  --tokenizer marin-community/marin-tokenizer --tokenizer-revision a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2
echo "HERO_SWEEP_STAGE metadata_seconds=$(( $(date +%s) - started ))"

# shellcheck disable=SC2086 # HERO_SWEEP_ARGS carries several driver arguments.
exec /tmp/hero-venv/bin/python hero_task_runtime.py \
  --rendezvous-dir "$HERO_OUTPUT/rendezvous" --ray-log-dir "$HERO_OUTPUT/ray-logs" \
  --rendezvous-timeout 7200 --cluster-join-timeout 7200 --driver-liveness-timeout 3600 \
  -- /tmp/hero-venv/bin/python hero_decode_sweep.py --output "$HERO_OUTPUT" \
  --serving-nodes "$HERO_SERVING_NODES" $HERO_SWEEP_ARGS
