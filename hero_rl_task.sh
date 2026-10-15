#!/usr/bin/env bash
# Two-step hero RL timing run: hero_te219_task.sh settings with batch, checkpoint interval and readback set
# by the environment, no evaluation and no runtime-bundle identity check.
set -euo pipefail

root="$PWD"
bash cloud/iris/bootstrap_runtime.sh "$root" /tmp/hero-venv /tmp/hero-runtime.env megatron development
source /tmp/hero-runtime.env
export PYTHONPATH="$root/skyrl-train:$root/skyrl-gym:$root"
export SKYRL_HOME="$root"
export HERO_SOURCE_REVISION HERO_RUNTIME_BUNDLE_SHA256=unverified HERO_SOURCE_MANIFEST_SHA256=unverified
export HERO_VLLM_REVISION="$(/tmp/hero-venv/bin/python -c 'import importlib.metadata as m; print(m.version("vllm"))')"
export HERO_VLLM_WHEEL_SHA256=unverified

started=$(date +%s)
/tmp/hero-venv/bin/python hero_qualification.py --metadata-only --source "$HERO_SOURCE" \
  --tokenizer marin-community/marin-tokenizer --tokenizer-revision a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2
/tmp/hero-venv/bin/python - <<'PY'
import hashlib
import os
from pathlib import Path
from hero_qualification import s3_client, s3_location

bucket, key = s3_location(os.environ["HERO_ASYNC_DATA_URI"])
data = s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
assert hashlib.sha256(data).hexdigest() == os.environ["HERO_ASYNC_DATA_SHA256"]
Path("/tmp/hero-async-train.jsonl").write_bytes(data)
PY
echo "HERO_RL_STAGE metadata_and_data_seconds=$(( $(date +%s) - started ))"

verify=()
if [[ "${HERO_EXPERT_BLOCK_VERIFY:-0}" == 1 ]]; then verify=(--expert-block-verify); fi
exec /tmp/hero-venv/bin/python hero_task_runtime.py \
  --rendezvous-dir "$HERO_OUTPUT/rendezvous" --ray-log-dir "$HERO_OUTPUT/ray-logs" \
  --rendezvous-timeout 7200 --cluster-join-timeout 7200 --driver-liveness-timeout 3600 \
  -- /tmp/hero-venv/bin/python hero_async.py --source "$HERO_SOURCE" \
  --data /tmp/hero-async-train.jsonl --model-family hero \
  --prompt-length "$HERO_PROMPT_LENGTH" --response-length "$HERO_RESPONSE_LENGTH" \
  --output "$HERO_OUTPUT" --name "$HERO_RUN_NAME" --variant "$HERO_VARIANT" \
  --policy-nodes "$HERO_POLICY_NODES" --serving-nodes "$HERO_SERVING_NODES" \
  --pp "$HERO_PP" --ep "$HERO_EP" --cp "$HERO_CP" --batch "$HERO_BATCH" --epochs 1 \
  --max-steps "$HERO_MAX_STEPS" --checkpoint-interval "$HERO_CHECKPOINT_INTERVAL" \
  --eval-interval 0 --generation-workers 4 --max-staleness-steps 1 --max-buffered-groups "$HERO_BATCH" \
  --optimizer MuonH --learning-rate 1e-6 --adam-learning-rate 1e-6 --optimizer-offload-fraction 0 \
  --serving-memory-utilization 0.92 --serving-max-seqs 64 --serving-batched-tokens 512 \
  --weight-sync-transport expert_block "${verify[@]}"
