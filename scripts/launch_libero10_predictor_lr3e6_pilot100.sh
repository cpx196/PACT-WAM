#!/usr/bin/env bash
set -euo pipefail
cd /data/pxchen/PACT-WAM
source scripts/local_libero_env.sh
gpu=${1:?GPU index required}
export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PACT_TRAIN_EPOCH=1
export PACT_OUTPUT_DIR=/data/pxchen/PACT-WAM/artifacts/libero10_predictor_action_lr3e6_pilot100_20260930
export PACT_CACHE_ROOT=/data/pxchen/PACT-WAM/artifacts/libero10_predictor_action_oneepoch_20260930
export PACT_LEARNING_RATE=3e-6
export PACT_TRAIN_STEPS=100
export PACT_PAIRED_EVAL=1
mkdir -p "$PACT_OUTPUT_DIR"
exec 9> "$PACT_OUTPUT_DIR/launcher.lock"
flock -n 9 || exit 3
used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
if [[ "$used" -gt 1024 ]]; then echo "GPU${gpu} occupied (${used} MiB)"; exit 4; fi
echo $$ > "$PACT_OUTPUT_DIR/launcher.pid"
echo "$gpu" > "$PACT_OUTPUT_DIR/gpu.txt"
trap 'code=$?; echo "$(date "+%F %T") PIPELINE_EXIT=$code" | tee -a "$PACT_OUTPUT_DIR/completion.log"' EXIT
echo "$(date '+%F %T') PHASE=prepare START"
"$FASTWAM_PYTHON" scripts/train_guidance_libero10_one_epoch.py --phase prepare
echo "$(date '+%F %T') PHASE=prepare PASSED"
echo "$(date '+%F %T') PHASE=train START"
"$FASTWAM_PYTHON" scripts/train_guidance_libero10_one_epoch.py --phase train
echo "$(date '+%F %T') PHASE=train PASSED"
