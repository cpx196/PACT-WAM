#!/usr/bin/env bash
set -euo pipefail
cd /data/pxchen/PACT-WAM
source scripts/local_libero_env.sh
export CUDA_VISIBLE_DEVICES=4 MUJOCO_EGL_DEVICE_ID=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
out=artifacts/libero10_predictor_action_oneepoch_20260930
mkdir -p "$out"
exec 9> "$out/launcher.lock"
flock -n 9 || exit 3
used=$(nvidia-smi --id=4 --query-gpu=memory.used --format=csv,noheader,nounits)
if [[ "$used" -gt 1024 ]]; then echo "GPU4 occupied ($used MiB)"; exit 4; fi
echo $$ > "$out/launcher.pid"
trap 'code=$?; echo "$(date "+%F %T") PIPELINE_EXIT=$code" | tee -a "$out/completion.log"' EXIT
run() {
    echo "$(date '+%F %T') PHASE=$1 START"
    "$FASTWAM_PYTHON" scripts/train_guidance_libero10_one_epoch.py --phase "$1" "${@:2}"
    echo "$(date '+%F %T') PHASE=$1 PASSED"
}
run prepare
run cache-wam --limit 3
run cache-encoder --limit 3
run smoke
run cache-wam
run cache-encoder
run train
