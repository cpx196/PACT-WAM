#!/usr/bin/env bash
set -euo pipefail
cd /data/pxchen/PACT-WAM
source scripts/local_libero_env.sh
export CUDA_VISIBLE_DEVICES=4 MUJOCO_EGL_DEVICE_ID=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
out=artifacts/libero10_guidance_backward_smoke_20260929
exec 9> "$out/smoke.lock"
flock -n 9 || exit 3
used=$(nvidia-smi --id=4 --query-gpu=memory.used --format=csv,noheader,nounits)
if [[ "$used" -gt 1024 ]]; then echo "GPU4 occupied: $used MiB"; exit 4; fi
echo $$ > "$out/launcher.pid"
if [[ ! -s "$out/wam_windows.pt" ]]; then
    "$FASTWAM_PYTHON" scripts/smoke_guidance_predictor_backward.py --phase cache
fi
"$FASTWAM_PYTHON" scripts/smoke_guidance_predictor_backward.py --phase backward
echo "SMOKE_COMPLETED $(date '+%F %T')"
