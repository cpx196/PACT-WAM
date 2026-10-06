#!/usr/bin/env bash
set -euo pipefail
cd /data/pxchen/PACT-WAM
source scripts/local_libero_env.sh
export CUDA_VISIBLE_DEVICES=4 MUJOCO_EGL_DEVICE_ID=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
out=artifacts/libero10_guidance_backward_smoke_20260929
exec 9> "$out/three_step.lock"
flock -n 9 || exit 3
used=$(nvidia-smi --id=4 --query-gpu=memory.used --format=csv,noheader,nounits)
if [[ "$used" -gt 1024 ]]; then echo "GPU4 occupied: $used MiB"; exit 4; fi
echo $$ > "$out/three_step_launcher.pid"
if [[ ! -s "$out/joint_encoded_windows.pt" ]]; then
    "$FASTWAM_PYTHON" scripts/smoke_guidance_three_step_joint.py --phase encode
fi
"$FASTWAM_PYTHON" scripts/smoke_guidance_three_step_joint.py --phase joint
echo "THREE_STEP_SMOKE_COMPLETED $(date '+%F %T')"
