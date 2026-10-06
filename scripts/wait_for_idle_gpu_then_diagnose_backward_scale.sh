#!/usr/bin/env bash
set -euo pipefail
cd /data/pxchen/PACT-WAM
source scripts/local_libero_env.sh
out=artifacts/libero10_guidance_backward_scale_20260930
mkdir -p "$out"
exec 9> "$out/launcher.lock"
flock -n 9 || exit 3
echo "$(date '+%F %T') WATCHER_START"
while true; do
    for gpu in 0 1 2 3 4 5 6 7; do
        used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
        [[ "$used" -le 512 ]] || continue
        # Avoid claiming a card during a brief process restart or model reload.
        stable=1
        for _ in 1 2 3; do
            sleep 5
            used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
            if [[ "$used" -gt 512 ]]; then stable=0; break; fi
        done
        [[ "$stable" -eq 1 ]] || continue
        echo "$(date '+%F %T') GPU${gpu}_STABLY_IDLE launching"
        export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu"
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        echo "$gpu" > "$out/gpu.txt"
        set +e
        "$FASTWAM_PYTHON" scripts/diagnose_guidance_backward_scale.py
        code=$?
        set -e
        if [[ "$code" -eq 0 ]]; then
            echo "$(date '+%F %T') DIAGNOSTIC_EXIT=0"
            exit 0
        fi
        echo "$(date '+%F %T') DIAGNOSTIC_ATTEMPT_FAILED=$code; returning to wait"
        sleep 30
    done
    sleep 30
done
