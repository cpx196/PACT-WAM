#!/usr/bin/env bash
set -euo pipefail
cd /data/pxchen/PACT-WAM
out=artifacts/libero10_predictor_action_lr3e6_pilot100_20260930
mkdir -p "$out"
exec 8> "$out/watcher.lock"
flock -n 8 || exit 3
echo "$(date '+%F %T') WATCHER_START" 
while true; do
    for gpu in 0 1 2 3 4 5 6 7; do
        used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
        if [[ "$used" -le 1024 ]]; then
            echo "$(date '+%F %T') GPU${gpu}_IDLE launching"
            exec bash scripts/launch_libero10_predictor_lr3e6_pilot100.sh "$gpu"
        fi
    done
    sleep 30
done
