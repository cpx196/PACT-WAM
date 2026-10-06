#!/usr/bin/env bash
set -euo pipefail
cd /data/pxchen/PACT-WAM
first=artifacts/libero10_predictor_action_oneepoch_20260930
second=artifacts/libero10_predictor_action_epoch2_20260930
mkdir -p "$second"
exec 8> "$second/watcher.lock"
flock -n 8 || exit 3
echo "$(date '+%F %T') WATCHER_START waiting for first epoch"
while ! grep -q 'PIPELINE_EXIT=0' "$first/completion.log" 2>/dev/null; do
    if grep -q 'PIPELINE_EXIT=[1-9]' "$first/completion.log" 2>/dev/null; then
        echo "$(date '+%F %T') FIRST_EPOCH_FAILED; second epoch not launched"
        exit 4
    fi
    sleep 30
done
test -s "$first/train_one_epoch/step_0450.pt"
echo "$(date '+%F %T') FIRST_EPOCH_COMPLETE waiting for GPU4"
until [[ $(nvidia-smi --id=4 --query-gpu=memory.used --format=csv,noheader,nounits) -le 1024 ]]; do
    sleep 30
done
echo "$(date '+%F %T') SECOND_EPOCH_LAUNCH"
exec bash scripts/launch_libero10_predictor_epoch2.sh
