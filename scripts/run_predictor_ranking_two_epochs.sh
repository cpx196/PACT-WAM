#!/usr/bin/env bash
set -euo pipefail
cd /data/pxchen/PACT-WAM
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export PYTHONPATH=/data/pxchen/PACT-WAM/src:/data/pxchen/libero-suite/LIBERO:/data/pxchen/PACT-WAM
export LIBERO_CONFIG_PATH=/data/pxchen/PACT-WAM/.libero_official_config
PY=/data/pxchen/envs/jepa-wam/bin/python
OUT=artifacts/libero10_predictor_real_path_ranking_2epoch900_20260930
mkdir -p "$OUT"
exec 9>"$OUT/launcher.lock"
flock -n 9 || exit 1
echo "$$" > "$OUT/launcher.pid"
trap 's=$?; echo "LAUNCHER_EXIT status=$s time=$(date -Is)"; exit "$s"' EXIT
"$PY" scripts/train_predictor_ranking_two_epochs.py --phase cache-real
"$PY" scripts/train_predictor_ranking_two_epochs.py --phase cache-wam
"$PY" scripts/train_predictor_ranking_two_epochs.py --phase smoke
"$PY" scripts/train_predictor_ranking_two_epochs.py --phase train
