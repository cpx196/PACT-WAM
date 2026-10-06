#!/usr/bin/env bash
set -euo pipefail
cd /data/pxchen/PACT-WAM
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH=/data/pxchen/PACT-WAM/src:/data/pxchen/libero-suite/LIBERO:/data/pxchen/PACT-WAM
export LIBERO_CONFIG_PATH=/data/pxchen/PACT-WAM/.libero_official_config
export OMP_NUM_THREADS=4
PY=/data/pxchen/envs/jepa-wam/bin/python
OUT=artifacts/libero10_mlp_residual_pilot_20261004
mkdir -p "$OUT"
exec 9>"$OUT/launcher.lock"
flock -n 9 || exit 1
echo "$$" > "$OUT/launcher.pid"
trap 's=$?; echo "LAUNCHER_EXIT status=$s time=$(date -Is)"; exit "$s"' EXIT
"$PY" scripts/train_libero10_residual_mlp.py --phase prepare
"$PY" scripts/train_libero10_residual_mlp.py --phase smoke
"$PY" scripts/train_libero10_residual_mlp.py --phase train
