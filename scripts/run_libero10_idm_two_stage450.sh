#!/usr/bin/env bash
set -euo pipefail
cd /data/pxchen/PACT-WAM
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export PYTHONPATH=/data/pxchen/PACT-WAM
PY=/data/pxchen/envs/jepa-wam/bin/python
OUT=artifacts/libero10_idm_pretrain_then_mlp450_20261004
mkdir -p "$OUT"
exec 9>"$OUT/launcher.lock"
flock -n 9 || exit 1
echo "$$" > "$OUT/launcher.pid"
trap 's=$?; echo "TWO_STAGE_EXIT status=$s time=$(date -Is)"; exit "$s"' EXIT
"$PY" scripts/train_libero10_idm_two_stage450.py --phase smoke
"$PY" scripts/train_libero10_idm_two_stage450.py --phase stage1
"$PY" scripts/train_libero10_idm_two_stage450.py --phase stage2
