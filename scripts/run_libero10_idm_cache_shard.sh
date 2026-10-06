#!/usr/bin/env bash
set -euo pipefail
cd /data/pxchen/PACT-WAM
source scripts/local_libero_env.sh
SHARD="$1"
GPU="$2"
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_EGL_DEVICE_ID="$GPU"
export PYTHONPATH=/data/pxchen/PACT-WAM/src:/data/pxchen/libero-suite/LIBERO:/data/pxchen/PACT-WAM
export LIBERO_CONFIG_PATH=/data/pxchen/PACT-WAM/.libero_official_config
export OMP_NUM_THREADS=4
PY=/data/pxchen/envs/jepa-wam/bin/python
OUT=artifacts/libero10_idm_cache_stride32_20261004
mkdir -p "$OUT"
exec 9>"$OUT/shard${SHARD}.lock"
flock -n 9 || exit 1
echo "$$" > "$OUT/shard${SHARD}.pid"
trap 's=$?; echo "SHARD_EXIT status=$s time=$(date -Is)"; exit "$s"' EXIT
# End-to-end smoke before generating the full shard.
"$PY" scripts/cache_libero10_idm_tokens.py --phase wam --shard "$SHARD" --limit 2
"$PY" scripts/cache_libero10_idm_tokens.py --phase tokens --shard "$SHARD" --limit 2
echo "END_TO_END_SMOKE_PASSED shard=$SHARD"
"$PY" scripts/cache_libero10_idm_tokens.py --phase wam --shard "$SHARD"
"$PY" scripts/cache_libero10_idm_tokens.py --phase tokens --shard "$SHARD"
