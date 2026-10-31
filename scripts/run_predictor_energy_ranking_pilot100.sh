#!/usr/bin/env bash
set -euo pipefail
cd /data/pxchen/PACT-WAM
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH=/data/pxchen/PACT-WAM/src:/data/pxchen/libero-suite/LIBERO:/data/pxchen/PACT-WAM
export LIBERO_CONFIG_PATH=/data/pxchen/PACT-WAM/.libero_official_config
PY=/data/pxchen/envs/jepa-wam/bin/python
mkdir -p artifacts/libero10_predictor_energy_ranking_pilot100_20260930
$PY scripts/train_predictor_energy_ranking_pilot.py --phase cache-real
$PY scripts/train_predictor_energy_ranking_pilot.py --phase cache-wam
rm -rf artifacts/libero10_predictor_energy_ranking_pilot100_20260930/smoke
$PY scripts/train_predictor_energy_ranking_pilot.py --phase smoke
$PY scripts/train_predictor_energy_ranking_pilot.py --phase train
