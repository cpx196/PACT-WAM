#!/usr/bin/env bash
set -euo pipefail
cd /data/pxchen/PACT-WAM
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONPATH=/data/pxchen/PACT-WAM/src:/data/pxchen/libero-suite/LIBERO:/data/pxchen/PACT-WAM
export LIBERO_CONFIG_PATH=/data/pxchen/PACT-WAM/.libero_official_config
rm -rf artifacts/libero10_predictor_direction_step9_pilot100_20260930/smoke
/data/pxchen/envs/jepa-wam/bin/python scripts/train_guidance_direction_step9_pilot.py --phase smoke
/data/pxchen/envs/jepa-wam/bin/python scripts/train_guidance_direction_step9_pilot.py --phase train
