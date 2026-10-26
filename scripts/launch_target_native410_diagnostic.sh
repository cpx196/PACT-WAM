#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${1:-2}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID="$CUDA_VISIBLE_DEVICES"
export LIBERO_PATH=/data/pxchen/LIBERO-plus
export LIBERO_CONFIG_PATH=/data/pxchen/PACT-WAM/artifacts/jepawam_deploy_smoke_20261009/libero_config
export PYTHONPATH=/data/pxchen/PACT-WAM/artifacts/jepawam_deploy_smoke_20261009/python_deps:/data/pxchen/JEPA_WAM:/data/pxchen/LIBERO-plus
export TOKENIZERS_PARALLELISM=false NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR=/data/pxchen/PACT-WAM/artifacts/jepawam_deploy_smoke_20261009/matplotlib
exec /data/pxchen/envs/jepa-wam/bin/python -u /data/pxchen/PACT-WAM/scripts/diagnose_guidance_target_and_native410.py
