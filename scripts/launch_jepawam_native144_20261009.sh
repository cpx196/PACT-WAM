#!/bin/bash
set -euo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=4 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=4
export LIBERO_PATH=/data/pxchen/LIBERO-plus
export LIBERO_CONFIG_PATH=/data/pxchen/PACT-WAM/artifacts/jepawam_deploy_smoke_20261009/libero_config
export PYTHONPATH=/data/pxchen/PACT-WAM/artifacts/jepawam_deploy_smoke_20261009/python_deps:/data/pxchen/JEPA_WAM:/data/pxchen/LIBERO-plus
export TOKENIZERS_PARALLELISM=false NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR=/data/pxchen/PACT-WAM/artifacts/jepawam_deploy_smoke_20261009/matplotlib
root=/data/pxchen/PACT-WAM/artifacts/jepawam_native_full144_20261009
cd /data/pxchen/JEPA_WAM
exec /data/pxchen/envs/jepa-wam/bin/python -u /data/pxchen/PACT-WAM/scripts/eval_jepawam_native144_20261009.py --root "$root" --manifest "$root/task_manifest144.json"
