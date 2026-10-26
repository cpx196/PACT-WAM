#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${1:-2}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PYTHONPATH=/data/pxchen/PACT-WAM/artifacts/jepawam_deploy_smoke_20261009/python_deps:/data/pxchen/JEPA_WAM:/data/pxchen/vjepa2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec /data/pxchen/envs/jepa-wam/bin/python -u /data/pxchen/PACT-WAM/scripts/prepare_fastwam_ar2_and_smoke.py
