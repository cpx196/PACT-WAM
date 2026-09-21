#!/usr/bin/env bash
# Shared runtime environment for the local LIBERO deployment.
# Source this file before invoking any FastWAM LIBERO command.

set -euo pipefail

FASTWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTWAM_PYTHON="/data/pxchen/envs/jepa-wam/bin/python"
FASTWAM_EXTRA_SITE_PACKAGES=""
LIBERO_ROOT="/data/pxchen/LIBERO-plus"
VJEPA2_ROOT="/data/pxchen/vjepa2"
VJEPA2_REPO="/data/pxchen/JEPA_WAM"
CHECKPOINT_ROOT="${FASTWAM_ROOT}/checkpoints"
WAN22_CHECKPOINT_ROOT="${CHECKPOINT_ROOT}/wan"
FASTWAM_CHECKPOINT_ROOT="${CHECKPOINT_ROOT}/fastwam"
VJEPA2_CHECKPOINT_ROOT="${CHECKPOINT_ROOT}/vjepa2"
FASTWAM_CHECKPOINT="${FASTWAM_CHECKPOINT_ROOT}/libero_optional_idm_2cam224.pt"
FASTWAM_DATASET_STATS="${FASTWAM_CHECKPOINT_ROOT}/libero_optional_idm_2cam224_dataset_stats.json"
VJEPA2_CHECKPOINT="${VJEPA2_CHECKPOINT_ROOT}/vjepa2-ac-vitg.pt"

export FASTWAM_ROOT FASTWAM_PYTHON
export PYTHONPATH="${VJEPA2_ROOT}:${FASTWAM_ROOT}/src:${LIBERO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${FASTWAM_ROOT}/.libero_plus_config}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${WAN22_CHECKPOINT_ROOT}}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope}"
export HF_HOME="${HF_HOME:-/data/pxchen/hf_home}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"

if [[ ! -x "${FASTWAM_PYTHON}" ]]; then
    echo "FastWAM Python interpreter not found: ${FASTWAM_PYTHON}" >&2
    exit 1
fi

for required_path in \
    "${LIBERO_CONFIG_PATH}" \
    "${WAN22_CHECKPOINT_ROOT}/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" \
    "${WAN22_CHECKPOINT_ROOT}/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth" \
    "${WAN22_CHECKPOINT_ROOT}/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl" \
    "${FASTWAM_CHECKPOINT}" \
    "${FASTWAM_DATASET_STATS}" \
    "${VJEPA2_CHECKPOINT}" \
    "${VJEPA2_REPO}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required deployment path not found: ${required_path}" >&2
        exit 1
    fi
done

export FASTWAM_ROOT FASTWAM_PYTHON CHECKPOINT_ROOT WAN22_CHECKPOINT_ROOT
export FASTWAM_CHECKPOINT_ROOT FASTWAM_CHECKPOINT FASTWAM_DATASET_STATS
export VJEPA2_ROOT VJEPA2_REPO VJEPA2_CHECKPOINT
