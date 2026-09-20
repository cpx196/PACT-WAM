#!/usr/bin/env bash
# Shared runtime environment for the local LIBERO deployment.
# Source this file before invoking any FastWAM LIBERO command.

set -euo pipefail

FASTWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTWAM_PYTHON="/data/chenpengxu/conda_envs/HMoE/bin/python"
FASTWAM_EXTRA_SITE_PACKAGES="/data/chenpengxu/jepa_wam_py312"
LIBERO_ROOT="/data/chenpengxu/LIBERO"

export FASTWAM_ROOT FASTWAM_PYTHON
export PYTHONPATH="${FASTWAM_EXTRA_SITE_PACKAGES}:${FASTWAM_ROOT}/src:${LIBERO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/home/chenpengxu/.libero}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${FASTWAM_ROOT}/checkpoints/wan}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope}"
export HF_HOME="${HF_HOME:-/data/chenpengxu/hf_home}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"

mkdir -p "${DIFFSYNTH_MODEL_BASE_PATH}" "${FASTWAM_ROOT}/checkpoints/fastwam_release"

if [[ ! -x "${FASTWAM_PYTHON}" ]]; then
    echo "FastWAM Python interpreter not found: ${FASTWAM_PYTHON}" >&2
    exit 1
fi

