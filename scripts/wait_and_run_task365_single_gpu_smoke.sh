#!/usr/bin/env bash

# Wait for one GPU with enough headroom, then run a minimal original-FastWAM
# task365 smoke test using the camera configured in sim_libero.yaml.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local_libero_env.sh"

THRESHOLD_MIB="${THRESHOLD_MIB:-14000}"
POLL_SECONDS="${POLL_SECONDS:-30}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FASTWAM_ROOT}/evaluate_results/fastwam_original_task365_camera_smoke_${RUN_TAG}}"
CHECKPOINT="${CHECKPOINT:-${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
DATASET_STATS="${DATASET_STATS:-${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"

export PYTHONPATH="${LIBERO_PLUS_ROOT:-/data/chenpengxu/LIBERO-plus}:${FASTWAM_ROOT}/src:${FASTWAM_EXTRA_SITE_PACKAGES}"
export LIBERO_CONFIG_PATH="${LIBERO_PLUS_CONFIG:-/data/chenpengxu/JEPA_WAM/.libero_plus_config}"
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUTPUT_ROOT}"
printf '%s\n' "$$" > "${OUTPUT_ROOT}/launcher.pid"
printf '%s\n' "${OUTPUT_ROOT}" > "${OUTPUT_ROOT}/output_root.txt"
printf '%s\n' queued > "${OUTPUT_ROOT}/status.txt"
exec >>"${OUTPUT_ROOT}/launcher.log" 2>&1

echo "[$(date -Is)] queued single-GPU original FastWAM task365 smoke test"
echo "[$(date -Is)] waiting for >=${THRESHOLD_MIB} MiB free on one GPU"

while true; do
    gpu="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | while IFS=, read -r idx free; do
        free="${free// /}"
        if [[ -n "${free}" && "${free}" -ge "${THRESHOLD_MIB}" ]]; then
            echo "${idx}"
            break
        fi
    done)"
    if [[ -n "${gpu}" ]]; then
        break
    fi
    echo "[$(date -Is)] no suitable GPU yet"
    sleep "${POLL_SECONDS}"
done

echo "[$(date -Is)] starting on physical GPU ${gpu}"
printf '%s\n' running > "${OUTPUT_ROOT}/status.txt"

export CUDA_VISIBLE_DEVICES="${gpu}"
export MUJOCO_EGL_DEVICE_ID="${gpu}"
export HYDRA_FULL_ERROR=1

cd "${FASTWAM_ROOT}"
set +e
"${FASTWAM_PYTHON}" experiments/libero/eval_libero_single.py \
    task=libero_uncond_2cam224_1e-4 \
    "ckpt=${CHECKPOINT}" \
    "gpu_id=${gpu}" \
    EVALUATION.task_suite_name=libero_10 \
    EVALUATION.task_id=365 \
    EVALUATION.num_trials=1 \
    "EVALUATION.output_dir=${OUTPUT_ROOT}" \
    "EVALUATION.dataset_stats_path=${DATASET_STATS}" \
    EVALUATION.vjepa2_ac.enabled=false \
    EVALUATION.visualize_future_video=false \
    EVALUATION.replan_steps=10 \
    EVALUATION.timing_enabled=true \
    EVALUATION.action_denoise_trace=false \
    EVALUATION.offload_text_encoder=false
rc=$?
set -e

printf '%s\n' "${rc}" > "${OUTPUT_ROOT}/exit_code.txt"
if [[ "${rc}" -eq 0 ]]; then
    printf '%s\n' complete > "${OUTPUT_ROOT}/status.txt"
else
    printf '%s\n' failed > "${OUTPUT_ROOT}/status.txt"
fi
echo "[$(date -Is)] smoke test exited rc=${rc}"
exit "${rc}"
