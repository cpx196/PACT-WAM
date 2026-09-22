#!/usr/bin/env bash
# Run one seed of strict-cascade Optional-IDM evaluation with raw-gradient
# V-JEPA2-AC guidance and an RMS trust-region cap.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local_libero_env.sh"
cd "${FASTWAM_ROOT}"

: "${TASK_FILE:?Set TASK_FILE to a manager task list, for example libero_10,365}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-7}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FASTWAM_ROOT}/evaluate_results/raw_gradient_guidance_seed${SEED}_${RUN_TAG}}"
RAW_GRADIENT_SCALE="${RAW_GRADIENT_SCALE:-1.0}"
MAX_DELTA_RMS="${MAX_DELTA_RMS:-0.02}"
VERIFY_DESCENT="${VERIFY_DESCENT:-true}"

if [[ ! -f "${TASK_FILE}" ]]; then
    echo "Task file not found: ${TASK_FILE}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec env CUDA_VISIBLE_DEVICES="${GPU_ID}" MUJOCO_EGL_DEVICE_ID="${GPU_ID}" \
    "${FASTWAM_PYTHON}" experiments/libero/run_libero_manager.py \
    task=libero_optional_idm_2cam224_1e-4 \
    "ckpt=${FASTWAM_CHECKPOINT}" \
    "EVALUATION.dataset_stats_path=${FASTWAM_DATASET_STATS}" \
    "EVALUATION.output_dir=${OUTPUT_ROOT}" \
    "MULTIRUN.task_file=${TASK_FILE}" \
    MULTIRUN.num_gpus=1 \
    EVALUATION.num_trials=1 \
    EVALUATION.replan_steps=8 \
    EVALUATION.num_inference_steps=10 \
    EVALUATION.compile_action_infer=false \
    EVALUATION.visualize_future_video=true \
    EVALUATION.offload_text_encoder=true \
    EVALUATION.action_infer_mode=idm \
    EVALUATION.video_sigma_shift=1.0 \
    EVALUATION.action_sigma_shift=1.0 \
    EVALUATION.sigma_shift=null \
    EVALUATION.trigger_probe.enabled=false \
    EVALUATION.vjepa2_ac.enabled=true \
    EVALUATION.vjepa2_ac.num_action_candidates=8 \
    EVALUATION.vjepa2_ac.replan_steps=8 \
    EVALUATION.vjepa2_ac.candidate_generation_mode=separate \
    "EVALUATION.vjepa2_ac.checkpoint_path=${VJEPA2_CHECKPOINT}" \
    "EVALUATION.vjepa2_ac.repo_path=${VJEPA2_REPO}" \
    EVALUATION.vjepa2_ac.dtype=float32 \
    EVALUATION.vjepa2_ac.camera.preset=libero_default \
    EVALUATION.vjepa2_ac.camera.position=null \
    EVALUATION.vjepa2_ac.camera.quaternion=null \
    EVALUATION.vjepa2_ac.camera.fovy=null \
    EVALUATION.vjepa2_ac.camera.image_rotation_degrees=180 \
    EVALUATION.vjepa2_ac.guidance.enabled=true \
    EVALUATION.vjepa2_ac.guidance.overlap_encoder_with_action=true \
    'EVALUATION.vjepa2_ac.guidance.after_flow_steps=[7,8,9]' \
    "EVALUATION.vjepa2_ac.guidance.step_size=${RAW_GRADIENT_SCALE}" \
    EVALUATION.vjepa2_ac.guidance.normalize_gradient=false \
    "EVALUATION.vjepa2_ac.guidance.max_delta_rms=${MAX_DELTA_RMS}" \
    EVALUATION.vjepa2_ac.guidance.ac_steps=2 \
    "EVALUATION.vjepa2_ac.guidance.verify_descent=${VERIFY_DESCENT}" \
    model.load_text_encoder=true \
    model.redirect_common_files=false \
    model.skip_dit_load_from_pretrain=true \
    "seed=${SEED}"
