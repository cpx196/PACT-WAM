#!/usr/bin/env bash
# Compare inter-step JEPA guidance against one-step flow + JEPA on the fixed
# LIBERO-Plus Long core3 slice. Four fixed seeds are staggered across four GPUs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local_libero_env.sh"

GPUS_TEXT="${GPU_LIST:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "${GPUS_TEXT}"
SEEDS=(${SEEDS:-7 17 27 37})
TASK_FILE="${TASK_FILE:-${FASTWAM_ROOT}/experiments/libero/task_lists/libero_plus_long_robot_init_core3.txt}"
CHECKPOINT="${CHECKPOINT:-${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
DATASET_STATS="${DATASET_STATS:-${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FASTWAM_ROOT}/evaluate_results/jepa_guidance_boundary_compare_long_core3_${RUN_TAG}}"

if (( ${#GPUS[@]} != ${#SEEDS[@]} )); then
    echo "Expected one GPU per seed; got ${#GPUS[@]} GPUs and ${#SEEDS[@]} seeds." >&2
    exit 1
fi

export PYTHONPATH="/data/chenpengxu/LIBERO-plus:${FASTWAM_ROOT}/src:${FASTWAM_EXTRA_SITE_PACKAGES}"
export LIBERO_CONFIG_PATH="/data/chenpengxu/JEPA_WAM/.libero_plus_config"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${OUTPUT_ROOT}/logs"
printf '%s\n' "$$" > "${OUTPUT_ROOT}/launcher.pid"
printf '%s\n' "${OUTPUT_ROOT}" > "${OUTPUT_ROOT}/output_root.txt"

run_seed() {
    local setting="$1"
    local after_flow_steps="$2"
    local num_inference_steps="$3"
    local gpu="$4"
    local seed="$5"
    local output_dir="${OUTPUT_ROOT}/${setting}/seed_${seed}"
    local log_file="${OUTPUT_ROOT}/logs/${setting}_seed${seed}_gpu${gpu}.log"

    env \
        CUDA_VISIBLE_DEVICES="${gpu}" \
        MUJOCO_EGL_DEVICE_ID="${gpu}" \
        PYOPENGL_PLATFORM=egl \
        MUJOCO_GL=egl \
        "${FASTWAM_PYTHON}" experiments/libero/run_libero_manager.py \
        task=libero_uncond_2cam224_1e-4 \
        "ckpt=${CHECKPOINT}" \
        "seed=${seed}" \
        EVALUATION.num_trials=1 \
        "EVALUATION.output_dir=${output_dir}" \
        "EVALUATION.dataset_stats_path=${DATASET_STATS}" \
        EVALUATION.sigma_shift=5.0 \
        "EVALUATION.num_inference_steps=${num_inference_steps}" \
        EVALUATION.replan_steps=8 \
        EVALUATION.visualize_future_video=true \
        EVALUATION.offload_text_encoder=true \
        EVALUATION.timing_enabled=true \
        EVALUATION.vjepa2_ac.enabled=true \
        EVALUATION.vjepa2_ac.candidate_generation_mode=separate \
        EVALUATION.vjepa2_ac.guidance.enabled=true \
        EVALUATION.vjepa2_ac.guidance.last_flow_steps=0 \
        "EVALUATION.vjepa2_ac.guidance.after_flow_steps=${after_flow_steps}" \
        EVALUATION.vjepa2_ac.guidance.step_size=0.02 \
        EVALUATION.vjepa2_ac.guidance.ac_steps=2 \
        EVALUATION.vjepa2_ac.guidance.verify_descent=false \
        "MULTIRUN.task_file=${TASK_FILE}" \
        MULTIRUN.num_gpus=1 \
        >"${log_file}" 2>&1
}

run_setting() {
    local setting="$1"
    local after_flow_steps="$2"
    local num_inference_steps="$3"
    local pids=()

    mkdir -p "${OUTPUT_ROOT}/${setting}"
    for index in "${!GPUS[@]}"; do
        local gpu="${GPUS[$index]}"
        local seed="${SEEDS[$index]}"
        run_seed "${setting}" "${after_flow_steps}" "${num_inference_steps}" "${gpu}" "${seed}" &
        local job_pid="$!"
        pids+=("${job_pid}")
        printf '%s %s %s %s\n' "${setting}" "${gpu}" "${seed}" "${job_pid}" >> "${OUTPUT_ROOT}/jobs.txt"

        # T5 alone uses about 10.6 GiB of host-backed model weights per worker.
        # Wait until it is released before starting the next GPU.
        if (( index + 1 < ${#GPUS[@]} )); then
            local worker_log="${OUTPUT_ROOT}/${setting}/seed_${seed}/.worker_pool/logs/worker_${gpu}.log"
            echo "Waiting for ${setting} seed ${seed} T5 release before starting the next GPU"
            while kill -0 "${job_pid}" 2>/dev/null; do
                if [[ -f "${worker_log}" ]] && grep -q "Released standalone T5" "${worker_log}"; then
                    break
                fi
                sleep 5
            done
        fi
    done

    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    if (( failed != 0 )); then
        echo "Setting ${setting} failed; inspect ${OUTPUT_ROOT}/logs." >&2
        return 1
    fi
    echo "Completed ${setting}: ${OUTPUT_ROOT}/${setting}"
}

# With ten zero-based denoise steps 0..9, these insertions occur after steps
# 6, 7, and 8, so the corrected states are consumed by steps 7, 8, and 9.
run_setting "between_67_78_89" "[6,7,8]" 10

# A one-step schedule has sigma 1 -> 0. Its sole flow update is the clean
# estimate; guidance after step 0 therefore becomes the directly returned action.
run_setting "one_denoise_then_jepa" "[0]" 1

echo "Completed both JEPA boundary-guidance settings: ${OUTPUT_ROOT}"
