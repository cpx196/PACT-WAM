#!/usr/bin/env bash
# Four-GPU paired LIBERO-Plus Long evaluation; T5 loads are staggered per arm.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local_libero_env.sh"
cd "${FASTWAM_ROOT}"

GPUS=(0 1 2 3)
SEEDS=(7 17 27 37)
TASK_FILE="${FASTWAM_ROOT}/experiments/libero/task_lists/libero_plus_long_8_mixed_perturbations.txt"
CHECKPOINT="${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224.pt"
DATASET_STATS="${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${FASTWAM_ROOT}/evaluate_results/long8_one_jepa_vs_ten_control_${RUN_TAG}"

export PYTHONPATH="/data/chenpengxu/LIBERO-plus:${FASTWAM_ROOT}/src:${FASTWAM_EXTRA_SITE_PACKAGES}"
export LIBERO_CONFIG_PATH="/data/chenpengxu/JEPA_WAM/.libero_plus_config"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUTPUT_ROOT}/logs"
printf '%s\n' "$$" > "${OUTPUT_ROOT}/launcher.pid"
printf '%s\n' "${OUTPUT_ROOT}" > "${OUTPUT_ROOT}/output_root.txt"
cp "${TASK_FILE}" "${OUTPUT_ROOT}/task_list.txt"

run_seed() {
    local arm="$1" gpu="$2" seed="$3"
    local steps guidance output_dir log_file
    output_dir="${OUTPUT_ROOT}/${arm}/seed_${seed}"
    log_file="${OUTPUT_ROOT}/logs/${arm}_seed${seed}_gpu${gpu}.log"
    if [[ "${arm}" == "ten_control" ]]; then
        steps=10
        guidance=false
    else
        steps=1
        guidance=true
    fi

    local overrides=(
        task=libero_uncond_2cam224_1e-4
        "ckpt=${CHECKPOINT}"
        "seed=${seed}"
        EVALUATION.num_trials=1
        "EVALUATION.output_dir=${output_dir}"
        "EVALUATION.dataset_stats_path=${DATASET_STATS}"
        EVALUATION.sigma_shift=5.0
        "EVALUATION.num_inference_steps=${steps}"
        EVALUATION.replan_steps=8
        EVALUATION.visualize_future_video=true
        EVALUATION.offload_text_encoder=true
        EVALUATION.timing_enabled=true
        "EVALUATION.vjepa2_ac.enabled=${guidance}"
        EVALUATION.vjepa2_ac.candidate_generation_mode=separate
        EVALUATION.vjepa2_ac.replan_steps=8
        "EVALUATION.vjepa2_ac.guidance.enabled=${guidance}"
        "MULTIRUN.task_file=${TASK_FILE}"
        MULTIRUN.num_gpus=1
    )
    if [[ "${guidance}" == true ]]; then
        overrides+=(
            EVALUATION.vjepa2_ac.guidance.last_flow_steps=0
            'EVALUATION.vjepa2_ac.guidance.after_flow_steps=[0]'
            EVALUATION.vjepa2_ac.guidance.step_size=0.02
            EVALUATION.vjepa2_ac.guidance.ac_steps=2
            EVALUATION.vjepa2_ac.guidance.verify_descent=false
        )
    fi

    env \
        CUDA_VISIBLE_DEVICES="${gpu}" \
        MUJOCO_EGL_DEVICE_ID="${gpu}" \
        PYOPENGL_PLATFORM=egl \
        MUJOCO_GL=egl \
        "${FASTWAM_PYTHON}" experiments/libero/run_libero_manager.py \
        "${overrides[@]}" >"${log_file}" 2>&1
}

run_arm() {
    local arm="$1" index gpu seed pid worker_log failed=0
    local pids=()
    echo "Starting ${arm} at $(date -Is)"
    for index in "${!GPUS[@]}"; do
        gpu="${GPUS[$index]}"
        seed="${SEEDS[$index]}"
        run_seed "${arm}" "${gpu}" "${seed}" &
        pid="$!"
        pids+=("${pid}")
        printf '%s %s %s %s\n' "${arm}" "${gpu}" "${seed}" "${pid}" >> "${OUTPUT_ROOT}/jobs.txt"

        if (( index + 1 < ${#GPUS[@]} )); then
            worker_log="${OUTPUT_ROOT}/${arm}/seed_${seed}/.worker_pool/logs/worker_${gpu}.log"
            echo "Waiting for ${arm} seed ${seed} T5 release before starting the next GPU"
            while kill -0 "${pid}" 2>/dev/null; do
                if [[ -f "${worker_log}" ]] && grep -q "Released standalone T5" "${worker_log}"; then
                    break
                fi
                sleep 5
            done
            if ! kill -0 "${pid}" 2>/dev/null; then
                echo "${arm} seed ${seed} exited before T5 release" >&2
                return 1
            fi
        fi
    done

    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    if (( failed != 0 )); then
        echo "${arm} failed; inspect ${OUTPUT_ROOT}/logs" >&2
        return 1
    fi
    echo "Completed ${arm} at $(date -Is)"
}

if run_arm ten_control && run_arm one_step_jepa; then
    echo "complete" > "${OUTPUT_ROOT}/status.txt"
    echo "Completed paired Long8 evaluation: ${OUTPUT_ROOT}"
else
    echo "failed" > "${OUTPUT_ROOT}/status.txt"
    exit 1
fi
