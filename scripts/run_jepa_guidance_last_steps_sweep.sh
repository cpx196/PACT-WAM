#!/usr/bin/env bash
# Four-GPU matched sweep for JEPA action-flow guidance start depth.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local_libero_env.sh"

GPUS_TEXT="${GPU_LIST:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "${GPUS_TEXT}"
SEEDS=(${SEEDS:-7 17 27 37})
LAST_STEPS=(${LAST_STEPS:-2 4 6 8 10})
TASK_FILE="${TASK_FILE:-${FASTWAM_ROOT}/experiments/libero/task_lists/jepa_guidance_probe_2169_2211.txt}"
CHECKPOINT="${CHECKPOINT:-${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
DATASET_STATS="${DATASET_STATS:-${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FASTWAM_ROOT}/evaluate_results/jepa_guidance_last_steps_sweep_${RUN_TAG}}"

if (( ${#GPUS[@]} != ${#SEEDS[@]} )); then
    echo "This sweep expects one GPU per seed; got ${#GPUS[@]} GPUs and ${#SEEDS[@]} seeds." >&2
    exit 1
fi

export PYTHONPATH="/data/chenpengxu/LIBERO-plus:${FASTWAM_ROOT}/src:${FASTWAM_EXTRA_SITE_PACKAGES}"
export LIBERO_CONFIG_PATH="/data/chenpengxu/JEPA_WAM/.libero_plus_config"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${OUTPUT_ROOT}/logs"

run_job() {
    local gpu="$1"
    local seed="$2"
    local label="$3"
    local guidance_enabled="$4"
    local last_steps="$5"
    local output_dir="${OUTPUT_ROOT}/${label}/seed_${seed}"
    local log_file="${OUTPUT_ROOT}/logs/${label}_seed${seed}_gpu${gpu}.log"

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
        EVALUATION.replan_steps=8 \
        EVALUATION.visualize_future_video=true \
        EVALUATION.offload_text_encoder=true \
        EVALUATION.timing_enabled=true \
        "EVALUATION.vjepa2_ac.enabled=${guidance_enabled}" \
        EVALUATION.vjepa2_ac.candidate_generation_mode=separate \
        "EVALUATION.vjepa2_ac.guidance.enabled=${guidance_enabled}" \
        "EVALUATION.vjepa2_ac.guidance.last_flow_steps=${last_steps}" \
        EVALUATION.vjepa2_ac.guidance.step_size=0.02 \
        EVALUATION.vjepa2_ac.guidance.ac_steps=1 \
        EVALUATION.vjepa2_ac.guidance.verify_descent=true \
        "MULTIRUN.task_file=${TASK_FILE}" \
        MULTIRUN.num_gpus=1 \
        >"${log_file}" 2>&1
}

run_wave() {
    local label="$1"
    local guidance_enabled="$2"
    local last_steps="$3"
    local pids=()
    echo "Starting ${label}: last_flow_steps=${last_steps}"
    for index in "${!GPUS[@]}"; do
        local gpu="${GPUS[$index]}"
        local seed="${SEEDS[$index]}"
        run_job "${gpu}" "${seed}" "${label}" "${guidance_enabled}" "${last_steps}" &
        local job_pid="$!"
        pids+=("${job_pid}")

        # Four simultaneous T5 loads exceed this host's 94 GiB RAM even
        # though all four GPUs have enough VRAM. Stagger only the T5 phase;
        # once its weights are released, rollout proceeds concurrently.
        if (( index + 1 < ${#GPUS[@]} )); then
            local worker_log="${OUTPUT_ROOT}/${label}/seed_${seed}/.worker_pool/logs/worker_${gpu}.log"
            echo "Waiting for seed ${seed} T5 release before starting the next GPU"
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
        echo "Wave ${label} failed; inspect ${OUTPUT_ROOT}/logs." >&2
        exit 1
    fi
    echo "Completed ${label}"
}

# Matched K=1 future-video baseline. Historical K=8 ranking results are kept
# separately and do not need to be recomputed for this interval sweep.
run_wave baseline_k1 false 0
for last_steps in "${LAST_STEPS[@]}"; do
    run_wave "last_${last_steps}" true "${last_steps}"
done

echo "Sweep complete: ${OUTPUT_ROOT}"
