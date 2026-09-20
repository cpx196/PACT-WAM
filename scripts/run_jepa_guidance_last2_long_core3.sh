#!/usr/bin/env bash
# Four-GPU, four-seed LIBERO-Plus Long core3 run with JEPA guidance on the
# final two action-flow steps. Each GPU runs all three fixed tasks once.

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${FASTWAM_ROOT}/evaluate_results/jepa_guidance_last2_long_core3_${RUN_TAG}}"
LAST_FLOW_STEPS="${LAST_FLOW_STEPS:-2}"
GUIDANCE_AC_STEPS="${GUIDANCE_AC_STEPS:-1}"
VERIFY_DESCENT="${VERIFY_DESCENT:-true}"

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
    local gpu="$1"
    local seed="$2"
    local output_dir="${OUTPUT_ROOT}/seed_${seed}"
    local log_file="${OUTPUT_ROOT}/logs/seed${seed}_gpu${gpu}.log"

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
        EVALUATION.vjepa2_ac.enabled=true \
        EVALUATION.vjepa2_ac.candidate_generation_mode=separate \
        EVALUATION.vjepa2_ac.guidance.enabled=true \
        "EVALUATION.vjepa2_ac.guidance.last_flow_steps=${LAST_FLOW_STEPS}" \
        EVALUATION.vjepa2_ac.guidance.step_size=0.02 \
        "EVALUATION.vjepa2_ac.guidance.ac_steps=${GUIDANCE_AC_STEPS}" \
        "EVALUATION.vjepa2_ac.guidance.verify_descent=${VERIFY_DESCENT}" \
        "MULTIRUN.task_file=${TASK_FILE}" \
        MULTIRUN.num_gpus=1 \
        >"${log_file}" 2>&1
}

pids=()
for index in "${!GPUS[@]}"; do
    gpu="${GPUS[$index]}"
    seed="${SEEDS[$index]}"
    run_seed "${gpu}" "${seed}" &
    job_pid="$!"
    pids+=("${job_pid}")
    printf '%s %s %s\n' "${gpu}" "${seed}" "${job_pid}" >> "${OUTPUT_ROOT}/jobs.txt"

    # Loading four T5 encoders at once exceeds host RAM. Stagger only this
    # phase; completed T5 weights are released before the next seed starts.
    if (( index + 1 < ${#GPUS[@]} )); then
        worker_log="${OUTPUT_ROOT}/seed_${seed}/.worker_pool/logs/worker_${gpu}.log"
        echo "Waiting for seed ${seed} T5 release before starting the next GPU"
        while kill -0 "${job_pid}" 2>/dev/null; do
            if [[ -f "${worker_log}" ]] && grep -q "Released standalone T5" "${worker_log}"; then
                break
            fi
            sleep 5
        done
    fi
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done

if (( failed != 0 )); then
    echo "One or more seed workers failed; inspect ${OUTPUT_ROOT}/logs." >&2
    exit 1
fi
echo "Completed JEPA guidance Long core3 (last_flow_steps=${LAST_FLOW_STEPS}, ac_steps=${GUIDANCE_AC_STEPS}): ${OUTPUT_ROOT}"
