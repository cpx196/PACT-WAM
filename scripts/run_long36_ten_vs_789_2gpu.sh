#!/usr/bin/env bash
# Complete the 10-step control and run 10-step JEPA guidance on Long36.
# Two GPU workers run at a time; each T5 load finishes before the next starts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local_libero_env.sh"
cd "${FASTWAM_ROOT}"

CONTROL_TASK_FILE="${FASTWAM_ROOT}/experiments/libero/task_lists/libero_plus_long_30_ten_step_control_supplement.txt"
GUIDANCE_TASK_FILE="${FASTWAM_ROOT}/experiments/libero/task_lists/libero_plus_long_36_three_perturbations_two_variants.txt"
CHECKPOINT="${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224.pt"
DATASET_STATS="${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FASTWAM_ROOT}/evaluate_results/long36_ten_vs_789_${RUN_TAG}_2gpu}"
BASELINE_ROOT="${FASTWAM_ROOT}/evaluate_results/long8_one_jepa_vs_ten_control_20260916_long8_4gpu/ten_control"

export PYTHONPATH="/data/chenpengxu/LIBERO-plus:${FASTWAM_ROOT}/src:${FASTWAM_EXTRA_SITE_PACKAGES}"
export LIBERO_CONFIG_PATH="/data/chenpengxu/JEPA_WAM/.libero_plus_config"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUTPUT_ROOT}/logs"
printf '%s\n' "$$" > "${OUTPUT_ROOT}/launcher.pid"
printf '%s\n' "${OUTPUT_ROOT}" > "${OUTPUT_ROOT}/output_root.txt"
printf '%s\n' "${BASELINE_ROOT}" > "${OUTPUT_ROOT}/existing_control_root.txt"
cp "${CONTROL_TASK_FILE}" "${OUTPUT_ROOT}/control_supplement_task_list.txt"
cp "${GUIDANCE_TASK_FILE}" "${OUTPUT_ROOT}/guidance_task_list.txt"
printf '%s\n' running > "${OUTPUT_ROOT}/status.txt"

run_seed() {
    local arm="$1" gpu="$2" seed="$3" task_file="$4"
    local output_dir="${OUTPUT_ROOT}/${arm}/seed_${seed}"
    local log_file="${OUTPUT_ROOT}/logs/${arm}_seed${seed}_gpu${gpu}.log"
    local overrides=(
        task=libero_uncond_2cam224_1e-4
        "ckpt=${CHECKPOINT}"
        "seed=${seed}"
        EVALUATION.num_trials=1
        "EVALUATION.output_dir=${output_dir}"
        "EVALUATION.dataset_stats_path=${DATASET_STATS}"
        EVALUATION.sigma_shift=5.0
        EVALUATION.num_inference_steps=10
        EVALUATION.replan_steps=8
        EVALUATION.visualize_future_video=true
        EVALUATION.offload_text_encoder=true
        EVALUATION.timing_enabled=true
        EVALUATION.vjepa2_ac.candidate_generation_mode=separate
        EVALUATION.vjepa2_ac.replan_steps=8
        "MULTIRUN.task_file=${task_file}"
        MULTIRUN.num_gpus=1
    )
    if [[ "${arm}" == "ten_control_supplement" ]]; then
        overrides+=(
            EVALUATION.vjepa2_ac.enabled=false
            EVALUATION.vjepa2_ac.guidance.enabled=false
        )
    else
        overrides+=(
            EVALUATION.vjepa2_ac.enabled=true
            EVALUATION.vjepa2_ac.guidance.enabled=true
            EVALUATION.vjepa2_ac.guidance.last_flow_steps=3
            EVALUATION.vjepa2_ac.guidance.after_flow_steps=null
            EVALUATION.vjepa2_ac.guidance.step_size=0.02
            EVALUATION.vjepa2_ac.guidance.ac_steps=2
            EVALUATION.vjepa2_ac.guidance.verify_descent=false
        )
    fi
    env CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_EGL_DEVICE_ID="${gpu}" \
        PYOPENGL_PLATFORM=egl MUJOCO_GL=egl \
        "${FASTWAM_PYTHON}" experiments/libero/run_libero_manager.py \
        "${overrides[@]}" >"${log_file}" 2>&1
}

wait_for_t5_release() {
    local pid="$1" arm="$2" seed="$3" gpu="$4"
    local worker_log="${OUTPUT_ROOT}/${arm}/seed_${seed}/.worker_pool/logs/worker_${gpu}.log"
    while kill -0 "${pid}" 2>/dev/null; do
        if [[ -f "${worker_log}" ]] && rg -q 'Released standalone T5' "${worker_log}"; then
            echo "$(date -Is) T5 released: ${arm} seed ${seed} GPU ${gpu}"
            return 0
        fi
        sleep 5
    done
    echo "${arm} seed ${seed} exited before T5 release; inspect logs" >&2
    return 1
}

verify_seed() {
    local arm="$1" seed="$2" expected="$3"
    python - "${OUTPUT_ROOT}/${arm}/seed_${seed}/summary.json" "${expected}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
expected = int(sys.argv[2])
data = json.loads(path.read_text())
actual = len(data['task_results'])
if actual != expected:
    raise SystemExit(f'{path}: expected {expected} task results, found {actual}')
print(f'{path}: verified {actual} task results', flush=True)
PY
}

run_arm() {
    local arm="$1" task_file="$2" expected="$3"
    local seed0 seed1 pid0 pid1 failed
    echo "$(date -Is) starting ${arm}"
    for pair in '7 17' '27 37'; do
        read -r seed0 seed1 <<< "${pair}"
        run_seed "${arm}" 0 "${seed0}" "${task_file}" &
        pid0=$!
        printf '%s %s %s %s\n' "${arm}" 0 "${seed0}" "${pid0}" >> "${OUTPUT_ROOT}/jobs.txt"
        wait_for_t5_release "${pid0}" "${arm}" "${seed0}" 0 || return 1

        run_seed "${arm}" 1 "${seed1}" "${task_file}" &
        pid1=$!
        printf '%s %s %s %s\n' "${arm}" 1 "${seed1}" "${pid1}" >> "${OUTPUT_ROOT}/jobs.txt"
        failed=0
        wait "${pid0}" || failed=1
        wait "${pid1}" || failed=1
        if (( failed )); then
            echo "${arm} pair ${seed0}/${seed1} failed; inspect logs" >&2
            return 1
        fi
        verify_seed "${arm}" "${seed0}" "${expected}" || return 1
        verify_seed "${arm}" "${seed1}" "${expected}" || return 1
    done
    echo "$(date -Is) completed ${arm}"
}

if run_arm ten_control_supplement "${CONTROL_TASK_FILE}" 30 && \
   run_arm ten_guidance_789 "${GUIDANCE_TASK_FILE}" 36; then
    printf '%s\n' complete > "${OUTPUT_ROOT}/status.txt"
    echo "$(date -Is) completed Long36 evaluation: ${OUTPUT_ROOT}"
else
    printf '%s\n' failed > "${OUTPUT_ROOT}/status.txt"
    exit 1
fi
