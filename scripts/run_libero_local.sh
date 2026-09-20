#!/usr/bin/env bash
# Run the official FastWAM LIBERO evaluator with the pre-existing compatible env.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local_libero_env.sh"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

args=("$@")
has_offload_override=false
single_task=false
for arg in "${args[@]}"; do
    if [[ "${arg}" == EVALUATION.offload_text_encoder=* ]]; then
        has_offload_override=true
    fi
    if [[ "${arg}" == EVALUATION.task_id=* ]]; then
        single_task=true
    fi
done
if [[ "${has_offload_override}" == false ]]; then
    args+=("EVALUATION.offload_text_encoder=true")
fi

if [[ "${single_task}" == true ]]; then
    entrypoint="${FASTWAM_ROOT}/experiments/libero/eval_libero_single.py"
else
    entrypoint="${FASTWAM_ROOT}/experiments/libero/run_libero_manager.py"
fi

exec "${FASTWAM_PYTHON}" "${entrypoint}" "${args[@]}"
