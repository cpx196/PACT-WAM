#!/usr/bin/env bash
# Wait for enough GPU headroom, then run the four-seed FastWAM-JEPA pose16 eval.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local_libero_env.sh"

GPU_LIST="${GPU_LIST:-2,3}"
SEEDS="${SEEDS:-7 17 27 37}"
MAX_PREEXISTING_GPU_MIB="${MAX_PREEXISTING_GPU_MIB:-4096}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FASTWAM_ROOT}/evaluate_results/fastwam_jepa_libero_plus_pose16_${RUN_TAG}}"

mkdir -p "${OUTPUT_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Waiting for GPUs ${GPU_LIST}: pre-existing use must be <= ${MAX_PREEXISTING_GPU_MIB} MiB per GPU."

IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
while true; do
    mapfile -t USED_MIB < <(
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    )
    ready=true
    status=()
    for gpu_id in "${GPU_IDS[@]}"; do
        used="${USED_MIB[${gpu_id}]}"
        status+=("GPU${gpu_id}=${used}MiB")
        if (( used > MAX_PREEXISTING_GPU_MIB )); then
            ready=false
        fi
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${status[*]} ready=${ready}"
    if [[ "${ready}" == "true" ]]; then
        break
    fi
    sleep "${GPU_POLL_SECONDS}"
done

exec env \
    ENABLE_VJEPA=true \
    GPU_LIST="${GPU_LIST}" \
    SEEDS="${SEEDS}" \
    RUN_TAG="${RUN_TAG}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    "${SCRIPT_DIR}/run_fastwam_libero_plus_pose16.sh"
