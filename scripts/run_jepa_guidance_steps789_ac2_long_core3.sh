#!/usr/bin/env bash
# Reproducible Long core3 treatment: guide action-flow steps 7/8/9 against
# both four-action V-JEPA2-AC transitions spanning the executed 8-action chunk.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/chenpengxu/FastWAM/evaluate_results/jepa_guidance_steps789_ac2_long_core3_${RUN_TAG}}"

exec env \
    RUN_TAG="${RUN_TAG}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    LAST_FLOW_STEPS=3 \
    GUIDANCE_AC_STEPS=2 \
    VERIFY_DESCENT=false \
    bash "${SCRIPT_DIR}/run_jepa_guidance_last2_long_core3.sh"
