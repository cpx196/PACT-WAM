#!/usr/bin/env bash
# Evaluate the released, single-sample FastWAM policy on 16 official
# LIBERO-Plus object-layout tasks with four independent inference seeds.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local_libero_env.sh"

LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/data/chenpengxu/LIBERO-plus}"
LIBERO_PLUS_CONFIG="${LIBERO_PLUS_CONFIG:-/data/chenpengxu/JEPA_WAM/.libero_plus_config}"
TASK_FILE="${TASK_FILE:-${FASTWAM_ROOT}/experiments/libero/libero_plus_pose16_tasks.txt}"
CHECKPOINT="${CHECKPOINT:-${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
DATASET_STATS="${DATASET_STATS:-${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
GPU_LIST="${GPU_LIST:-2,3}"
SEEDS="${SEEDS:-7 17 27 37}"
ENABLE_VJEPA="${ENABLE_VJEPA:-false}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
if [[ "${ENABLE_VJEPA}" == "true" ]]; then
    DEFAULT_RUN_NAME="fastwam_jepa_libero_plus_pose16"
    SUMMARY_TITLE="FastWAM-JEPA on LIBERO-Plus pose16"
    POLICY_DESCRIPTION='released `libero_uncond_2cam224.pt`; Best-of-8 action sampling; FP32 V-JEPA2-AC ranking; 8-step replanning; future-video prediction enabled.'
    MODE_DESCRIPTION="FastWAM-JEPA Best-of-8 (V-JEPA2-AC enabled)"
    REPLAN_STEPS=8
    VISUALIZE_FUTURE_VIDEO=true
else
    DEFAULT_RUN_NAME="fastwam_original_libero_plus_pose16"
    SUMMARY_TITLE="Original FastWAM on LIBERO-Plus pose16"
    POLICY_DESCRIPTION='released `libero_uncond_2cam224.pt`; single action sample; no V-JEPA; no Best-of-K; no future-video rollout.'
    MODE_DESCRIPTION="original FastWAM single-sample baseline (V-JEPA disabled)"
    REPLAN_STEPS=10
    VISUALIZE_FUTURE_VIDEO=false
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${FASTWAM_ROOT}/evaluate_results/${DEFAULT_RUN_NAME}_${RUN_TAG}}"

if [[ ! -d "${LIBERO_PLUS_ROOT}/libero/libero" ]]; then
    echo "LIBERO-Plus repository is incomplete: ${LIBERO_PLUS_ROOT}" >&2
    exit 1
fi
for required_file in "${LIBERO_PLUS_CONFIG}" "${TASK_FILE}" "${CHECKPOINT}" "${DATASET_STATS}"; do
    if [[ ! -e "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done
if [[ "${ENABLE_VJEPA}" == "true" ]]; then
    for required_path in /data/chenpengxu/jepa_wam_assets/vjepa2-ac-vitg.pt /data/chenpengxu/vjepa2; do
        if [[ ! -e "${required_path}" ]]; then
            echo "Required V-JEPA2-AC resource not found: ${required_path}" >&2
            exit 1
        fi
    done
fi

# LIBERO-Plus installs under the same namespace as vanilla LIBERO. Put the
# official Plus checkout first and point its path resolver at Plus assets.
export PYTHONPATH="${LIBERO_PLUS_ROOT}:${FASTWAM_ROOT}/src:${FASTWAM_EXTRA_SITE_PACKAGES}"
export LIBERO_CONFIG_PATH="${LIBERO_PLUS_CONFIG}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
NUM_GPUS="${#GPU_IDS[@]}"
mkdir -p "${OUTPUT_ROOT}"
cp -f "${TASK_FILE}" "${OUTPUT_ROOT}/tasks.txt"

# Resolve every task and init-state file before loading T5 / FastWAM. This
# catches an incomplete or wrongly configured LIBERO-Plus checkout early.
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" \
MUJOCO_EGL_DEVICE_ID="${GPU_IDS[0]}" \
TASK_FILE="${TASK_FILE}" "${FASTWAM_PYTHON}" - <<'PY'
import os

from libero.libero import benchmark

task_file = os.environ["TASK_FILE"]
suites = {}
checked = 0
with open(task_file, encoding="utf-8") as handle:
    for line_number, raw_line in enumerate(handle, 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        suite_name, task_id_text = (part.strip() for part in line.split(",", 1))
        task_id = int(task_id_text)
        suite = suites.get(suite_name)
        if suite is None:
            suite = benchmark.get_benchmark_dict()[suite_name]()
            suites[suite_name] = suite
        task = suite.get_task(task_id)
        states = suite.get_task_init_states(task_id)
        if len(states) == 0:
            raise RuntimeError(
                f"No init states for {suite_name},{task_id} ({task.name})"
            )
        checked += 1
print(f"Preflight passed: resolved {checked} LIBERO-Plus tasks and init states.")
PY

summarize_results() {
    OUTPUT_ROOT="${OUTPUT_ROOT}" SEEDS="${SEEDS}" SUMMARY_TITLE="${SUMMARY_TITLE}" \
        POLICY_DESCRIPTION="${POLICY_DESCRIPTION}" "${FASTWAM_PYTHON}" - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"]).resolve()
seeds = os.environ["SEEDS"].split()
tasks = [
    ("Alphabet Soup", "add_10 control", 1818),
    ("Alphabet Soup", "pose level 1", 1839),
    ("Alphabet Soup", "pose level 3", 1847),
    ("Alphabet Soup", "pose level 5", 1855),
    ("Milk", "add_10 control", 2037),
    ("Milk", "pose level 1", 2062),
    ("Milk", "pose level 3", 2070),
    ("Milk", "pose level 5", 2078),
    ("Salad Dressing", "add_10 control", 2131),
    ("Salad Dressing", "pose level 1", 2156),
    ("Salad Dressing", "pose level 3", 2163),
    ("Salad Dressing", "pose level 5", 2169),
    ("Tomato Sauce", "add_10 control", 2174),
    ("Tomato Sauce", "pose level 1", 2203),
    ("Tomato Sauce", "pose level 3", 2211),
    ("Tomato Sauce", "pose level 5", 2217),
]

rows = []
for target, variant, task_id in tasks:
    successes = 0
    trials = 0
    completed_seeds = []
    videos = []
    for seed in seeds:
        seed_root = root / f"seed_{seed}" / "libero_object"
        result_files = sorted(seed_root.glob(f"gpu*_task{task_id}_results.json"))
        if result_files:
            result = json.loads(result_files[0].read_text(encoding="utf-8"))
            successes += int(result.get("successes", 0))
            trials += int(result.get("total_episodes", 0))
            completed_seeds.append(seed)
        videos.extend(str(path.relative_to(root)) for path in sorted((seed_root / "videos").glob(f"*task{task_id}_trial0*.mp4")))
    rows.append({
        "target": target,
        "variant": variant,
        "task_id": task_id,
        "successes": successes,
        "trials": trials,
        "success_rate": successes / trials if trials else "",
        "completed_seeds": ";".join(completed_seeds),
        "videos": ";".join(videos),
    })

csv_path = root / "summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

md = [
    f"# {os.environ['SUMMARY_TITLE']}",
    "",
    f"Policy: {os.environ['POLICY_DESCRIPTION']}",
    "",
    "| Target | Variant | Task ID | Success | Rate | Completed seeds |",
    "|---|---|---:|---:|---:|---|",
]
for row in rows:
    rate = f"{100 * row['success_rate']:.1f}%" if row["trials"] else "pending"
    md.append(
        f"| {row['target']} | {row['variant']} | {row['task_id']} | "
        f"{row['successes']}/{row['trials']} | {rate} | {row['completed_seeds'] or 'pending'} |"
    )
md.extend(["", "Per-rollout video paths are recorded in `summary.csv` and stored below each `seed_*/libero_object/videos/` directory.", ""])
(root / "summary.md").write_text("\n".join(md), encoding="utf-8")
print(f"Updated aggregate tables: {csv_path} and {root / 'summary.md'}")
PY
}

echo "Output root: ${OUTPUT_ROOT}"
echo "GPUs: ${GPU_LIST}"
echo "Seeds: ${SEEDS}"
echo "Mode: ${MODE_DESCRIPTION}"

summarize_results
for seed in ${SEEDS}; do
    seed_output="${OUTPUT_ROOT}/seed_${seed}"
    echo "Starting seed ${seed}: ${seed_output}"
    "${FASTWAM_PYTHON}" "${FASTWAM_ROOT}/experiments/libero/run_libero_manager.py" \
        task=libero_uncond_2cam224_1e-4 \
        "ckpt=${CHECKPOINT}" \
        "seed=${seed}" \
        "EVALUATION.output_dir=${seed_output}" \
        EVALUATION.num_trials=1 \
        "EVALUATION.dataset_stats_path=${DATASET_STATS}" \
        EVALUATION.sigma_shift=5.0 \
        "EVALUATION.replan_steps=${REPLAN_STEPS}" \
        "EVALUATION.visualize_future_video=${VISUALIZE_FUTURE_VIDEO}" \
        "EVALUATION.vjepa2_ac.enabled=${ENABLE_VJEPA}" \
        EVALUATION.offload_text_encoder=true \
        "MULTIRUN.task_file=${TASK_FILE}" \
        "MULTIRUN.num_gpus=${NUM_GPUS}"
    summarize_results
done

echo "All 64 rollouts completed. Summary: ${OUTPUT_ROOT}/summary.md"
