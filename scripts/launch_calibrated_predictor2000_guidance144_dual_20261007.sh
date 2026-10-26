#!/usr/bin/env bash
set -euo pipefail

cd /data/pxchen/PACT-WAM
source scripts/local_libero_env.sh
export PYTHONPATH=/data/pxchen/PACT-WAM:${PYTHONPATH}
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

output_root=/data/pxchen/PACT-WAM/evaluate_results/guidance789_rawgrad4x_long36_full144_libero10_calibrated_step2000_20261007
task_file="$output_root/tasks36.txt"
jepa_checkpoint=/data/pxchen/PACT-WAM/checkpoints/vjepa2-ac-libero10-response-calibrated-rot180-step2000-inference.pt
trained=/data/pxchen/PACT-WAM/artifacts/jepa_libero10_response_calibrated_rot180_20261007/train_2000/step_002000.pt
base=/data/pxchen/JEPA_WAM/checkpoints/vjepa2/vjepa2-ac-vitg.pt
cal=/data/pxchen/PACT-WAM/artifacts/libero_ac_command_response_calibration_20261007/calibration.json

test -s "$FASTWAM_CHECKPOINT" || exit 2
mkdir -p "$output_root"
exec 9> "$output_root/launcher.lock"
flock -n 9 || exit 3
if [[ -d "$output_root/seed_7" || -d "$output_root/smoke" ]]; then echo 'Refuse duplicate run' >&2;exit 4;fi
trap 'code=$?; if (( code != 0 )); then echo "FAILED exit=$code" > "$output_root/status.txt"; fi' EXIT
echo $$ > "$output_root/launcher.pid"
echo EXPORT_AND_PREFLIGHT > "$output_root/status.txt"
if [[ ! -e "$jepa_checkpoint" ]]; then
    CUDA_VISIBLE_DEVICES='' "$FASTWAM_PYTHON" scripts/export_libero90_vjepa2_ac.py \
        --base "$base" --trained "$trained" --output "$jepa_checkpoint" > "$output_root/export.log" 2>&1
fi
CUDA_VISIBLE_DEVICES='' "$FASTWAM_PYTHON" scripts/prepare_calibrated_guidance144_20261007.py \
    --out "$output_root" --inference "$jepa_checkpoint" > "$output_root/preflight.log" 2>&1
read -r position_scale rotation_scale < <("$FASTWAM_PYTHON" -c 'import json,sys;c=json.load(open(sys.argv[1]));print(c["osc_position_scale"]*c["translation_response_gain"],c["osc_rotation_scale"]*c["rotation_response_gain"])' "$cal")
pick_two() {
"$FASTWAM_PYTHON" - <<'PY'
import subprocess
busy=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).split())
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines();free=[]
for r in rows:
    i,u,m,v=[x.strip() for x in r.split(',')]
    if u not in busy and int(m)<128 and int(v)<5:free.append(i)
if len(free)<2:raise SystemExit(1)
print(' '.join(free[:2]))
PY
}
echo WAITING_TWO_IDLE_GPUS > "$output_root/status.txt"
deadline=$((SECONDS+7200))
until selected=$(pick_two); do
    if (( SECONDS >= deadline )); then exit 75;fi
    sleep 20
done
read -r gpu_a gpu_b <<< "$selected"
echo "$(date '+%F %T %Z') SMOKE_GPU=$gpu_a SECOND_GPU=$gpu_b" > "$output_root/gpu_assignment.txt"

run_seed() {
    local seed=$1
    local gpu=$2
    local run_dir="$output_root/seed_${seed}"
    local selected_tasks="$task_file"
    local max_steps=null
    if [[ "${3:-formal}" == smoke ]]; then
        run_dir="$output_root/smoke";selected_tasks="$output_root/smoke_task.txt";max_steps=48
    fi
    mkdir -p "$run_dir"
    echo "[$(date '+%F %T')] START seed=$seed gpu=$gpu" | tee -a "$output_root/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" \
        "$FASTWAM_PYTHON" experiments/libero/run_libero_manager.py \
        task=libero_optional_idm_2cam224_1e-4 \
        "ckpt=$FASTWAM_CHECKPOINT" \
        "EVALUATION.dataset_stats_path=$FASTWAM_DATASET_STATS" \
        "EVALUATION.output_dir=$run_dir" \
        "MULTIRUN.task_file=$selected_tasks" \
        MULTIRUN.num_gpus=1 \
        EVALUATION.num_trials=1 \
        "EVALUATION.max_steps_override=$max_steps" \
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
        "EVALUATION.vjepa2_ac.checkpoint_path=$jepa_checkpoint" \
        "EVALUATION.vjepa2_ac.repo_path=$VJEPA2_REPO" \
        EVALUATION.vjepa2_ac.dtype=float32 \
        EVALUATION.vjepa2_ac.camera.preset=libero_default \
        EVALUATION.vjepa2_ac.camera.position=null \
        EVALUATION.vjepa2_ac.camera.quaternion=null \
        EVALUATION.vjepa2_ac.camera.fovy=null \
        EVALUATION.vjepa2_ac.camera.image_rotation_degrees=180 \
        ++EVALUATION.vjepa2_ac.encoder_image_rotation_degrees=0 \
        "EVALUATION.vjepa2_ac.osc_position_scale=$position_scale" \
        "EVALUATION.vjepa2_ac.osc_rotation_scale=$rotation_scale" \
        EVALUATION.vjepa2_ac.guidance.enabled=true \
        EVALUATION.vjepa2_ac.guidance.trigger.enabled=false \
        EVALUATION.vjepa2_ac.guidance.compile_predictor=false \
        EVALUATION.vjepa2_ac.guidance.overlap_encoder_with_action=true \
        'EVALUATION.vjepa2_ac.guidance.after_flow_steps=[7,8,9]' \
        EVALUATION.vjepa2_ac.guidance.step_size=4.0 \
        EVALUATION.vjepa2_ac.guidance.normalize_gradient=false \
        EVALUATION.vjepa2_ac.guidance.max_delta_rms=0.02 \
        EVALUATION.vjepa2_ac.guidance.ac_steps=2 \
        EVALUATION.vjepa2_ac.guidance.verify_descent=true \
        model.load_text_encoder=true \
        model.redirect_common_files=false \
        model.skip_dit_load_from_pretrain=true \
        "seed=$seed" \
        > "$run_dir/run.log" 2>&1
    local status=$?
    echo "[$(date '+%F %T')] END seed=$seed gpu=$gpu status=$status" | tee -a "$output_root/progress.log"
    return "$status"
}

echo SMOKE_RUNNING > "$output_root/status.txt"
run_seed 7 "$gpu_a" smoke
"$FASTWAM_PYTHON" scripts/verify_calibrated_guidance144_smoke_20261007.py --root "$output_root" \
    > "$output_root/smoke_check.log" 2>&1
# Smoke worker has exited. Recheck availability instead of assuming both cards
# remained free during export/smoke. Formal seeds start fresh, not from smoke.
echo WAITING_TWO_IDLE_GPUS_FOR_FORMAL > "$output_root/status.txt"
until selected=$(pick_two); do
    if (( SECONDS >= deadline )); then exit 75;fi
    sleep 20
done
read -r gpu_a gpu_b <<< "$selected"
echo "$(date '+%F %T %Z') FORMAL_GPU=$gpu_a,$gpu_b SEEDS=$gpu_a:7,17;$gpu_b:27,37" >> "$output_root/gpu_assignment.txt"
echo FULL144_RUNNING > "$output_root/status.txt"
(run_seed 7 "$gpu_a" && run_seed 17 "$gpu_a") & p0=$!
(run_seed 27 "$gpu_b" && run_seed 37 "$gpu_b") & p1=$!
overall_status=0
wait "$p0" || overall_status=1
wait "$p1" || overall_status=1
echo "[$(date '+%F %T')] FINISHED status=$overall_status" | tee -a "$output_root/completion.log"
if (( overall_status == 0 )); then echo COMPLETED > "$output_root/status.txt";fi
exit "$overall_status"
