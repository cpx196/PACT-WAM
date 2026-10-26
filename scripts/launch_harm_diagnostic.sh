#!/usr/bin/env bash
set -uo pipefail
cd /data/pxchen/PACT-WAM
source scripts/local_libero_env.sh
set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
root=/data/pxchen/PACT-WAM/evaluate_results/harm_diagnostic_20260928
tasks=${TASK_FILE_OVERRIDE:-$root/tasks.txt}
jepa=/data/pxchen/PACT-WAM/checkpoints/vjepa2-ac-libero90-rot180-step2000-inference.pt
test "$(wc -l < "$tasks")" -eq "${EXPECTED_COUNT:-72}" || exit 2
exec 9> "$root/launcher${ONLY_GROUP:+_$ONLY_GROUP}.lock"
flock -n 9 || exit 3
echo $$ > "$root/launcher${ONLY_GROUP:+_$ONLY_GROUP}.pid"
run_group() {
    local group=$1 gpu=$2 multiplier=$3 protect=$4
    local output=$root/$group
    mkdir -p "$output"
    echo "[$(date '+%F %T')] START $group GPU $gpu" >> "$root/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" \
    "$FASTWAM_PYTHON" experiments/libero/run_harm_diagnostic_manager.py \
        task=libero_optional_idm_2cam224_1e-4 "ckpt=$FASTWAM_CHECKPOINT" \
        "EVALUATION.dataset_stats_path=$FASTWAM_DATASET_STATS" \
        "EVALUATION.output_dir=$output" "MULTIRUN.task_file=$tasks" MULTIRUN.num_gpus=1 \
        EVALUATION.num_trials=1 EVALUATION.replan_steps=8 EVALUATION.num_inference_steps=10 \
        EVALUATION.num_steps_wait=30 EVALUATION.max_steps_override=700 \
        EVALUATION.compile_action_infer=false EVALUATION.visualize_future_video=true \
        EVALUATION.offload_text_encoder=true EVALUATION.action_infer_mode=idm \
        EVALUATION.video_sigma_shift=1.0 EVALUATION.action_sigma_shift=1.0 EVALUATION.sigma_shift=null \
        EVALUATION.trigger_probe.enabled=false EVALUATION.vjepa2_ac.enabled=true \
        EVALUATION.vjepa2_ac.num_action_candidates=8 EVALUATION.vjepa2_ac.replan_steps=8 \
        EVALUATION.vjepa2_ac.candidate_generation_mode=separate \
        "EVALUATION.vjepa2_ac.checkpoint_path=$jepa" "EVALUATION.vjepa2_ac.repo_path=$VJEPA2_REPO" \
        EVALUATION.vjepa2_ac.dtype=float32 EVALUATION.vjepa2_ac.camera.preset=libero_default \
        EVALUATION.vjepa2_ac.camera.position=null EVALUATION.vjepa2_ac.camera.quaternion=null \
        EVALUATION.vjepa2_ac.camera.fovy=null EVALUATION.vjepa2_ac.camera.image_rotation_degrees=180 \
        +EVALUATION.vjepa2_ac.encoder_image_rotation_degrees=0 \
        EVALUATION.vjepa2_ac.guidance.enabled=true \
        EVALUATION.vjepa2_ac.guidance.overlap_encoder_with_action=true \
        'EVALUATION.vjepa2_ac.guidance.after_flow_steps=[7,8,9]' \
        "EVALUATION.vjepa2_ac.guidance.step_size=$multiplier" EVALUATION.vjepa2_ac.guidance.normalize_gradient=false \
        EVALUATION.vjepa2_ac.guidance.max_delta_rms=0.02 EVALUATION.vjepa2_ac.guidance.ac_steps=2 \
        EVALUATION.vjepa2_ac.guidance.verify_descent=true EVALUATION.vjepa2_ac.guidance.trigger.enabled=false \
        "+EVALUATION.diagnostic.protect_gripper=$protect" +EVALUATION.diagnostic.paired_every=4 \
        "+EVALUATION.diagnostic.checkpoint_compat_state=${CHECKPOINT_COMPAT_STATE:-false}" \
        model.load_text_encoder=true model.redirect_common_files=false model.skip_dit_load_from_pretrain=true \
        seed=7 > "$output/run.log" 2>&1
    local status=$?
    echo "[$(date '+%F %T')] END $group status=$status" >> "$root/progress.log"
    return "$status"
}
if [[ -n "${ONLY_GROUP:-}" ]]; then
    run_group "$ONLY_GROUP" "${TEST_GPU:-7}" "${TEST_MULTIPLIER:-4.0}" "${PROTECT_GRIPPER:-false}"
    exit "$?"
fi
run_group guidance1x 3 1.0 false & a=$!
run_group guidance2x 4 2.0 false & b=$!
run_group guidance4x_protected_gripper 7 4.0 true & c=$!
status=0
wait "$a" || status=1
wait "$b" || status=1
wait "$c" || status=1
echo "[$(date '+%F %T')] FINISHED status=$status" >> "$root/progress.log"
exit "$status"
