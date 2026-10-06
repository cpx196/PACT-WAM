#!/usr/bin/env bash
set -uo pipefail

cd /data/pxchen/PACT-WAM
source scripts/local_libero_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

output_root=/data/pxchen/PACT-WAM/evaluate_results/guidance789_rawgrad4x_long36_full144_libero90_rot180_step2000_20260925
task_file=/data/pxchen/PACT-WAM/artifacts/evaluate_results/long36_ten_vs_789_20260916_231154_2gpu/ten_guidance_789/seed_7/libero_plus_long_36_three_perturbations_two_variants.txt
jepa_checkpoint=/data/pxchen/PACT-WAM/checkpoints/vjepa2-ac-libero90-rot180-step2000-inference.pt
gpu_a=${GPU_A:-1}
gpu_b=${GPU_B:-2}
gpu_c=${GPU_C:-3}

test -s "$FASTWAM_CHECKPOINT" || exit 2
test -s "$jepa_checkpoint" || exit 2
test "$(wc -l < "$task_file")" -eq 36 || exit 2
mkdir -p "$output_root"
echo $$ > "$output_root/launcher.pid"
printf 'fastwam_checkpoint=%s\njepa_checkpoint=%s\ntrained_frame_rotation_degrees=180\nfastwam_input_rotation_degrees=180\nadditional_online_encoder_rotation_degrees=0\ngpus=%s,%s,%s\n' \
    "$FASTWAM_CHECKPOINT" "$jepa_checkpoint" "$gpu_a" "$gpu_b" "$gpu_c" > "$output_root/run_config.txt"

run_seed() {
    local seed=$1
    local gpu=$2
    local run_dir="$output_root/seed_${seed}"
    mkdir -p "$run_dir"
    echo "[$(date '+%F %T')] START seed=$seed gpu=$gpu" | tee -a "$output_root/progress.log"
    CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" \
        "$FASTWAM_PYTHON" experiments/libero/run_libero_manager.py \
        task=libero_optional_idm_2cam224_1e-4 \
        "ckpt=$FASTWAM_CHECKPOINT" \
        "EVALUATION.dataset_stats_path=$FASTWAM_DATASET_STATS" \
        "EVALUATION.output_dir=$run_dir" \
        "MULTIRUN.task_file=$task_file" \
        MULTIRUN.num_gpus=1 \
        EVALUATION.num_trials=1 \
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
        +EVALUATION.vjepa2_ac.encoder_image_rotation_degrees=0 \
        EVALUATION.vjepa2_ac.guidance.enabled=true \
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

run_seed 7 "$gpu_a" & p0=$!
(run_seed 17 "$gpu_b" && run_seed 37 "$gpu_b") & p1=$!
run_seed 27 "$gpu_c" & p2=$!
overall_status=0
wait "$p0" || overall_status=1
wait "$p1" || overall_status=1
wait "$p2" || overall_status=1
echo "[$(date '+%F %T')] FINISHED status=$overall_status" | tee -a "$output_root/completion.log"
exit "$overall_status"
