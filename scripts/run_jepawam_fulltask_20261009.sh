#!/bin/bash
set -euo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=4 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=4
export LIBERO_PATH=/data/pxchen/LIBERO-plus
export LIBERO_CONFIG_PATH=/data/pxchen/PACT-WAM/artifacts/jepawam_deploy_smoke_20261009/libero_config
export PYTHONPATH=/data/pxchen/PACT-WAM/artifacts/jepawam_deploy_smoke_20261009/python_deps:/data/pxchen/JEPA_WAM:/data/pxchen/LIBERO-plus
export TOKENIZERS_PARALLELISM=false NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR=/data/pxchen/PACT-WAM/artifacts/jepawam_deploy_smoke_20261009/matplotlib
asset_root=/data/pxchen/PACT-WAM/checkpoints/jepa_wam_assets_20261009
output_root=/data/pxchen/PACT-WAM/artifacts/jepawam_deploy_smoke_20261009
mkdir -p "$MPLCONFIGDIR"
cd /data/pxchen/JEPA_WAM
status=0
/data/pxchen/envs/jepa-wam/bin/python -u experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "$asset_root/JEPA_WAM/checkpoints/libero/jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90+n1+b32+x7--visual-cosine-projector-allviews--20260723_232305/checkpoints/step-040000-epoch-37-loss=0.0262.pt" \
  --base_vlm "$asset_root/JEPA_WAM/checkpoints/pretrained_vlm/prism-qwen25-vjepa21-vitl-384px+0_5b+stage-finetune+x7" \
  --llm_checkpoint_path "$asset_root/Qwen2.5-0.5B" \
  --vjepa_checkpoint_path "$asset_root/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt" \
  --task_suite_name libero_10 --libero_plus_categories camera --num_trials_per_task 1 \
  --max_tasks 1 --max_episode_steps 520 --save_rollouts True --seed 7 \
  --local_log_dir "$output_root/eval_logs" || status=$?
printf '%s\n' "$status" > "$output_root/exit_code.txt"
exit "$status"
