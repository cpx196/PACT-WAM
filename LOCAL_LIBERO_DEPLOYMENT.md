# Local LIBERO deployment

This deployment uses the existing local runtime instead of creating another
Python environment:

- Python/PyTorch: `/data/chenpengxu/conda_envs/HMoE`
- FastWAM-compatible site packages: `/data/chenpengxu/jepa_wam_py312`
- LIBERO source and assets: `/data/chenpengxu/LIBERO`
- FastWAM source: `/data/chenpengxu/FastWAM`

Initialize the runtime variables with:

```bash
cd /data/chenpengxu/FastWAM
source scripts/local_libero_env.sh
```

The released LIBERO checkpoint is expected at:

```text
checkpoints/fastwam_release/libero_uncond_2cam224.pt
checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
```

The server must first pass the HIT Shenzhen campus-network admission page.
From a local machine, create a temporary SOCKS tunnel:

```bash
ssh -F /dev/null -N -D 127.0.0.1:1080 \
  -o User=chenpengxu 10.249.186.201
```

Configure an isolated browser profile to use SOCKS5 at `127.0.0.1:1080`, open
`https://net.hitsz.edu.cn/srun_portal_success?ac_id=1&theme=basic4`, and log in.
After admission succeeds, close the temporary tunnel. The download script uses
`hf-mirror.com` directly and deliberately clears all proxy environment variables.

Resume and validate all required model files with:

```bash
FASTWAM_DOWNLOAD_WORKERS=4 scripts/complete_local_libero_downloads.sh
```

Run one LIBERO task as a smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
  scripts/run_libero_local.sh \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=1 \
  MULTIRUN.num_gpus=1
```

For same-GPU FastWAM + V-JEPA2-AC ranking, T5 prompt embeddings are built
first and cached on CPU; T5 is then destroyed before either inference model is
loaded.  Each worker therefore needs only one visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
  scripts/run_libero_local.sh \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=1 \
  EVALUATION.visualize_future_video=true \
  EVALUATION.vjepa2_ac.enabled=true
```

The verified RTX 4090 memory figures are approximately 10.61 GiB for the
temporary T5 stage, 17.45 GiB after FastWAM plus FP32 V-JEPA2-AC are loaded,
and 18.57 GiB peak during the first Best-of-8 inference.

The ranker keeps a real observation from one FastWAM video interval earlier.
At each replan it encodes `[real(t-4), real(t)]` as the current two-frame clip,
then scores the JEPA rollout against all overlapping FastWAM future clips:
`[V0,V1]`, `[V1,V2]`, and so on.

Before V-JEPA2-AC scoring, normalized FastWAM/Robosuite OSC commands are
clipped to `[-1,1]` and converted to metric EEF deltas using the controller's
0.05 m position scale and 0.5 rad axis-angle scale.  The adapter then composes
four scaled low-level commands into each JEPA action step.

For the complete four-suite benchmark, omit the task-specific suite/id
overrides and use `MULTIRUN.num_gpus=4` on this machine.
