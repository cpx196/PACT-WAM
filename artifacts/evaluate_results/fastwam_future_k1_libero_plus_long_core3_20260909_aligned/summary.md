# LIBERO-Plus Long core3 — future-video-aligned K=1 baseline

## Protocol

- Checkpoint: `checkpoints/fastwam_release/libero_uncond_2cam224.pt`
- Dataset statistics: `checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json`
- Tasks: LIBERO-Plus `libero_10` IDs 410, 662, and 457
- Seeds: 7, 17, 27, and 37; one rollout for every task/seed pair
- `sigma_shift=5.0`, `visualize_future_video=true`, `replan_steps=8`
- V-JEPA2-AC disabled. Consequently `infer_joint` uses its default single action candidate (effective K=1); the inactive `vjepa2_ac.num_action_candidates` config value is ignored.

This is the valid K=1 accuracy control for the historical JEPA Best-of-8 run. Unlike the earlier action-only screening baseline, both arms generate future video and use the same eight-action replan interval.

## Results

| Plus task ID | Seed 7 | Seed 17 | Seed 27 | Seed 37 | Aligned K=1 | Historical two-call JEPA | Current joint K=8 JEPA |
|---:|:---:|:---:|:---:|:---:|---:|---:|---:|
| 410 | success | success | failure | failure | 2/4 | 3/4 | 1/4 |
| 662 | failure | failure | failure | failure | 0/4 | 0/4 | 0/4 |
| 457 | failure | failure | failure | failure | 0/4 | 1/4 | 0/4 |
| **Overall** | — | — | — | — | **2/12 (16.7%)** | **4/12 (33.3%)** | **1/12 (8.3%)** |

The paired historical JEPA comparison contains two rescues and no regressions: task 410/seed 27 and task 457/seed 17. The observed difference is **+2/12, or +16.7 percentage points**.

Mean future-video PSNR over the 12 K=1 rollouts is **26.6452 dB**. All 12 result JSONs have one rollout video, one aggregate `replan=all` future-video comparison, and per-replan future-video comparisons (957 clips in total).

## Interpretation boundary

The old action-only result, 1/12, is retained only as a deployment reference. It is not the accuracy denominator for JEPA because it used `visualize_future_video=false` and `replan_steps=10`.

The historical JEPA column comes from `evaluate_results/fastwam_jepa_libero_plus_long_core3_20260909/`. The current single-pass result comes from `evaluate_results/fastwam_jepa_joint_k8_libero_plus_long_core3_20260910/`; it has zero rescues and one regression relative to this K=1 control. The single-pass optimization is therefore not behavior preserving on this slice.

Detailed paired rows are in `comparison_with_historical_jepa.csv`; self-contained per-seed configurations, summaries, JSON results, rollout videos, and predicted-future videos are under `seed_{7,17,27,37}/`.
