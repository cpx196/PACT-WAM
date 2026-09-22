# Raw-gradient JEPA guidance report (2026-09-22)

## Purpose

This diagnostic keeps the PACT premise fixed: the WAM imagined future is the target and
V-JEPA2-AC loss measures whether the proposed action is consistent with that target. It
tests the correction operator, specifically whether forcing every gradient to RMS 0.02
overwrites useful differences in local sensitivity.

## Shared protocol

- Checkpoint: `checkpoints/fastwam/libero_optional_idm_2cam224.pt`
- Inference: strict `infer_video() -> infer_action_from_video()` cascade
- Action mode: Optional-IDM
- Video/action sigma shift: `1.0 / 1.0`; legacy `sigma_shift=null`
- Flow steps: 10; guidance after steps `[7, 8, 9]`
- Closed-loop replan: 8 low-level actions
- JEPA conditioning: 2 transitions (`ac_steps=2`)
- JEPA dtype: FP32
- Frozen JEPA encoder overlaps ActionDiT denoising on a separate CUDA stream
- `verify_descent=true` records an additional clean-action candidate forward and never
  rejects a correction

## Correction definitions

Historical fixed-RMS guidance uses:

```yaml
normalize_gradient: true
step_size: 0.02
max_delta_rms: null
```

The raw-gradient diagnostic uses:

```yaml
normalize_gradient: false
step_size: 1.0
max_delta_rms: 0.02
```

Thus 0.02 changes from a mandatory correction RMS to a maximum trust-region radius.

## Matched 64-pair fixed-RMS result

The exact seed/task split, including the 24 control failures and 40 control successes, is
stored in [`task_lists/strict_shift1_matched64_20260922.yaml`](./task_lists/strict_shift1_matched64_20260922.yaml).

| Method | Success | Rescue | Harm | Both success | Both fail |
|---|---:|---:|---:|---:|---:|
| No guidance | 40/64 (62.5%) | - | - | - | - |
| Fixed RMS 0.02 | 41/64 (64.1%) | 5 | 4 | 36 | 19 |

The clean-action candidate loss decreased in `9654/9939` corrections (97.13%). A high
descent rate did not by itself imply a large closed-loop gain.

## Raw-gradient result on the 24 control failures

The failure-only subset contains every no-guidance failure from the matched 64 pairs.

| Method | Rescued failures |
|---|---:|
| Fixed RMS 0.02 | 5/24 (20.8%) |
| Raw gradient x1, cap 0.02 | 3/24 (12.5%) |

Raw-gradient rescues:

- `seed17/task195`
- `seed27/task1960`
- `seed37/task411`

Fixed-RMS-only rescues:

- `seed7/task35`
- `seed37/task2128`

Raw-gradient diagnostics:

- Corrections: 5,871
- Candidate-loss decreases: 5,850 (99.64%)
- Median correction RMS: 0.00137
- Corrections clipped by the 0.02 cap: 5
- `seed37/task411`: three clipped denoise steps at one planning point; rollout rescued
- `seed17/task158`: two clipped denoise steps; rollout still failed in control,
  fixed-RMS, and raw-gradient variants

The raw-gradient rescues are a subset of the fixed-RMS rescues. This subset alone measures
rescue but cannot measure harm because every control rollout in it failed.

## Required retention evaluation

The remaining 40 matched pairs are no-guidance successes. Running raw-gradient guidance on
them measures whether the smaller local correction preserves successful FastWAM actions.
Only after combining those 40 retention outcomes with the 24 failure outcomes is it valid
to report a complete raw-gradient success rate, rescue count, harm count, and net gain on
the 64-pair diagnostic set.

## Reproduction

Use [`scripts/run_jepa_guidance_raw_gradient.sh`](../../scripts/run_jepa_guidance_raw_gradient.sh)
with an explicit task list, GPU, seed, and output directory. Rollout artifacts belong under
`evaluate_results/` and remain excluded from Git.
