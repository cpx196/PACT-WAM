# LIBERO-10 robot initial-pose OOD: seven-task three-arm results

The single-GPU study completed all 813 episodes at 2026-11-02 04:26 (Asia/Shanghai), after approximately 37.5 hours. Seven base tasks cover 271 official LIBERO-plus Robot Initial States versions. Each version runs once with seed7 in each of three arms. This is not a multi-seed study.

| Base task | Versions | No guidance | Unweighted guidance | Weighted guidance |
|---|---:|---:|---:|---:|
| Turn on stove, place moka pot | 44 | 43 | 43 | 43 |
| Black bowl into bottom drawer, close drawer | 44 | 39 | 39 | 39 |
| Yellow/white mug into microwave, close door | 42 | 27 | 28 | 27 |
| Both moka pots onto stove | 34 | 23 | 23 | 23 |
| Cream cheese and butter into basket | 35 | 32 | 30 | 31 |
| White mug left plate, yellow/white mug right plate | 38 | 33 | 31 | 31 |
| Book into caddy back compartment | 34 | 27 | 27 | 26 |
| Total | 271 | **224 (82.7%)** | **221 (81.5%)** | **220 (81.2%)** |

Weighted guidance rescues 11 baseline failures but introduces 15 failures on baseline successes: net -4 successful episodes. Equal per-task totals sometimes hide changes in which versions succeed. The results do not support a broad success-rate gain from this fixed weighting method.

## Earlier tasks and scope of combined comparisons

| Base task | Versions | No guidance | Unweighted guidance | Weighted guidance |
|---|---:|---:|---:|---:|
| White mug and chocolate pudding | 42 | 30 | 30 | 34 |
| Alphabet soup and tomato sauce into basket | 41 | 31 | Pending | 34 |
| Alphabet soup and cream cheese into basket | 39 | Pending | Pending | Pending |

Across the eight tasks with all three groups already available (313 versions), results are 254/313 control, 251/313 unweighted, 254/313 weighted. Across nine tasks with control and weighted available (354 versions), results are 285/354 versus 288/354 (+3 episodes, +0.85 percentage points). These ranges must not be conflated.

The missing four groups (41 + 39*3 = 158 episodes) were queued at 2026-11-02 10:43. No result is asserted for them. The launcher polls all GPUs every five seconds and requires no active compute process, at least 22,500 MiB free, and utilization below 10%. Once selected, a single physical GPU is retained for serial execution. Final aggregation targets ten tasks, 393 versions and 1,179 episodes. Current live status requires reading the server; queued status is the last confirmed snapshot, not a completion claim.

## Fixed evaluation contract and checks

- Plan 32 actions, replan after 8, 10 denoising steps, at most 700 controls. Guidance nodes 7/8/9, scale 4, delta RMS cap 0.02, original AR2 gradient chain and L1 energy retained.
- GPU-encoded batch1 T5 contexts are saved once per task and shared immutably across arms. Previously frozen overlapping prompts are preserved. T5 is unloaded before WAM/JEPA transfer.
- The native action function is reset between arms; no stale recording wrapper is carried into baseline. Formal first actions must equal their own preflight actions. Per-version observations/native actions are checked across arms; gripper inputs match execution and gripper gradients remain zero.
- All 813 preflights passed. Native full32 hashes matched across arms for all 271 versions; fixed weight hashes matched. The complete per-episode records and audit are in `final_three_arm_audit.json`.
- Offline weight construction for the other seven tasks completed on October31 at 14:20. Combined with the three earlier tables, ten task weights are available. Each uses only 45 base-task training demonstrations, 200 states, and 32 action chunks per state; shape [2,576,1024]. Evaluation perturbations are not used for calibration.

The early six-case diagnostic selection identifies particular imagination/execution mismatch cases, not every perturbation of the three corresponding base tasks. Do not infer that all expanded versions share that failure mechanism.
