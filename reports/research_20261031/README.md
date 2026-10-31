# Research snapshot, 2026-10-31

This update publishes the current server source and lightweight experimental evidence. It does not claim a validated decoder-sharding optimization.

## Completed perturbation experiments

LIBERO-plus robot initial-state perturbations, seed 7, one episode per version. Within each version, arms reuse the same T5 context.

| Base task | Versions | No guidance | Unweighted guidance | Weighted guidance |
|---|---:|---:|---:|---:|
| White mug and chocolate pudding | 42 | 30/42 (71.4%) | 30/42 (71.4%) | 34/42 (81.0%) |
| Alphabet soup and tomato sauce into basket | 41 | 31/41 (75.6%) | Not evaluated | 34/41 (82.9%) |

For soup/tomato, the original baseline attempt stopped before formal episodes because a stale guidance wrapper invalidated its preflight. The repaired baseline completed all 41 episodes on October 29. Weighted guidance rescued versions 302, 322 and 329, with no baseline-success/weighted-failure reversals. These are two base tasks under one seed, not evidence of broad cross-seed generalization.

Protocol: plan 32 actions, execute 8 before replanning, 10 denoising steps, at most 700 control steps. AR2 guidance at configured nodes 7/8/9, scale 4.0, delta RMS cap 0.02, execution-consistent gripper handling. Formal success-rate evaluations use eager execution and must not be conflated with compiled component benchmarks.

Offline weights use 45 training demonstrations from each corresponding base LIBERO-Long task, 200 states and 32 demonstrated 8-step action chunks per state. Per-element prediction variance is averaged across states and normalized separately for the two prediction horizons. Shape: [2, 576, 1024]. One fixed task-specific table is shared across that task's perturbation versions. Perturbed evaluation trajectories are not used to estimate these weights.

## Timing and contention

The October 27 steady-state reference is approximately 562 ms without guidance and 868 ms with guidance. October 28 profiler-instrumented measurements are a separate measurement condition and must not be substituted into the same timeline.

Hardware records identify some heavy decoder convolution blocks with 254 registers/thread, 256 threads/block and 96 KiB shared memory/block. The simple register product is 65,024 before allocation-granularity rounding. These footprints constrain same-SM co-residency; they do not establish that register/shared-memory contention is the only source of slowdown. Both action denoising and decoder convolution include Tensor Core kernels. Exact convolution-kernel-to-module attribution and sharding benefit remain unverified.

## Evidence

- Original weighted soup run: `evaluate_results/soup_tomato_robot41_seed7_sameT5_20261028/weighted/summary.json` under this report directory.
- Repaired control: `evaluate_results/soup_tomato_robot41_baseline_repair_20261029/control/summary.json`.
- Async timing and hardware samples are grouped by their original run directory under `evaluate_results/`.

The main runtime checkout and its Git index are preserved. Checkpoints, full tensor caches, profiler binaries, service environment snapshots and credentials are outside this update.
