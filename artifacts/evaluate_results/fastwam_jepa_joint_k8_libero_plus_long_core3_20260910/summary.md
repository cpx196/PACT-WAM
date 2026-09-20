# LIBERO-Plus Long core3 — single-pass joint K=8 JEPA

## Protocol

- Current single-pass implementation: one `infer_joint` call produces one future-video trajectory and eight action candidates; V-JEPA2-AC ranks those candidates.
- Same checkpoint, dataset statistics, LIBERO-Plus task IDs 410/662/457, seeds 7/17/27/37, `sigma_shift=5.0`, future-video generation, and `replan_steps=8` as the aligned K=1 control.
- Physical GPU1 ran seeds 7 and 27; physical GPU3 ran seeds 17 and 37. At most two evaluation workers ran concurrently.

## Results

| Plus task ID | Seed 7 | Seed 17 | Seed 27 | Seed 37 | Aligned K=1 | Old two-call JEPA | New joint K=8 JEPA |
|---:|:---:|:---:|:---:|:---:|---:|---:|---:|
| 410 | success | failure | failure | failure | 2/4 | 3/4 | **1/4** |
| 662 | failure | failure | failure | failure | 0/4 | 0/4 | **0/4** |
| 457 | failure | failure | failure | failure | 0/4 | 1/4 | **0/4** |
| **Overall** | — | — | — | — | **2/12 (16.7%)** | **4/12 (33.3%)** | **1/12 (8.3%)** |

Relative to aligned K=1, the new treatment has zero rescues and one regression (task 410/seed 17), for -1/12 or -8.3 percentage points. Relative to old two-call JEPA, it loses three successes: task 410/seeds 17 and 27, and task 457/seed 17.

Mean future-video PSNR across the 12 new rollouts is **26.9725 dB**. Integrity checks found 12 result JSONs, 12 rollout videos, 12 aggregate `replan=all` future-video comparisons, and 1006 per-replan future-video comparison clips.

## Why the policy changed

K=8 denotes eight independently initialized action latent trajectories, not eight video trajectories and not eight serial Python calls. The video branch remains batch one; its per-layer K/V is expanded as a view for the action batch.

The old implementation generated the reference future with `infer_joint(K=1)`, then generated candidates with a separate `infer_action(K=8)` current-frame-cache pass. The new implementation advances one video trajectory and eight action trajectories together through every flow step. The attention mask still restricts action queries to first-frame video tokens, so these paths are mathematically intended to be close. They are not bitwise identical: the new path computes first-frame video states inside a full masked video attention kernel and recomputes them during joint flow, whereas the old path prefills a standalone first-frame cache. Different kernel shapes and floating-point accumulation produce small changes that compound through 30 transformer layers and ten flow steps.

The selector is particularly sensitive to those changes. At the first task-410 replan, old and new implementations chose different candidate IDs for three of four seeds. Many top-two V-JEPA energy gaps are around `1e-5` to `1e-4`, so small action/video numerical changes can flip the argmin. Once a different eight-action chunk executes, the closed-loop observations diverge and later replans are no longer directly comparable.

This 12-rollout stress slice shows that the current latency optimization is not behavior preserving. It does not establish a general expected accuracy difference; a direct same-observation candidate-equivalence probe should precede further accuracy scaling.

## Decision

For the current backbone, production/evaluation defaults back to the best observed old two-call treatment: `infer_joint(K=1)` for the reference future followed by the released `infer_action(K=8)` candidate path. The single-pass joint K=8 implementation remains available only through `candidate_generation_mode=joint` for controlled experiments.

The planned replacement backbone will use a fully cascaded architecture: a video-only first stage followed by an action stage. In that design, future generation no longer emits an unused joint action, so retaining separate reference-video and candidate-action stages does not waste the K=1 action trajectory present in the current `infer_joint` API.

Machine-readable paired outcomes are in `comparison.csv`.
