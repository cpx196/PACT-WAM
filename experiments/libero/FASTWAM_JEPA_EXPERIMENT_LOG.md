# FastWAM–JEPA Experiment Log

This append-only log records measurements for the local FastWAM–JEPA closed-loop implementation. Keep cold-start, compilation, steady-state inference, and complete closed-loop latency separate: they answer different questions.

## 2026-09-09 — Single-pass joint Best-of-8 latency

### Question

Can the separate `infer_action(K=8)` call be removed by generating one shared future-video trajectory and eight action trajectories directly inside `infer_joint`?

### Implementation

- `infer_joint` now accepts `num_action_candidates`.
- The video branch remains batch size one. At every MoT layer, its K/V tensors are expanded as views for the batch of action queries, so the approximately 5B-parameter video expert is not evaluated eight times.
- The action branch is evaluated as a single K=8 batch during the same ten flow steps as video generation.
- The LIBERO JEPA path now calls `infer_joint(K=8)` once and no longer follows it with a separate `infer_action(K=8)` call.
- This changes candidate generation from the old standalone current-frame action flow to joint action/video flow. Latency is directly comparable, but policy success requires a new paired rollout evaluation.

### Old implementation versus new implementation

The old implementation and the optimized implementation both produce eight candidates as one GPU batch; the old implementation did not make eight serial Python calls. Their difference is where that batch is generated:

| Implementation | FastWAM execution path | Video branch batch | Action branch batch | Separate FastWAM action call |
|---|---|---:|---:|---:|
| Future-video K=1 control | `infer_joint(K=1)` | 1 | 1 | No |
| Old Best-of-8 + JEPA | `infer_joint(K=1)` -> `infer_action(K=8)` -> JEPA | 1, then current-frame cache prefill | 1, then 8 | **Yes** |
| New Best-of-8 + JEPA | `infer_joint(video B=1, action B=8)` -> JEPA | **1** | **8** | **No** |

In the old path, the 179.1 ms measurement was the total latency of one additional batched `infer_action(K=8)` call, not the marginal cost of changing an existing call from K=1 to K=8. That additional call did not rerun future-video denoising in the released base FastWAM model; it did, however, repeat input-image encoding/current-frame visual-cache preparation and run a separate ten-step action flow. The new mixed-batch joint path keeps one video trajectory and shares its per-layer K/V with eight action trajectories, eliminating the second FastWAM call.

### Setup

The setup matches the earlier end-to-end latency probe: released two-camera 224x224 FastWAM checkpoint, colocated FP32 V-JEPA2-AC on one otherwise-idle RTX 4090, LIBERO Spatial task 0 with seed 7, four initial no-op steps, 24 controlled steps, three replans, Best-of-8, ten flow steps, and CUDA synchronization around measured stages. Replan 0 is cold compile and replans 1–2 form the steady-state sample.

Result artifact: `evaluate_results/latency_probe_joint_k8_20260909/libero_spatial/gpu0_task0_results.json`.

### Results

| Replan | Total | Joint future + K=8 action | Separate action call | JEPA rank | Encoder phase | Predictor, 2 steps |
|---:|---:|---:|---:|---:|---:|---:|
| 0, cold compile | 113.967 s | 113.623 s | 0 | 332.5 ms | 130.5 ms | 198.1 ms |
| 1, warm | 860.6 ms | 566.0 ms | 0 | 285.1 ms | 95.0 ms | 189.6 ms |
| 2, warm | 859.8 ms | 566.4 ms | 0 | 284.7 ms | 94.3 ms | 190.0 ms |
| **warm mean** | **860.2 ms** | **566.2 ms** | **0** | **284.9 ms** | **94.7 ms** | **189.8 ms** |

Compared with the previous two-call warm mean of 965.3 ms, single-pass joint Best-of-8 saves **105.1 ms/replan (10.9%)**. At an eight-action replan interval, latency falls from 120.7 to **107.5 ms per low-level action**. Against the future-video K=1 control at 498.0 ms, the optimized selector-system increment is 362.2 ms/replan rather than 467.3 ms.

### Canonical complete-system comparison after optimization

JEPA ranking is included in both Best-of-8 rows below. The K=1 control has no meaningful selection problem and therefore does not run JEPA; future-video generation is enabled in all three rows so the baseline is aligned.

| Future-video-aligned system | Warm latency / replan | Versus K=1 | Latency / low-level action |
|---|---:|---:|---:|
| FastWAM K=1, no JEPA | **498.0 ms** | baseline | **62.3 ms** |
| Old FastWAM Best-of-8 + JEPA | **965.3 ms** | +467.3 ms, **1.94x** | **120.7 ms** |
| New single-pass Best-of-8 + JEPA | **860.2 ms** | +362.2 ms, **1.73x** | **107.5 ms** |

The optimized complete FastWAM–JEPA system therefore remains 362.2 ms/replan slower than the future-video K=1 baseline, but recovers 105.1 ms/replan from the old Best-of-8 implementation. The optimized 362.2 ms increment consists approximately of 68.2 ms additional joint FastWAM work for K=8, 284.9 ms JEPA ranking, and 9.1 ms preprocessing/adaptation/bookkeeping.

This section supersedes the older "Canonical end-to-end reporting result" below for the active implementation. The older 965.3 ms result is retained as the historical two-call measurement.

The cold compiled graph took longer than the former two separate compiled graphs (113.6 s for the joint K=8 stage versus 76.9 s combined previously). This is startup cost and is excluded from the warm comparison.

After rollout, peak allocated memory was 18.566 GiB and reserved memory was 19.041 GiB, versus 18.671 GiB and 19.494 GiB in the previous two-call probe. The reduction in reserved memory is allocator/compiled-graph-pool behavior; it should not be interpreted as an equivalent reduction in live tensor storage.

## Entry template

```text
Date / experiment:
Question:
Code and checkpoint:
Hardware and software:
Configuration:
Warm-up and repetitions:
Measurements:
Interpretation:
Limitations / next measurement:
```

## 2026-09-09 — Action flow candidate-batch latency

### Question

How much steady-state latency is added when FastWAM action sampling changes from one candidate to the current JEPA Best-of-8 candidate batch?

### Setup

- Checkpoint: `checkpoints/fastwam_release/libero_uncond_2cam224.pt` (approximately 12 GB).
- Model: released LIBERO two-camera FastWAM; approximately 6B parameters across the video and action experts.
- Hardware: NVIDIA GeForce RTX 4090, 24 GiB. K=1 ran on physical GPU 2 and K=8 on physical GPU 3; both were otherwise idle.
- Precision: BF16.
- Input: synthetic task-shaped 224x224 observation, cached/random text context, proprioception enabled.
- Action tensor per candidate: `[32, 7]`.
- Flow integration: 10 Euler steps.
- Inference path: `infer_action`, `torch.no_grad`, `compile_action_infer=true`.
- Timing: CUDA synchronization immediately before and after each call.
- Protocol: 3 warm-up calls followed by 12 measured calls. The first compilation call is excluded from steady-state statistics.
- Reproduction support: `scripts/dryrun_fastwam.py` accepts `+DRYRUN.num_action_candidates=<K>`.

### Results

| Candidates | Output shape | Mean | Median | Min | Max | Reported peak allocated |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `[32, 7]` | 88.1 ms | 88.0 ms | 87.6 ms | 89.0 ms | 12.758 GiB |
| 8 | `[8, 32, 7]` | 143.0 ms | 143.0 ms | 142.6 ms | 143.5 ms | 12.758 GiB |

Derived comparisons:

- K=8 versus K=1: +54.9 ms per action replanning event, +62.3%, or 1.62x latency.
- Eight serial K=1 calls would take approximately 704.8 ms. The K=8 batch is approximately 4.93x faster than serial generation.
- With eight executed low-level actions per JEPA replanning event, the added action-flow cost is approximately 6.9 ms per executed environment step.

### Cold-start observations

- The first K=1 run encountered a cold filesystem/page cache. Model construction plus checkpoint loading took several minutes; its first compiled inference call took approximately 68.0 s.
- The later K=8 run reused filesystem page cache populated by the earlier 12 GB checkpoint read. It reached inference in approximately 57 s; its first compiled inference call took approximately 41.9 s.
- These numbers are startup observations, not online policy latency. The shorter second run does not imply K=8 loads faster than K=1. It reflects warm host page cache and likely partial reuse of compiler artifacts.

### Memory interpretation

The equal `12.758 GiB` values do not establish that K=8 has exactly zero incremental memory cost:

- Model weights and persistent compiled buffers dominate allocated memory.
- The initial normalized action latent grows only from `1 x 32 x 7` to `8 x 32 x 7`; in BF16 the raw increase is only 3,136 bytes.
- The current-frame visual KV cache is shared with candidates using PyTorch `expand`, which creates a view rather than eight physical copies.
- Inference uses `torch.no_grad`, so it does not retain per-layer activations for backward.
- The logger prints GiB to three decimals, approximately 1 MiB resolution. Small transient differences can therefore be hidden.
- Peak statistics were reset after warm-up, when compiled graph pools were already resident. The reported number is consequently best interpreted as total steady resident allocation, not a sensitive measurement of candidate-batch incremental memory.

### Scope and limitations

This entry measures only batched action flow. Complete FastWAM–JEPA replanning additionally runs `infer_joint` to generate the reference future and runs V-JEPA2-AC to rank candidates. Those stages must be timed separately before reporting complete closed-loop latency.

## 2026-09-09 — End-to-end JEPA rollout latency and memory

### Question

What is the actual FastWAM–JEPA closed-loop replan latency, how much time is spent in the V-JEPA2-AC encoder and predictor, and what is the resident GPU memory cost of enabling JEPA?

### Setup

- Checkpoint and GPU: the released LIBERO FastWAM checkpoint on one otherwise-idle RTX 4090 (24 GiB), with V-JEPA2-AC FP32 colocated on the same GPU.
- Policy: FastWAM future-video `infer_joint`, Best-of-8 action `infer_action`, then V-JEPA2-AC ranking of the first two AC steps.
- Flow settings: 32x7 action chunk, 10 Euler steps, replan every 8 low-level actions.
- Environment: one real LIBERO `libero_spatial` task-0 episode, seed 7. The latency probe executes 24 policy-controlled actions after 4 initial no-op steps, yielding three replans. It is a latency sample, not a success-rate evaluation.
- Timing: `EVALUATION.timing_enabled=true`; CUDA is synchronized around every measured GPU stage. The first replan includes `torch.compile` cold-start; replans 1 and 2 are the steady-state sample.
- Result artifact: `evaluate_results/latency_probe_20260909/libero_spatial/gpu0_task0_results.json`.

### Replan latency

| Replan | Total | Joint future flow | K=8 action flow | JEPA rank | JEPA encoder phase | JEPA predictor, 2 steps |
|---:|---:|---:|---:|---:|---:|---:|
| 0, cold compile | 77.174 s | 36.872 s | 39.985 s | 305.5 ms | 108.6 ms | 193.4 ms |
| 1, warm | 1.012 s | 504.0 ms | 219.2 ms | 279.9 ms | 93.0 ms | 186.4 ms |
| 2, warm | 918.6 ms | 495.3 ms | 139.0 ms | 276.2 ms | 90.6 ms | 185.1 ms |
| warm mean | 965.3 ms | 499.7 ms | 179.1 ms | 278.0 ms | 91.8 ms | 185.8 ms |

The predictor numbers include the predictor forward, layer normalization, latent-history append, and energy computation. Per predictor step, the warm mean is 92.9 ms. The encoder-phase number includes V-JEPA image preparation plus the encoder forward for three two-frame clips: one current clip and two FastWAM target-future clips.

At an 8-action replan interval, the warm policy latency amortizes to 120.7 ms per low-level action. The simulator itself consumed 657.0 ms over the 24 policy-controlled actions, or 27.4 ms/action. The 24-action probe wall time was 81.0 s because it includes the first compiled replan; the task-level duration was 87.5 s because it additionally includes output-video encoding.

### Memory snapshots

| Mode / point | Allocated | Reserved | Peak allocated |
|---|---:|---:|---:|
| FastWAM only, after load | 12.552 GiB | 12.732 GiB | 12.552 GiB |
| FastWAM + V-JEPA2-AC, after load | 17.545 GiB | 17.773 GiB | 17.545 GiB |
| After rollout | 17.545 GiB | 19.494 GiB | 18.671 GiB |

Enabling colocated FP32 V-JEPA2-AC adds 4.993 GiB allocated and 5.041 GiB reserved memory at load time. The rollout temporarily raises allocated memory by 1.127 GiB above the post-load baseline; the CUDA allocator retains that workspace, so reserved memory remains 19.494 GiB after the episode.

### Interpretation

In the warm closed loop, `infer_joint` is the largest single stage (about 52% of the 965 ms replan). V-JEPA2-AC ranking is the next largest (about 29%); its predictor accounts for about two-thirds of rank time, and its encoder phase accounts for about one-third. The K=8 action flow is about 19% of steady replan latency.

The first replan must never be used as a steady-state latency estimate: it includes two newly encountered compiled shapes, one for joint future generation and one for K=8 action generation.

## 2026-09-09 — FastWAM action-only latency control

### Setup

The same real LIBERO task-0 latency probe was repeated with the released FastWAM policy alone: one action candidate, no future-video `infer_joint`, no V-JEPA2-AC, and the same 8-low-level-action replan interval. The 24 controlled actions again give three replans; the first compiled replan is excluded from the warm comparison.

### Results and fair comparison

| Mode | Warm replan 1 | Warm replan 2 | Warm mean |
|---|---:|---:|---:|
| FastWAM only, K=1 | 151.1 ms | 93.6 ms | 122.3 ms |
| FastWAM–JEPA, K=8 | 1.012 s | 918.6 ms | 965.3 ms |

Against the released action-only FastWAM protocol, JEPA closed-loop operation adds 843.0 ms/replan, a 7.90x replan latency multiplier under the matched 8-action interval. This comparison includes adding FastWAM future-video generation, so it is a system-level comparison, not an isolated JEPA-selector comparison. The extra latency should not all be attributed to V-JEPA2-AC weights:

| Incremental source | Warm contribution |
|---|---:|
| FastWAM reference-future `infer_joint` required by JEPA | 499.7 ms |
| K=1 to K=8 action flow | 62.1 ms |
| V-JEPA2-AC rank (encoder + predictor + small adaptation) | 278.0 ms |
| Remaining preprocessing/bookkeeping difference | approximately 3 ms |

The original released protocol normally replans every 10 low-level actions, whereas the matched control uses 8 to align with two JEPA AC steps. At the original 10-action interval, the K=1 action call is still approximately 122 ms/replan; only its per-action amortization changes.

### V-JEPA2-AC parameter accounting

The official checkpoint contains 1.317B FP32 parameters in total:

| Component | Parameters | Theoretical FP32 weight storage |
|---|---:|---:|
| Encoder | 1.012B | 3.771 GiB |
| Predictor | 0.305B | 1.137 GiB |
| Total | 1.317B | 4.908 GiB |

This accounts for the observed +4.993 GiB allocated when enabling colocated V-JEPA2-AC. The approximately 85 MiB difference covers module buffers and allocator/model bookkeeping.

## 2026-09-09 — Future-video-matched FastWAM control

### Latency definitions

The action-only control above is useful for comparison to the released baseline, but it does not isolate JEPA because the JEPA policy additionally requires a FastWAM future-video rollout. This control therefore enables `visualize_future_video=true` but keeps K=1 and disables V-JEPA2-AC. Its `infer_joint` debug action-equivalence check is explicitly disabled, matching the JEPA inference path.

This log uses three distinct latency terms:

- **JEPA model cost**: V-JEPA2-AC encoder plus action-conditioned predictor/ranking only. This is the incremental cost once a reference future and K candidate actions already exist.
- **Best-of-8 candidate-generation cost**: FastWAM action-flow inference that changes one candidate into eight. It is required by the current selection method but is not JEPA model compute.
- **JEPA selector-system increment**: the end-to-end extra cost of the current Best-of-8 + JEPA implementation versus future-video FastWAM K=1. It includes both preceding terms and small adaptation overhead.

### Results

| Mode | Warm replan 1 | Warm replan 2 | Warm mean |
|---|---:|---:|---:|
| FastWAM future video, K=1, no JEPA | 499.0 ms | 497.0 ms | **498.0 ms** |
| FastWAM future video + JEPA, K=8 | 1.012 s | 918.6 ms | **965.3 ms** |

With future-video generation held constant, the current Best-of-8 + JEPA selector system increases warm replan latency by **467.3 ms**, or **1.94x**. At the 8-action replan interval, this is an additional **58.4 ms per low-level action**.

The selector-system increment decomposes approximately as follows:

| Component | Warm contribution | Reporting category |
|---|---:|
| Separate K=8 action-flow generation | 179.1 ms | Best-of-8 candidate-generation cost |
| V-JEPA2-AC encoder + predictor ranking | **278.0 ms** | **JEPA model cost** |
| Candidate conversion, crop, and residual bookkeeping | approximately 10 ms | Selector-system overhead |

Thus the answer to “how much latency does JEPA itself add?” is **278.0 ms/replan** in this experiment: 91.8 ms encoder phase plus 185.8 ms predictor phase. The answer to “how much does the current Best-of-8 JEPA policy add?” is **467.3 ms/replan**, because the policy also needs a separate 179.1 ms K=8 FastWAM candidate-generation pass.

The corresponding control artifact is `evaluate_results/latency_probe_fastwam_future_k1_20260909_clean/libero_spatial/gpu0_task0_results.json`.

### Canonical end-to-end reporting result

For future reports, the canonical end-to-end latency comparison must keep FastWAM future-video generation enabled in both arms. Do **not** compare the 122.3 ms action-only released-policy result directly against the JEPA policy; retain it only as a reference for the original deployment protocol.

| Future-video-aligned mode | Warm model latency / replan | Replan interval | Amortized model latency / low-level action |
|---|---:|---:|---:|
| FastWAM future video, K=1, no JEPA | **498.0 ms** | 8 | **62.3 ms** |
| FastWAM future video + JEPA, K=8 | **965.3 ms** | 8 | **120.7 ms** |
| Difference | **+467.3 ms (1.94x)** | — | **+58.4 ms (1.94x)** |

This is the authoritative end-to-end latency conclusion for the current implementation after excluding model load and first-replan compilation. The 467.3 ms selector-system increment consists of 278.0 ms V-JEPA2-AC model cost, 179.1 ms K=8 FastWAM candidate generation, and approximately 10 ms adaptation/bookkeeping.

## 2026-09-07 — Historical LIBERO-Plus pose16 paired rollout evaluation

### Scope and protocol

- Benchmark slice: 16 LIBERO-Plus object-pose task IDs; four paired seeds (`7, 17, 27, 37`) per task, for 64 rollouts per method.
- Original FastWAM: released `libero_uncond_2cam224.pt`, one action sample, no V-JEPA2-AC, replan interval 10.
- FastWAM–JEPA: Best-of-8 action candidates, FP32 V-JEPA2-AC ranking, replan interval 8, future-video rollout enabled.
- The same task IDs and seeds were used, but the replan interval differs. This is a paired **system-level** comparison, not a replan-frequency-matched causal ablation of JEPA ranking.

### Overall result

| Method | Successes | Success rate |
|---|---:|---:|
| Original FastWAM | 35/64 | 54.7% |
| FastWAM–JEPA Best-of-8 | 37/64 | 57.8% |
| Difference | +2/64 | +3.1 percentage points |

Across paired rollouts, the JEPA system rescued seven original failures and regressed five original successes; 52 outcomes were unchanged. The sample indicates changed failure allocation rather than a sufficiently established global improvement.

### Breakdown by perturbation level

| Condition | Original FastWAM | FastWAM–JEPA | Difference |
|---|---:|---:|---:|
| `add_10` control/reference | 16/16 | 16/16 | 0 |
| Pose level 1 | 9/16 | 11/16 | +2 |
| Pose level 3 | 9/16 | 7/16 | -2 |
| Pose level 5 | 1/16 | 3/16 | +2 |

### Breakdown by target object

| Target object | Original FastWAM | FastWAM–JEPA | Difference |
|---|---:|---:|---:|
| Alphabet Soup | 9/16 | 10/16 | +1 |
| Milk | 4/16 | 4/16 | 0 |
| Salad Dressing | 9/16 | 14/16 | +5 |
| Tomato Sauce | 13/16 | 9/16 | -4 |

### Per-task results (four paired seeds each)

| Target | Variant | Task ID | Original FastWAM | FastWAM–JEPA | Delta |
|---|---|---:|---:|---:|---:|
| Alphabet Soup | control | 1818 | 4/4 | 4/4 | 0 |
| Alphabet Soup | level 1 | 1839 | 4/4 | 4/4 | 0 |
| Alphabet Soup | level 3 | 1847 | 1/4 | 2/4 | +1 |
| Alphabet Soup | level 5 | 1855 | 0/4 | 0/4 | 0 |
| Milk | control | 2037 | 4/4 | 4/4 | 0 |
| Milk | level 1 | 2062 | 0/4 | 0/4 | 0 |
| Milk | level 3 | 2070 | 0/4 | 0/4 | 0 |
| Milk | level 5 | 2078 | 0/4 | 0/4 | 0 |
| Salad Dressing | control | 2131 | 4/4 | 4/4 | 0 |
| Salad Dressing | level 1 | 2156 | 1/4 | 3/4 | +2 |
| Salad Dressing | level 3 | 2163 | 4/4 | 4/4 | 0 |
| Salad Dressing | level 5 | 2169 | 0/4 | 3/4 | +3 |
| Tomato Sauce | control | 2174 | 4/4 | 4/4 | 0 |
| Tomato Sauce | level 1 | 2203 | 4/4 | 4/4 | 0 |
| Tomato Sauce | level 3 | 2211 | 4/4 | 1/4 | -3 |
| Tomato Sauce | level 5 | 2217 | 1/4 | 0/4 | -1 |

Changed paired episodes: rescues are seed 7/task 1847; seed 17/tasks 2156 and 2169; seed 27/task 2169; and seed 37/tasks 1847, 2156, and 2169. Regressions are seed 7/task 2211; seed 27/tasks 1847 and 2211; and seed 37/tasks 2211 and 2217.

### Interpretation and artifacts

Salad Dressing is the main positive case (five paired rescues and no regression); Tomato Sauce is the main negative case, consistent with reference-model bias amplification when the predicted future hallucinates successful contact or object motion. No candidate-level actions, energies, selected IDs, or top-two margins were saved in this historical run, so these observations are video-based rather than candidate-level causal evidence.

## 2026-09-09 — LIBERO-Plus Long core3 paired rollout evaluation

### Scope and protocol

- Benchmark slice: `libero_10` Plus task IDs 410, 662, and 457, using the fixed Robot Initial States `14`, `310`, and `115` respectively; four paired policy/inference seeds (`7, 17, 27, 37`) per task, for 12 rollouts.
- Original FastWAM control: released `libero_uncond_2cam224.pt`, one action sample, `sigma_shift=5.0`, no future-video rollout, no V-JEPA2-AC, replan interval 10.
- FastWAM–JEPA: the same checkpoint, task ordering, Plus initial states, seeds, and trial count; future-video rollout enabled; FP32 V-JEPA2-AC Best-of-8 ranking; `replan_steps=8` for both FastWAM and the action conditioner.
- Hardware: physical GPUs 2 and 3; two persistent workers, one seed at a time.

### Overall result

| Method | Successes | Success rate |
|---|---:|---:|
| Original FastWAM | 1/12 | 8.3% |
| FastWAM–JEPA Best-of-8 | 4/12 | 33.3% |
| Difference | +3/12 | +25.0 percentage points |

### Per-task results

| Plus task ID | Task | Difficulty | Original FastWAM | FastWAM–JEPA | JEPA paired seeds that succeeded |
|---:|---|---:|---:|---:|---|
| 410 | Put black bowl in bottom drawer and close it | L4 | 1/4 | 3/4 | 7, 17, 27 |
| 662 | Put yellow-and-white mug in microwave and close it | L4 | 0/4 | 0/4 | — |
| 457 | Put white mug on left plate; yellow-and-white mug on right plate | L3 | 0/4 | 1/4 | 17 |
| **Overall** | — | — | **1/12 (8.3%)** | **4/12 (33.3%)** | — |

### Per-seed outcome table

| Task ID | Seed 7 | Seed 17 | Seed 27 | Seed 37 | JEPA total |
|---:|:---:|:---:|:---:|:---:|---:|
| 410 | success | success | success | failure | 3/4 |
| 662 | failure | failure | failure | failure | 0/4 |
| 457 | failure | success | failure | failure | 1/4 |

### Interpretation and artifacts

This is a stress-screening slice selected because tasks 662 and 457 had a 0/4 action-only baseline. The observed result is descriptive: task 410 improved from 1/4 to 3/4, task 457 produced one rescue, and task 662 remained unsolved. This slice alone should not be used to estimate a general selector gain.

The four seed runs completed normally and the evaluation process exited after all workers finished. Each seed has three task-result JSON files, a summary, CSV tables, worker logs, rollout videos, and predicted-future comparison videos under:
`evaluate_results/fastwam_jepa_libero_plus_long_core3_20260909/`.

The seed-27 manager was resumed after an interactive interruption; existing task 410/662 result files were preserved and only the missing task-457 result was completed before summary generation. Canonical per-seed summaries are in `seed_{7,17,27,37}/summary.json` and task tables in `seed_{7,17,27,37}/task_success_rates.csv`.

- Paired comparison and per-task table: [`evaluate_results/fastwam_jepa_libero_plus_pose16_20260907_153555/comparison_with_original.md`](../../evaluate_results/fastwam_jepa_libero_plus_pose16_20260907_153555/comparison_with_original.md)
- JEPA run summaries and videos: `evaluate_results/fastwam_jepa_libero_plus_pose16_20260907_153555/`
- Original FastWAM run summaries: `evaluate_results/fastwam_original_libero_plus_pose16_20260907_150235/`

## Proposed next evaluation — LIBERO-10 long-horizon reference set

### Why the current pose16 result is insufficient

The historical pose16 slice mostly evaluates one pick-and-place event under target-pose perturbations. It can reveal local grasp and placement effects, but it cannot establish that JEPA ranking improves multi-stage task completion. The local repository's original LIBERO README identifies `libero_10` as the ten-task downstream test split held out from LIBERO-100; FastWAM permits up to 700 environment steps for this suite, compared with 400 for `libero_object` and `libero_goal`.

### Original-data task references

The following IDs are the order in the original `libero_10/tasks_info.txt` and can be passed directly as `EVALUATION.task_suite_name=libero_10`, `EVALUATION.task_id=<id>`:

| ID | Original task language | Long-horizon property |
|---:|---|---|
| 0 | Turn on the stove and put the moka pot on it | appliance state change then placement |
| 1 | Put the black bowl in the bottom drawer and close it | placement then closure verification |
| 2 | Put the yellow-and-white mug in the microwave and close it | placement then closure verification |
| 3 | Put both moka pots on the stove | two-object sequential placement |
| 4 | Put both alphabet soup and cream cheese box in the basket | two-object sequential placement |
| 5 | Put both alphabet soup and tomato sauce in the basket | two-object sequential placement; directly tests the historically weak Tomato behavior |
| 6 | Put both cream cheese box and butter in the basket | two-object sequential placement |
| 7 | Put the white mug on the left plate and the yellow-and-white mug on the right plate | two objects plus ordered spatial assignment |
| 8 | Put the white mug on the plate and chocolate pudding right of the plate | two objects plus relational placement |
| 9 | Pick up the book and place it in the back compartment of the caddy | single-object fine spatial placement; useful control, but not a strong long-horizon task |

Recommended core long-horizon set: IDs `0, 1, 2, 3, 5, 7, 8`. It covers state transitions, closure after placement, two-object sequencing, and relational placement, while retaining task 5 as a stress test for the historical Tomato Sauce regression.

### Required reporting beyond final success

The current JEPA rank horizon is two AC steps (eight low-level actions), far shorter than these tasks' full subgoal chains. For every rollout, record final success **and** ordered subgoal events: object grasp/placement, drawer or microwave state, first-object completion before second-object attempt, and time/number of replans to each event. This distinguishes a useful local selector from a method that only improves the first grasp while harming completion of later subgoals.

## 2026-09-09 — Original FastWAM baseline screening on LIBERO-Plus Long core3

- Policy and protocol: released `libero_uncond_2cam224.pt`; `sigma_shift=5.0`; one action sample; `replan_steps=10`; no future-video rollout, Best-of-K, or V-JEPA.
- Benchmark: three `libero_10` LIBERO-Plus **Robot Initial States** variants. Each variant has one rollout for each fixed policy seed `7, 17, 27, 37` (12 rollouts total). Each selected Plus variant fixes its benchmark-defined initial state; randomness here is the policy/inference seed.
- Selected tasks: task 410 (drawer close, L4), task 662 (microwave close, L4), and task 457 (two mugs with left/right relation, L3).
- Execution: persistent two-worker pool on RTX 4090 GPUs 2 and 3; each seed starts fresh workers and exits cleanly after its three tasks. Wall time was about 18 minutes including four cold starts. All 12 task JSONs, 12 MP4 rollouts, four manager configurations, and worker logs were retained.

| Plus task ID | Original FastWAM successes | Per-seed outcomes |
|---:|---:|---|
| 410 — black bowl in drawer, then close | 1/4 (25.0%) | 7 ✗; 17 ✓; 27 ✗; 37 ✗ |
| 662 — yellow/white mug in microwave, then close | 0/4 (0.0%) | 7 ✗; 17 ✗; 27 ✗; 37 ✗ |
| 457 — white mug left plate, yellow/white mug right plate | 0/4 (0.0%) | 7 ✗; 17 ✗; 27 ✗; 37 ✗ |
| **Overall** | **1/12 (8.3%)** | — |

Interpretation: these three individually selected Long variants are materially harder for this released action-only checkpoint than the third-party marginal `Plus-Long` aggregate. They are therefore unsuitable as the only selector scorecard: two tasks exhibit a 0/4 floor under this screening budget. Retain task 410 as a possible stress task, but for a JEPA gain estimate select additional L2/L3 Long variants with a non-floor original baseline, then rerun the identical paired seeds.

- Aggregate result and artifact index: [`evaluate_results/fastwam_original_libero_plus_long_core3_20260909/summary.md`](../../evaluate_results/fastwam_original_libero_plus_long_core3_20260909/summary.md)
- Raw artifacts: `evaluate_results/fastwam_original_libero_plus_long_core3_20260909/seed_{7,17,27,37}/`

## 2026-09-09 — Future-video-aligned K=1 Long core3 baseline

### Reason for rerun

The preceding action-only Long core3 baseline was not a controlled accuracy comparison for JEPA: it used `visualize_future_video=false` and `replan_steps=10`, whereas the JEPA treatment generated a future video and replanned every eight low-level actions. This rerun aligns those system-level settings while disabling only the selector treatment.

### Protocol

- Same released `libero_uncond_2cam224.pt` checkpoint and dataset statistics as the historical JEPA run.
- Same LIBERO-Plus task IDs 410, 662, and 457, task ordering, fixed benchmark initial states, seeds 7/17/27/37, and one trial per pair.
- `sigma_shift=5.0`, `visualize_future_video=true`, and `replan_steps=8`.
- `vjepa2_ac.enabled=false`. Because the ranker is absent, the evaluation path does not forward the inactive `vjepa2_ac.num_action_candidates` config field; `infer_joint` uses its default effective K=1.
- The successful runs used at most two physical RTX 4090 GPUs concurrently (GPU0 and GPU2). Earlier workers that encountered shared-GPU OOMs produced no accepted task results; all reported rows below come from normally completed workers.

### Aligned baseline result

| Plus task ID | Seed 7 | Seed 17 | Seed 27 | Seed 37 | K=1 total |
|---:|:---:|:---:|:---:|:---:|---:|
| 410 | success | success | failure | failure | 2/4 |
| 662 | failure | failure | failure | failure | 0/4 |
| 457 | failure | failure | failure | failure | 0/4 |
| **Overall** | — | — | — | — | **2/12 (16.7%)** |

Mean future-video PSNR across all 12 rollouts was **26.6452 dB**. Artifact validation found 12 result JSONs, 12 rollout videos, 12 aggregate `replan=all` future-video comparisons, and 957 per-replan future-video comparison clips.

### Corrected comparison with the completed historical JEPA treatment

| Method | Task 410 | Task 662 | Task 457 | Overall |
|---|---:|---:|---:|---:|
| Historical action-only FastWAM (`future_video=false`, replan 10) | 1/4 | 0/4 | 0/4 | 1/12 (8.3%) |
| **Aligned future-video FastWAM K=1** (`future_video=true`, replan 8) | **2/4** | **0/4** | **0/4** | **2/12 (16.7%)** |
| Historical two-call JEPA Best-of-8 (`future_video=true`, replan 8) | 3/4 | 0/4 | 1/4 | 4/12 (33.3%) |

Against the aligned K=1 baseline, the historical JEPA treatment improves by **+2/12 (+16.7 percentage points)**, not the previously quoted +3/12. The paired changes are two rescues with no regressions: task 410/seed 27 and task 457/seed 17. Task 410/seeds 7 and 17 succeed in both arms; all other pairs fail in both arms.

### Interpretation boundary and artifacts

This correction makes the control factors fair for the completed historical treatment, but that JEPA run used the old two-call candidate path. The newer single-pass joint K=8 implementation changes candidate generation and therefore still requires a fresh 12-rollout JEPA treatment before its success rate can be reported.

- Aggregate report: [`evaluate_results/fastwam_future_k1_libero_plus_long_core3_20260909_aligned/summary.md`](../../evaluate_results/fastwam_future_k1_libero_plus_long_core3_20260909_aligned/summary.md)
- Paired machine-readable table: [`evaluate_results/fastwam_future_k1_libero_plus_long_core3_20260909_aligned/comparison_with_historical_jepa.csv`](../../evaluate_results/fastwam_future_k1_libero_plus_long_core3_20260909_aligned/comparison_with_historical_jepa.csv)
- Raw artifacts: `evaluate_results/fastwam_future_k1_libero_plus_long_core3_20260909_aligned/seed_{7,17,27,37}/`

## 2026-09-10 — Single-pass joint K=8 Long core3 rollout

### Protocol and result

The current single-pass `infer_joint(num_action_candidates=8)` implementation was evaluated on the same three LIBERO-Plus variants, seeds 7/17/27/37, checkpoint, statistics, future-video setting, and eight-action replan interval as the aligned K=1 control. Seeds 7/27 ran on physical GPU1 and seeds 17/37 on GPU3, with at most two workers concurrently.

| Plus task ID | Aligned K=1 | Old two-call JEPA | New joint K=8 JEPA |
|---:|---:|---:|---:|
| 410 | 2/4 | 3/4 | **1/4** |
| 662 | 0/4 | 0/4 | **0/4** |
| 457 | 0/4 | 1/4 | **0/4** |
| **Overall** | **2/12 (16.7%)** | **4/12 (33.3%)** | **1/12 (8.3%)** |

The new treatment's only success was task 410/seed 7. Relative to aligned K=1 it produced zero rescues and one regression (task 410/seed 17), or -1/12 (-8.3 percentage points). Relative to old two-call JEPA it lost three successes: task 410/seeds 17 and 27, and task 457/seed 17.

All 12 result JSONs, rollout videos, and aggregate future-video comparisons passed integrity checks; 1006 per-replan future-video clips were retained. Mean future-video PSNR was 26.9725 dB.

### Meaning of joint K=8 and non-equivalence

K=8 is the action-candidate batch only. One video trajectory remains batch one. At each MoT layer, the video K/V tensors are expanded as views for eight independent action trajectories; candidates do not attend to one another and are not generated by eight serial Python calls.

The old path generated the reference with `infer_joint(K=1)`, then ran a standalone `infer_action(K=8)` whose video expert prefills a current-frame cache once. The new path advances the video and eight actions inside the same joint flow. The attention mask permits actions to attend only first-frame video tokens, so the two paths are mathematically intended to be close, but their kernels are not bitwise identical: full masked video attention/recomputation in the joint graph replaces the standalone first-frame prefill graph. Floating-point differences compound through 30 layers and ten flow steps.

This matters because selector decisions are near ties. In the first task-410 replan, old and new paths selected different candidate IDs for seeds 7, 17, and 27. Typical top-two energy gaps are only around 1e-5 to 1e-4. A small numerical change can therefore flip the selected eight-action chunk; closed-loop states then diverge for all later replans. The 12-rollout result shows the optimization is not behavior preserving on this stress slice, though the sample is too small to estimate a general accuracy delta.

- Report: [`evaluate_results/fastwam_jepa_joint_k8_libero_plus_long_core3_20260910/summary.md`](../../evaluate_results/fastwam_jepa_joint_k8_libero_plus_long_core3_20260910/summary.md)
- Paired CSV: [`evaluate_results/fastwam_jepa_joint_k8_libero_plus_long_core3_20260910/comparison.csv`](../../evaluate_results/fastwam_jepa_joint_k8_libero_plus_long_core3_20260910/comparison.csv)
- Raw artifacts: `evaluate_results/fastwam_jepa_joint_k8_libero_plus_long_core3_20260910/seed_{7,17,27,37}/`

### Method decision and backbone roadmap

Use the best observed treatment for the current backbone: the historical two-call pipeline, which achieved 4/12 on this aligned slice. The LIBERO configuration now defaults `EVALUATION.vjepa2_ac.candidate_generation_mode=separate`, meaning `infer_joint(K=1)` generates the reference future and the released `infer_action(K=8)` path generates candidates. The 1/12 single-pass implementation remains reproducible with `candidate_generation_mode=joint`, but is experimental and must not be used as the default accuracy result.

This decision intentionally accepts the historical 965.3 ms/replan latency instead of the single-pass 860.2 ms/replan result until an optimization preserves candidate behavior. When the backbone is replaced, the intended architecture is fully cascaded: stage 1 generates video only, then stage 2 generates actions. That removes the current `infer_joint(K=1)` action output that is discarded before the separate Best-of-8 call, so the cascade will not waste that extra generated action trajectory.
