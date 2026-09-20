# FastWAM-JEPA vs. original FastWAM

The comparison uses the same 16 LIBERO-Plus task IDs and the same seeds: 7, 17, 27, and 37. Each cell therefore contains four paired rollouts.

## Overall

| Method | Success | Rate |
|---|---:|---:|
| Original FastWAM | 35/64 | 54.7% |
| FastWAM-JEPA Best-of-8 | 37/64 | 57.8% |
| Difference | +2/64 | +3.1 percentage points |

Among the 64 paired rollouts, JEPA rescued 7 original failures, harmed 5 original successes, and left 52 outcomes unchanged. It rescued 7/29 (24.1%) of the original failures and regressed 5/35 (14.3%) of the original successes.

## By difficulty

| Difficulty | Original | JEPA | Delta |
|---|---:|---:|---:|
| add_10 control | 16/16 | 16/16 | 0 |
| pose level 1 | 9/16 | 11/16 | +2 |
| pose level 3 | 9/16 | 7/16 | -2 |
| pose level 5 | 1/16 | 3/16 | +2 |

## By target object

| Target | Original | JEPA | Delta |
|---|---:|---:|---:|
| Alphabet Soup | 9/16 | 10/16 | +1 |
| Milk | 4/16 | 4/16 | 0 |
| Salad Dressing | 9/16 | 14/16 | +5 |
| Tomato Sauce | 13/16 | 9/16 | -4 |

## Per task

| Target | Variant | Task ID | Original | JEPA | Delta |
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

## Changed paired rollouts

JEPA rescues (original 0 -> JEPA 1):

- seed 7: task 1847
- seed 17: tasks 2156 and 2169
- seed 27: task 2169
- seed 37: tasks 1847, 2156, and 2169

JEPA regressions (original 1 -> JEPA 0):

- seed 7: task 2211
- seed 27: tasks 1847 and 2211
- seed 37: tasks 2211 and 2217

## Interpretation

The net gain is small and strongly object-dependent. The principal positive result is Salad Dressing, especially pose level 5. The principal negative result is Tomato Sauce level 3. Milk remains entirely unsolved outside the control condition. The 7-to-5 discordant split is insufficient evidence of a reliable global improvement; it is better treated as evidence that JEPA ranking changes the failure distribution and may help specific pose/object regimes.

The current run preserves 64 rollout videos and 2,183 future-prediction videos. It does not preserve candidate-level actions, energy values, or selected candidate IDs, so the exact selection mechanism cannot be reconstructed for completed episodes without an instrumented rerun.
