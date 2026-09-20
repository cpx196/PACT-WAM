# Original FastWAM — LIBERO-Plus Long core3

Policy: released `libero_uncond_2cam224.pt`; one action sample; no future-video rollout; no Best-of-K; no V-JEPA. LIBERO-Plus `libero_10` robot-initial-state variants; one rollout for each paired policy seed `7, 17, 27, 37`.

| Plus task ID | Task | Difficulty | Success | Paired seeds that succeeded |
|---:|---|---:|---:|---|
| 410 | Put black bowl in bottom drawer and close it | L4 | 1/4 (25.0%) | 17 |
| 662 | Put yellow-and-white mug in microwave and close it | L4 | 0/4 (0.0%) | — |
| 457 | White mug left plate; yellow-and-white mug right plate | L3 | 0/4 (0.0%) | — |
| **Overall** | — | — | **1/12 (8.3%)** | — |

Each seed directory contains its immutable `manager_config.yaml`, persistent-worker logs, per-task result JSON, per-seed summary, and three rollout videos (12 videos total). This is a baseline screening result, not a statistical claim: four policy seeds per task yields a wide binomial uncertainty interval.
