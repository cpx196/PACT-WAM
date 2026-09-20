# Original FastWAM on LIBERO-Plus pose16

Policy: released `libero_uncond_2cam224.pt`; single action sample; no V-JEPA; no Best-of-K; no future-video rollout.

| Target | Variant | Task ID | Success | Rate | Completed seeds |
|---|---|---:|---:|---:|---|
| Alphabet Soup | add_10 control | 1818 | 4/4 | 100.0% | 7;17;27;37 |
| Alphabet Soup | pose level 1 | 1839 | 4/4 | 100.0% | 7;17;27;37 |
| Alphabet Soup | pose level 3 | 1847 | 1/4 | 25.0% | 7;17;27;37 |
| Alphabet Soup | pose level 5 | 1855 | 0/4 | 0.0% | 7;17;27;37 |
| Milk | add_10 control | 2037 | 4/4 | 100.0% | 7;17;27;37 |
| Milk | pose level 1 | 2062 | 0/4 | 0.0% | 7;17;27;37 |
| Milk | pose level 3 | 2070 | 0/4 | 0.0% | 7;17;27;37 |
| Milk | pose level 5 | 2078 | 0/4 | 0.0% | 7;17;27;37 |
| Salad Dressing | add_10 control | 2131 | 4/4 | 100.0% | 7;17;27;37 |
| Salad Dressing | pose level 1 | 2156 | 1/4 | 25.0% | 7;17;27;37 |
| Salad Dressing | pose level 3 | 2163 | 4/4 | 100.0% | 7;17;27;37 |
| Salad Dressing | pose level 5 | 2169 | 0/4 | 0.0% | 7;17;27;37 |
| Tomato Sauce | add_10 control | 2174 | 4/4 | 100.0% | 7;17;27;37 |
| Tomato Sauce | pose level 1 | 2203 | 4/4 | 100.0% | 7;17;27;37 |
| Tomato Sauce | pose level 3 | 2211 | 4/4 | 100.0% | 7;17;27;37 |
| Tomato Sauce | pose level 5 | 2217 | 1/4 | 25.0% | 7;17;27;37 |

Per-rollout video paths are recorded in `summary.csv` and stored below each `seed_*/libero_object/videos/` directory.
