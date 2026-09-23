#!/usr/bin/env python3
"""Create a clearly labeled synthetic 144-result view from 13 new rollouts."""

import csv
import json
import re
from pathlib import Path


ROOT = Path("/data/pxchen/PACT-WAM/evaluate_results")
ORIGINAL = ROOT / "trigger30_r5r0_rawgrad4x_long36_full144_shift1_cap0p02_20260923"
REPLACEMENT = ROOT / "trigger30_forcefirst4_rawgrad4x_sensitive13_shift1_cap0p02_20260923"
MERGED = ROOT / "trigger30_forcefirst4_rawgrad4x_synthetic144_shift1_cap0p02_20260923"
NO_GUIDANCE_64 = ROOT / "trigger_probe_long36_failures64_shift1_20260922"
NO_GUIDANCE_80 = ROOT / "trigger_probe_long36_remaining80_shift1_20260922"
FULL_4X = ROOT / "guidance789_rawgrad4x_long36_full144_shift1_cap0p02_20260922"
SEEDS = (7, 17, 27, 37)


def result_map(root):
    out = {}
    for path in root.glob("seed_*/*/gpu*_task*_results.json"):
        seed = int(re.search(r"seed_(\d+)", str(path)).group(1))
        payload = json.loads(path.read_text())
        if payload["total_episodes"] != 1 or payload["successes"] not in (0, 1):
            raise ValueError(f"Expected one binary-result episode: {path}")
        key = (seed, payload["task_suite"], int(payload["task_id"]))
        if key in out:
            raise ValueError(f"Duplicate result for {key}: {path}")
        out[key] = (path, int(payload["successes"]))
    return out


def expected_keys():
    keys = set()
    for seed in SEEDS:
        path = REPLACEMENT / "task_lists" / f"seed{seed}.txt"
        for line in path.read_text().splitlines():
            suite, task_id = line.split(",")
            key = (seed, suite, int(task_id))
            if key in keys:
                raise ValueError(f"Duplicate task-list entry: {key}")
            keys.add(key)
    if len(keys) != 13:
        raise ValueError(f"Expected 13 seed/task pairs, found {len(keys)}")
    return keys


def main():
    expected = expected_keys()
    original = result_map(ORIGINAL)
    replacement = result_map(REPLACEMENT)
    no_guidance = {**result_map(NO_GUIDANCE_64), **result_map(NO_GUIDANCE_80)}
    full_4x = result_map(FULL_4X)
    if len(original) != 144 or set(replacement) != expected or not expected <= set(original):
        raise ValueError(
            f"Result-set mismatch: original={len(original)} replacement={len(replacement)} "
            f"missing_new={sorted(expected - set(replacement))} "
            f"unexpected_new={sorted(set(replacement) - expected)}"
        )
    if set(no_guidance) != set(original) or set(full_4x) != set(original):
        raise ValueError("The three 144-result baselines do not have matching seed/task keys")
    vs_no_guidance = {
        key for key in original if original[key][1] != no_guidance[key][1]
    }
    vs_full_4x = {key for key in original if original[key][1] != full_4x[key][1]}
    if expected != vs_no_guidance | vs_full_4x:
        raise ValueError("Replacement task list is not the requested union of differences")
    if MERGED.exists():
        raise FileExistsError(f"Refusing to overwrite existing merged result: {MERGED}")

    merged_rows = []
    for key in sorted(original):
        old_path, old_success = original[key]
        replaced = key in expected
        source_path, merged_success = replacement[key] if replaced else original[key]
        merged_rows.append(
            {
                "seed": key[0],
                "suite": key[1],
                "task_id": key[2],
                "source": "force_first4" if replaced else "original_trigger",
                "no_guidance_success": no_guidance[key][1],
                "original_trigger_success": old_success,
                "full_4x_success": full_4x[key][1],
                "merged_success": merged_success,
                "delta": merged_success - old_success,
                "selected_vs_no_guidance": int(key in vs_no_guidance),
                "selected_vs_full_4x": int(key in vs_full_4x),
                "source_result_path": str(source_path),
            }
        )

    original_wins = sum(r["original_trigger_success"] for r in merged_rows)
    merged_wins = sum(r["merged_success"] for r in merged_rows)
    summary = {
        "kind": "synthetic_splice_not_full_rollout",
        "assumption": "The other 131 original-trigger seed/task outcomes remain unchanged.",
        "original_root": str(ORIGINAL),
        "replacement_root": str(REPLACEMENT),
        "total_episodes": len(merged_rows),
        "replaced_episodes": len(expected),
        "unchanged_episodes": len(merged_rows) - len(expected),
        "selection_vs_no_guidance": len(vs_no_guidance),
        "selection_vs_full_4x": len(vs_full_4x),
        "no_guidance_successes": sum(r["no_guidance_success"] for r in merged_rows),
        "original_trigger_successes": original_wins,
        "full_4x_successes": sum(r["full_4x_success"] for r in merged_rows),
        "synthetic_successes": merged_wins,
        "synthetic_success_rate": merged_wins / len(merged_rows),
        "net_success_change": merged_wins - original_wins,
        "per_seed": {
            str(seed): {
                "episodes": sum(r["seed"] == seed for r in merged_rows),
                "successes": sum(r["merged_success"] for r in merged_rows if r["seed"] == seed),
                "replaced": sum(r["seed"] == seed and r["source"] == "force_first4" for r in merged_rows),
            }
            for seed in SEEDS
        },
    }

    MERGED.mkdir()
    for row in merged_rows:
        target_dir = MERGED / f"seed_{row['seed']}" / row["suite"]
        target_dir.mkdir(parents=True, exist_ok=True)
        source = Path(row["source_result_path"])
        (target_dir / source.name).symlink_to(source)
    with (MERGED / "merge_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged_rows[0]))
        writer.writeheader()
        writer.writerows(merged_rows)
    (MERGED / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (MERGED / "README.md").write_text(
        "# Synthetic 144 result\n\n"
        "This is **not** a full 144-episode rerun. It substitutes 13 new force-first4 "
        "rollouts into the prior 144-episode trigger results, assuming the other 131 "
        "outcomes would not change. Result JSON files are symlinks to their sources; "
        "see `merge_manifest.csv` for provenance. Videos are not copied.\n"
    )
    print(json.dumps(summary, indent=2))
    print(MERGED)


if __name__ == "__main__":
    main()
