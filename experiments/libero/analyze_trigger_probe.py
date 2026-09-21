#!/usr/bin/env python3
"""Analyze offline R_future -> JEPA-loss trigger probes from LIBERO results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("gpu*_task*_results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for episode_idx, episode in enumerate(payload.get("episode_timings", [])):
            for replan_idx, replan in enumerate(episode.get("per_replan", [])):
                trace = replan.get("action_response_trace", [])
                loss = replan.get("trigger_probe_jepa_loss")
                if loss is None or not trace:
                    continue
                row = {
                    "result_file": str(path),
                    "episode": episode_idx,
                    "planning_point": replan_idx,
                    "jepa_loss": float(loss),
                }
                for step in trace:
                    row[f"r_future_step_{int(step['denoise_step'])}"] = float(
                        step["r_future_layer_mean"]
                    )
                rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_dir or args.results_root / "trigger_analysis"
    output.mkdir(parents=True, exist_ok=True)

    rows = _rows(args.results_root)
    if not rows:
        raise SystemExit(f"No trigger-probe planning points found under {args.results_root}")
    _write_csv(output / "planning_points.csv", rows)

    loss = np.asarray([row["jepa_loss"] for row in rows], dtype=np.float64)
    step_keys = sorted(
        (key for key in rows[0] if key.startswith("r_future_step_")),
        key=lambda key: int(key.rsplit("_", 1)[-1]),
    )
    correlations = []
    groups = []
    trigger = []
    high_loss = loss >= np.quantile(loss, 0.8)
    for key in step_keys:
        response = np.asarray([row[key] for row in rows], dtype=np.float64)
        pearson = pearsonr(response, loss)
        spearman = spearmanr(response, loss)
        correlations.append(
            {
                "denoise_step": int(key.rsplit("_", 1)[-1]),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_r": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "n": len(rows),
            }
        )
        bounds = np.quantile(response, [1 / 3, 2 / 3])
        labels = np.where(response <= bounds[0], "low", np.where(response <= bounds[1], "mid", "high"))
        for label in ("low", "mid", "high"):
            selected = loss[labels == label]
            groups.append(
                {
                    "denoise_step": int(key.rsplit("_", 1)[-1]),
                    "group": label,
                    "count": int(selected.size),
                    "jepa_loss_mean": float(selected.mean()),
                    "jepa_loss_median": float(np.median(selected)),
                }
            )
        called = response <= np.quantile(response, 0.2)
        caught = called & high_loss
        trigger.append(
            {
                "denoise_step": int(key.rsplit("_", 1)[-1]),
                "recall": float(caught.sum() / max(1, high_loss.sum())),
                "precision": float(caught.sum() / max(1, called.sum())),
                "call_rate": float(called.mean()),
                "high_loss_count": int(high_loss.sum()),
                "called_count": int(called.sum()),
            }
        )

    _write_csv(output / "correlations.csv", correlations)
    _write_csv(output / "response_groups.csv", groups)
    _write_csv(output / "trigger_simulation.csv", trigger)
    summary = {
        "planning_points": len(rows),
        "definition": "top-20% JEPA loss vs bottom-20% R_future",
        "best_spearman_step": max(correlations, key=lambda item: abs(item["spearman_r"])),
        "best_recall_step": max(trigger, key=lambda item: item["recall"]),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
