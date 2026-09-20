"""Offline collection and plotting for action-flow convergence diagnostics."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


class ActionDenoiseConvergenceLogger:
    """Accumulate clean-action estimates across closed-loop generations.

    Only detached CPU/NumPy copies are retained.  Plotting is deliberately
    deferred until :meth:`save`, which is called after the rollout finishes.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self._rows: list[dict[str, Any]] = []
        self._estimate_records: list[dict[str, Any]] = []
        self._final_records: list[dict[str, Any]] = []
        self._next_generation_id = 0

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu", dtype=torch.float32).numpy().copy()
        return np.asarray(value, dtype=np.float32).copy()

    @staticmethod
    def _mse(lhs: np.ndarray, rhs: np.ndarray, horizon: int | None = None) -> float:
        if horizon is not None:
            if horizon <= 0:
                return math.nan
            lhs = lhs[..., :horizon, :]
            rhs = rhs[..., :horizon, :]
        return float(np.mean((lhs - rhs) ** 2, dtype=np.float64))

    def record(
        self,
        trace: Sequence[dict[str, Any]],
        clean_action_estimates: Sequence[Any],
        final_action: Any,
        executed_horizon: int,
        generation_id: int | None = None,
    ) -> int | None:
        """Record one complete action-generation call and return its id."""
        if not trace or not clean_action_estimates or final_action is None:
            return None
        if len(trace) != len(clean_action_estimates):
            raise ValueError(
                "Action trace and clean-action estimate counts differ: "
                f"{len(trace)} != {len(clean_action_estimates)}"
            )

        generation_id = (
            self._next_generation_id if generation_id is None else int(generation_id)
        )
        self._next_generation_id = max(self._next_generation_id, generation_id + 1)
        final = self._numpy(final_action)
        if final.ndim == 2:
            final = final[None, ...]
        if final.ndim != 3:
            raise ValueError(f"Expected final action [K,H,D], got {final.shape}")
        final_clean_estimate = self._numpy(clean_action_estimates[-1])
        if final_clean_estimate.ndim == 2:
            final_clean_estimate = final_clean_estimate[None, ...]
        if final_clean_estimate.shape != final.shape:
            raise ValueError(
                "Final clean estimate shape differs from final action: "
                f"{final_clean_estimate.shape} != {final.shape}"
            )
        horizon = max(0, min(int(executed_horizon), int(final.shape[1])))
        self._final_records.append(
            {
                "generation_id": generation_id,
                "final_action": final,
                "final_clean_estimate": final_clean_estimate,
                "action_shape": list(final.shape),
                "executed_horizon": horizon,
            }
        )

        previous: np.ndarray | None = None
        for fallback_step, (trace_item, estimate_value) in enumerate(
            zip(trace, clean_action_estimates)
        ):
            estimate = self._numpy(estimate_value)
            if estimate.ndim == 2:
                estimate = estimate[None, ...]
            if estimate.shape != final.shape:
                raise ValueError(
                    "Clean-action estimate shape differs from final action: "
                    f"{estimate.shape} != {final.shape}"
                )
            step = int(trace_item.get("step", fallback_step))
            timestep = float(trace_item.get("timestep", math.nan))
            row = {
                "generation_id": generation_id,
                "denoise_step": step,
                "timestep": timestep,
                "r_full": self._mse(estimate, previous) if previous is not None else math.nan,
                "r_exec": (
                    self._mse(estimate, previous, horizon) if previous is not None else math.nan
                ),
                # The requested d_i reference is the last clean estimate
                # \hat{a}_0^(N), not the final noisy/state tensor.
                "d_full": self._mse(estimate, final_clean_estimate),
                "d_exec": self._mse(estimate, final_clean_estimate, horizon),
                "action_shape": str(list(estimate.shape)),
                "executed_horizon": horizon,
            }
            self._rows.append(row)
            self._estimate_records.append(
                {
                    "generation_id": generation_id,
                    "denoise_step": step,
                    "timestep": timestep,
                    "estimate": estimate,
                }
            )
            previous = estimate
        return generation_id

    @staticmethod
    def _aggregate(values: list[float]) -> tuple[float, float, float]:
        finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
        if finite.size == 0:
            return math.nan, math.nan, math.nan
        return float(finite.mean()), float(finite.std()), float(np.median(finite))

    def save(self) -> dict[str, str]:
        """Write raw/summary CSVs, detached estimates, and both plots."""
        if not self._rows:
            return {}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = self.output_dir / "action_denoise_convergence.csv"
        summary_path = self.output_dir / "action_denoise_convergence_summary.csv"
        estimates_path = self.output_dir / "action_denoise_convergence_estimates.npz"
        change_plot_path = self.output_dir / "action_change_curve.png"
        distance_plot_path = self.output_dir / "distance_to_final_action.png"

        raw_fields = [
            "generation_id",
            "denoise_step",
            "timestep",
            "r_full",
            "r_exec",
            "d_full",
            "d_exec",
            "action_shape",
            "executed_horizon",
        ]
        with raw_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=raw_fields)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)

        steps = sorted({int(row["denoise_step"]) for row in self._rows})
        summary_fields = ["step", "count"]
        for metric in ("r_full", "r_exec", "d_full", "d_exec"):
            summary_fields.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_median"])
        summary_rows = []
        for step in steps:
            step_rows = [row for row in self._rows if int(row["denoise_step"]) == step]
            summary: dict[str, Any] = {"step": step, "count": len(step_rows)}
            for metric in ("r_full", "r_exec", "d_full", "d_exec"):
                mean, std, median = self._aggregate([float(row[metric]) for row in step_rows])
                summary[f"{metric}_mean"] = mean
                summary[f"{metric}_std"] = std
                summary[f"{metric}_median"] = median
            summary_rows.append(summary)
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary_fields)
            writer.writeheader()
            writer.writerows(summary_rows)

        np.savez_compressed(
            estimates_path,
            generation_id=np.asarray([item["generation_id"] for item in self._estimate_records]),
            denoise_step=np.asarray([item["denoise_step"] for item in self._estimate_records]),
            timestep=np.asarray([item["timestep"] for item in self._estimate_records]),
            clean_action_estimate=np.asarray(
                [item["estimate"] for item in self._estimate_records], dtype=object
            ),
            final_action=np.asarray(
                [item["final_action"] for item in self._final_records], dtype=object
            ),
        )

        self._plot_metric(
            summary_rows,
            "r_full",
            "r_exec",
            "Action change (element-wise MSE)",
            change_plot_path,
        )
        self._plot_metric(
            summary_rows,
            "d_full",
            "d_exec",
            "Distance to final action (element-wise MSE)",
            distance_plot_path,
        )
        return {
            "raw_csv": str(raw_path),
            "summary_csv": str(summary_path),
            "estimates_npz": str(estimates_path),
            "action_change_plot": str(change_plot_path),
            "distance_to_final_action_plot": str(distance_plot_path),
        }

    @staticmethod
    def _plot_metric(
        rows: list[dict[str, Any]],
        full_metric: str,
        exec_metric: str,
        ylabel: str,
        path: Path,
    ) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            ActionDenoiseConvergenceLogger._plot_metric_pillow(
                rows, full_metric, exec_metric, ylabel, path
            )
            return

        x = np.asarray([row["step"] for row in rows], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for metric, label, color in (
            (full_metric, "full chunk", "tab:blue"),
            (exec_metric, "executed chunk", "tab:orange"),
        ):
            mean = np.asarray([row[f"{metric}_mean"] for row in rows], dtype=np.float64)
            std = np.asarray([row[f"{metric}_std"] for row in rows], dtype=np.float64)
            ax.plot(x, mean, marker="o", label=label, color=color)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
        ax.set_xlabel("Denoising step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    @staticmethod
    def _plot_metric_pillow(
        rows: list[dict[str, Any]],
        full_metric: str,
        exec_metric: str,
        ylabel: str,
        path: Path,
    ) -> None:
        """Small dependency-free fallback used when matplotlib is unavailable."""
        from PIL import Image, ImageDraw

        width, height = 900, 560
        left, right, top, bottom = 90, 35, 45, 75
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        plot_w = width - left - right
        plot_h = height - top - bottom
        series = []
        for metric, label, color in (
            (full_metric, "full chunk", (40, 110, 190)),
            (exec_metric, "executed chunk", (220, 125, 35)),
        ):
            mean = np.asarray([row[f"{metric}_mean"] for row in rows], dtype=np.float64)
            std = np.asarray([row[f"{metric}_std"] for row in rows], dtype=np.float64)
            series.append((mean, std, label, color))
        bounds = np.concatenate(
            [np.concatenate((mean - std, mean + std)) for mean, std, _, _ in series]
        )
        finite = bounds[np.isfinite(bounds)]
        y_min = float(finite.min()) if finite.size else 0.0
        y_max = float(finite.max()) if finite.size else 1.0
        if y_max <= y_min:
            y_max = y_min + 1.0
        margin = 0.05 * (y_max - y_min)
        y_min -= margin
        y_max += margin

        def point(index: int, value: float) -> tuple[int, int] | None:
            if not np.isfinite(value):
                return None
            x = left if len(rows) == 1 else left + int(plot_w * index / (len(rows) - 1))
            y = top + int((y_max - value) / (y_max - y_min) * plot_h)
            return x, y

        draw.line((left, top, left, height - bottom), fill="black", width=2)
        draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=2)
        draw.text((left, 12), ylabel, fill="black")
        draw.text((width // 2 - 45, height - 35), "Denoising step", fill="black")
        for mean, std, label, color in series:
            upper = [point(i, float(mean[i] + std[i])) for i in range(len(rows))]
            lower = [point(i, float(mean[i] - std[i])) for i in range(len(rows))]
            band = [p for p in upper if p is not None] + [p for p in reversed(lower) if p is not None]
            if len(band) >= 3:
                draw.polygon(band, fill=tuple(min(255, c + 100) for c in color))
            line = [point(i, float(mean[i])) for i in range(len(rows))]
            line = [p for p in line if p is not None]
            if len(line) >= 2:
                draw.line(line, fill=color, width=3)
            elif line:
                draw.ellipse((line[0][0] - 2, line[0][1] - 2, line[0][0] + 2, line[0][1] + 2), fill=color)
        legend_x = width - 190
        for index, (_, _, label, color) in enumerate(series):
            y = top + index * 24
            draw.line((legend_x, y + 7, legend_x + 24, y + 7), fill=color, width=3)
            draw.text((legend_x + 30, y), label, fill="black")
        image.save(path)
