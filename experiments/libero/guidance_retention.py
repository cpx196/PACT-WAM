
from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.eval_libero_single import (  # noqa: E402
    DEFAULT_PROMPT,
    _agentview_for_jepa,
    _build_jepa_action_guidance,
    _get_num_video_frames,
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _obs_to_model_input,
    _preencode_prompt_contexts,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    _select_predicted_future_frames,
)
from experiments.libero.libero_utils import (  # noqa: E402
    LIBERO_ENV_RESOLUTION,
    apply_libero_camera,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
)
from experiments.libero.vjepa2_ac_ranker import LiberoACAdapter, VJEPA2ACRanker  # noqa: E402
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor  # noqa: E402
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from fastwam.models.wan22.helpers.loader import _resolve_configs  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402
from libero.libero import benchmark  # noqa: E402


def _as_batch(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device="cpu", dtype=torch.float32).numpy()
    value = np.asarray(value, dtype=np.float32).copy()
    if value.ndim == 2:
        value = value[None, ...]
    if value.ndim != 3:
        raise ValueError(f"Expected action/state tensor [B,H,D] or [H,D], got {value.shape}.")
    return value


def _normalizer_stats(processor: FastWAMProcessor) -> tuple[np.ndarray, np.ndarray]:
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("Guidance retention requires one merged action field.")
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    scale = normalizer.scale.detach().to(device="cpu", dtype=torch.float64).numpy()
    stats = normalizer.stats
    if "std" not in stats:
        raise KeyError("Dataset action statistics do not contain the selected action std.")
    action_std = stats["std"].detach().to(device="cpu", dtype=torch.float64).numpy()
    if scale.ndim != 1 or action_std.ndim != 1 or scale.shape != action_std.shape:
        raise ValueError(
            f"Expected per-dimension action scale/std, got scale={scale.shape}, std={action_std.shape}."
        )
    return scale, action_std


def _normalized_rms(
    delta: np.ndarray,
    *,
    scale: np.ndarray,
    action_std: np.ndarray,
    horizon: int,
) -> float:
    delta = _as_batch(delta)
    if horizon <= 0:
        return math.nan
    delta = delta[:, :horizon].astype(np.float64)
    # Flow states are in the model's normalized action space.  Convert their
    # difference back to raw action units by dividing by the linear normalizer
    # scale; the normalizer offset cancels in a difference.  Then divide by the
    # dataset global per-dimension action std.
    raw_delta = delta / scale.reshape(1, 1, -1)
    standardized = raw_delta / (action_std.reshape(1, 1, -1) + 1e-6)
    return float(np.sqrt(np.mean(np.square(standardized), dtype=np.float64)))


def _guidance_loss(
    guidance_fn,
    normalized_action: Any,
    *,
    device: str,
    dtype: torch.dtype,
) -> float:
    action = torch.as_tensor(normalized_action, device=device, dtype=dtype)
    action = _as_batch(action)
    action = torch.as_tensor(action, device=device, dtype=dtype)
    with torch.no_grad():
        value = guidance_fn(action)
        if value.numel() != 1:
            value = value.mean()
    return float(value.detach().to(device="cpu"))


def _make_initial_latents(
    model: torch.nn.Module,
    *,
    input_height: int,
    input_width: int,
    num_video_frames: int,
    action_horizon: int,
    seed: int,
    rand_device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    latent_t = (
        (num_video_frames - 1) // model.vae.temporal_downsample_factor + 1
    )
    latent_h = input_height // model.vae.upsampling_factor
    latent_w = input_width // model.vae.upsampling_factor
    video_shape = (
        1,
        model.vae.model.z_dim,
        latent_t,
        latent_h,
        latent_w,
    )
    action_shape = (
        1,
        action_horizon,
        model.action_expert.action_dim,
    )
    video_generator = torch.Generator(device=rand_device).manual_seed(seed)
    action_generator = torch.Generator(device=rand_device).manual_seed(seed)
    video_noise = torch.randn(
        video_shape,
        generator=video_generator,
        device=rand_device,
        dtype=torch.float32,
    )
    action_noise = torch.randn(
        action_shape,
        generator=action_generator,
        device=rand_device,
        dtype=torch.float32,
    )
    return video_noise.detach().cpu().clone(), action_noise.detach().cpu().clone()


def _prepare_single_observation(
    *,
    cfg: DictConfig,
    task,
    initial_state: Any,
    seed: int,
) -> tuple[Any, deque, str, Any]:
    env, task_description = get_libero_env(
        task,
        LIBERO_ENV_RESOLUTION,
        seed,
        env_num=1,
    )
    env.reset()
    camera_cfg = cfg.EVALUATION.get("vjepa2_ac", {}).get("camera", None)
    apply_libero_camera(env, camera_cfg)
    rotation = int((camera_cfg or {}).get("image_rotation_degrees", 180))
    obs = env.set_init_state(initial_state)
    frame_interval = int(cfg.data.train.action_video_freq_ratio)
    history = deque(maxlen=frame_interval + 1)
    history.append(get_libero_image(obs, image_rotation_degrees=rotation))
    wait_steps = int(cfg.EVALUATION.get("num_steps_wait", 5))
    for _ in range(wait_steps):
        obs, _, _, _ = env.step(get_libero_dummy_action())
        history.append(get_libero_image(obs, image_rotation_degrees=rotation))
    return env, history, task_description, obs


def _write_plots(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in rows]
    ratio = [row["retention_ratio"] for row in rows]
    loss_before = [row["loss_before"] for row in rows]
    loss_after = [row["loss_after"] for row in rows]
    loss_final = [row["loss_final"] for row in rows]

    retention_path = output_dir / "guidance_retention_ratio.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, ratio, marker="o", linewidth=2, label="final / instant")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Guidance insertion step (zero-based boundary i -> i+1)")
    ax.set_ylabel("Retention ratio")
    ax.set_title("JEPA guidance retention")
    ax.set_xticks(steps)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(retention_path, dpi=180)
    plt.close(fig)

    loss_path = output_dir / "guidance_loss_retention.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, loss_before, marker="o", linewidth=2, label="loss before")
    ax.plot(steps, loss_after, marker="o", linewidth=2, label="loss after")
    ax.plot(steps, loss_final, marker="o", linewidth=2, label="loss final")
    ax.set_xlabel("Guidance insertion step (zero-based boundary i -> i+1)")
    ax.set_ylabel("JEPA loss")
    ax.set_title("JEPA loss before, after, and after denoising")
    ax.set_xticks(steps)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_path, dpi=180)
    plt.close(fig)
    return retention_path, loss_path


def _write_outputs(
    *,
    rows: list[dict[str, Any]],
    traces: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "guidance_retention.csv"
    fieldnames = [
        "step",
        "flow_boundary",
        "instant_delta",
        "final_delta",
        "retention_ratio",
        "loss_before",
        "loss_after",
        "loss_final_base",
        "loss_final_guided",
        "final_loss_gain",
        "causal_loss_retention",
        "loss_final",
        "loss_retention",
        "executed_horizon",
        "x_shape",
        "video_condition_max_abs_diff",
        "baseline_boundary_max_abs_diff",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    npz_path = output_dir / "guidance_retention_states.npz"
    np.savez_compressed(
        npz_path,
        insertion_step=np.asarray([row["step"] for row in rows], dtype=np.int64),
        x_before=np.asarray(traces["x_before"], dtype=np.float32),
        x_after=np.asarray(traces["x_after"], dtype=np.float32),
        x_base=np.asarray(traces["x_base"], dtype=np.float32),
        guided_final_action=np.asarray(traces["guided_final_action"], dtype=np.float32),
        baseline_final_action=np.asarray(traces["baseline_final_action"], dtype=np.float32),
        clean_action_estimate=np.asarray(traces["clean_action_estimate"], dtype=np.float32),
    )

    retention_path, loss_path = _write_plots(rows, output_dir)
    report = {
        "rows": rows,
        "output_dir": str(output_dir),
        "state_archive": str(npz_path),
        "paired_comparison": {
            "same_initial_action_noise": True,
            "same_initial_video_noise": True,
            "same_prompt_context": True,
            "same_video_conditioning": True,
        },
        "interpretation": {
            "most_washed_out_step": min(rows, key=lambda row: row["retention_ratio"])["step"],
            "best_retained_step": max(rows, key=lambda row: row["retention_ratio"])["step"],
            "most_washed_out_loss_step": min(rows, key=lambda row: row["loss_retention"])["step"],
            "best_loss_retained_step": max(rows, key=lambda row: row["loss_retention"])["step"],
            "best_final_loss_gain_step": max(rows, key=lambda row: row["final_loss_gain"])["step"],
            "worst_final_loss_gain_step": min(rows, key=lambda row: row["final_loss_gain"])["step"],
            "best_causal_loss_retention_step": max(rows, key=lambda row: row["causal_loss_retention"])["step"],
            "worst_causal_loss_retention_step": min(rows, key=lambda row: row["causal_loss_retention"])["step"],
        },
    }
    report_json = output_dir / "guidance_retention_report.json"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    report_txt = output_dir / "guidance_retention_report.txt"
    with report_txt.open("w", encoding="utf-8") as handle:
        handle.write("Guidance Retention Experiment\n")
        handle.write("================================\n")
        handle.write("Step uses zero-based flow boundary i -> i+1.\n")
        handle.write("The immediate correction is measured after the ordinary scheduler update and before the next denoising step.\n")
        handle.write("Flow state differences are converted from normalized action space to raw action units and divided by global dataset action std.\n\n")
        for row in rows:
            handle.write(
                "step={step}: instant={instant_delta:.8f}, final={final_delta:.8f}, "
                "retention={retention_ratio:.8f}, loss_before={loss_before:.8f}, "
                "loss_after={loss_after:.8f}, loss_final_base={loss_final_base:.8f}, "
                "loss_final_guided={loss_final_guided:.8f}, final_loss_gain={final_loss_gain:.8f}, "
                "causal_loss_retention={causal_loss_retention:.8f}, "
                "loss_final={loss_final:.8f}, loss_retention={loss_retention:.8f}\n".format(**row)
            )
        handle.write("\n")
        handle.write(
            "Most wash-out by action retention: step {}\n".format(
                report["interpretation"]["most_washed_out_step"]
            )
        )
        handle.write(
            "Best action retention: step {}\n".format(
                report["interpretation"]["best_retained_step"]
            )
        )
        handle.write(
            "Most wash-out by loss retention: step {}\n".format(
                report["interpretation"]["most_washed_out_loss_step"]
            )
        )
        handle.write(
            "Best loss retention: step {}\n".format(
                report["interpretation"]["best_loss_retained_step"]
            )
        )
        handle.write(
            "Best final loss gain: step {}\n".format(
                report["interpretation"]["best_final_loss_gain_step"]
            )
        )
        handle.write(
            "Best causal loss retention: step {}\n".format(
                report["interpretation"]["best_causal_loss_retention_step"]
            )
        )

    return {
        "csv": str(csv_path),
        "states": str(npz_path),
        "retention_plot": str(retention_path),
        "loss_plot": str(loss_path),
        "report_json": str(report_json),
        "report_txt": str(report_txt),
    }


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def run_experiment(cfg: DictConfig) -> dict[str, Any]:
    if cfg.get("seed") is None:
        raise ValueError("A fixed cfg.seed is required for paired retention experiments.")
    seed = int(cfg.seed)
    set_global_seed(seed, get_worker_init_fn=False)

    steps = [4, 5, 6, 7, 8, 9]
    num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", 10))
    if num_inference_steps != 10:
        raise ValueError(
            "Guidance Retention Experiment requires exactly 10 action flow steps."
        )
    if any(step < 0 or step >= num_inference_steps for step in steps):
        raise ValueError("All insertion steps must be valid zero-based flow step indices.")

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model_cfg = cfg.model
    prompt_cache = None
    if bool(cfg.EVALUATION.get("offload_text_encoder", False)):
        if not bool(cfg.model.get("load_text_encoder", True)):
            raise ValueError(
                "EVALUATION.offload_text_encoder requires model.load_text_encoder=true."
            )
        prompt_cache = _preencode_prompt_contexts(
            cfg,
            device=model_device,
            dtype=model_dtype,
        )
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.load_text_encoder = False

    model = instantiate(model_cfg, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()
    if prompt_cache is not None:
        model._libero_prompt_context_cache = prompt_cache

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    scale, action_std = _normalizer_stats(processor)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    action_horizon = (
        int(cfg.data.train.num_frames) - 1
        if action_horizon_cfg is None
        else int(action_horizon_cfg)
    )
    replan_steps = int(
        cfg.EVALUATION.get(
            "replan_steps",
            cfg.EVALUATION.get("vjepa2_ac", {}).get("replan_steps", 10),
        )
    )
    if bool(cfg.EVALUATION.get("vjepa2_ac", {}).get("enabled", False)):
        replan_steps = int(cfg.EVALUATION.vjepa2_ac.get("replan_steps", replan_steps))
    executed_horizon = min(replan_steps, action_horizon)
    if executed_horizon <= 0:
        raise ValueError("replan_steps/action_horizon must yield a positive executed horizon.")

    video_size = cfg.data.train.get("video_size", [224, 224])
    input_height, input_width = int(video_size[0]), int(video_size[1])
    num_video_frames = _get_num_video_frames(cfg)
    rand_device = str(cfg.EVALUATION.get("rand_device", "cpu"))
    initial_video_latents, initial_action_latents = _make_initial_latents(
        model,
        input_height=input_height,
        input_width=input_width,
        num_video_frames=num_video_frames,
        action_horizon=action_horizon,
        seed=seed,
        rand_device=rand_device,
    )

    suite_name = str(cfg.EVALUATION.task_suite_name)
    task_id = int(cfg.EVALUATION.task_id)
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    if len(initial_states) == 0:
        raise RuntimeError("LIBERO task has no initial states.")
    env, history, task_description, obs = _prepare_single_observation(
        cfg=cfg,
        task=task,
        initial_state=initial_states[0],
        seed=seed,
    )

    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_width,
        height=input_height,
        device=model_device,
        dtype=model.torch_dtype,
    )
    prompt = DEFAULT_PROMPT.format(task=task_description)
    common_kwargs: dict[str, Any] = {
        "prompt": None,
        "input_image": image,
        "action_horizon": action_horizon,
        "num_video_frames": num_video_frames,
        "num_action_candidates": 1,
        "proprio": proprio,
        "context": None,
        "context_mask": None,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": seed,
        "rand_device": rand_device,
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
        "compile_action_infer": bool(cfg.EVALUATION.get("compile_action_infer", False)),
        "action_infer_mode": "idm",
        "initial_video_latents": initial_video_latents,
        "initial_action_latents": initial_action_latents,
        "action_flow_trace": True,
    }
    if bool(cfg.EVALUATION.get("offload_text_encoder", False)):
        prompt_context = model._libero_prompt_context_cache[prompt]
        common_kwargs["context"] = prompt_context[0].to(
            device=model_device,
            dtype=model.torch_dtype,
            non_blocking=True,
        )
        common_kwargs["context_mask"] = prompt_context[1].to(
            device=model_device,
            dtype=torch.bool,
            non_blocking=True,
        )
    else:
        common_kwargs["prompt"] = prompt

    ac_cfg = cfg.EVALUATION.get("vjepa2_ac", {})
    if not bool(ac_cfg.get("enabled", False)):
        raise ValueError("EVALUATION.vjepa2_ac.enabled must be true for this experiment.")
    ac_adapter = LiberoACAdapter(
        low_level_steps_per_ac_step=int(
            ac_cfg.get("low_level_steps_per_ac_step", cfg.data.train.action_video_freq_ratio)
        ),
        max_gripper_width=float(ac_cfg.get("max_gripper_width", 0.08)),
        osc_position_scale=float(ac_cfg.get("osc_position_scale", 0.05)),
        osc_rotation_scale=float(ac_cfg.get("osc_rotation_scale", 0.5)),
    )
    ac_dtype_name = str(ac_cfg.get("dtype", "float32")).strip().lower()
    ac_dtype = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }.get(ac_dtype_name)
    if ac_dtype is None:
        raise ValueError(f"Unsupported V-JEPA dtype: {ac_dtype_name}")
    ac_device = model_device if ac_cfg.get("device") is None else str(ac_cfg.device)
    ac_ranker = VJEPA2ACRanker(
        checkpoint_path=os.path.expanduser(os.path.expandvars(str(ac_cfg.checkpoint_path))),
        vjepa2_repo=os.path.expanduser(os.path.expandvars(str(ac_cfg.repo_path))),
        device=ac_device,
        dtype=ac_dtype,
        max_ac_steps=int(ac_cfg.get("max_ac_steps", 8)),
    )

    with torch.no_grad():
        baseline = model.infer_action(
            **common_kwargs,
            action_guidance_fn=None,
            action_guidance_last_steps=0,
            action_guidance_after_steps=None,
            action_guidance_step_size=0.0,
            action_guidance_horizon=None,
            action_guidance_verify_descent=False,
        )
        predicted_video = model._decode_latents(
            baseline["video_latents"],
            tiled=bool(cfg.EVALUATION.get("tiled", False)),
        )
    predicted_future_frames = _select_predicted_future_frames(predicted_video, cfg)
    previous_jepa_image = _agentview_for_jepa(history[0], processor)
    current_jepa_image = _agentview_for_jepa(imgs, processor)
    guidance_fn, guidance_horizon = _build_jepa_action_guidance(
        obs=obs,
        previous_jepa_image=previous_jepa_image,
        current_jepa_image=current_jepa_image,
        predicted_future_frames=predicted_future_frames,
        processor=processor,
        ac_ranker=ac_ranker,
        ac_adapter=ac_adapter,
        cfg=cfg,
        model_device=model_device,
    )

    baseline_trace = {
        int(item["step"]): item for item in baseline["action_flow_trace"]
    }
    baseline_final = _as_batch(baseline["action"])
    # Evaluate the paired baseline final action with the exact same JEPA
    # guidance closure (observation, current latent, WAM future target,
    # action horizon, AC steps, and normalization) used by guided runs.
    # The future target is not regenerated here.
    loss_final_base = _guidance_loss(
        guidance_fn,
        baseline_final,
        device=model_device,
        dtype=ac_ranker.dtype,
    )
    rows: list[dict[str, Any]] = []
    x_before_records = []
    x_after_records = []
    x_base_records = []
    clean_estimate_records = []
    guided_final_records = []
    baseline_final_records = []
    max_video_diff = 0.0

    for insertion_step in steps:
        guided_kwargs = dict(common_kwargs)
        guided = model.infer_action(
            **guided_kwargs,
            action_guidance_fn=guidance_fn,
            action_guidance_last_steps=0,
            action_guidance_after_steps=[insertion_step],
            action_guidance_step_size=0.02,
            action_guidance_horizon=guidance_horizon,
            action_guidance_verify_descent=True,
        )
        max_video_diff = max(
            max_video_diff,
            float(
                (
                    guided["video_latents"].detach().float()
                    - baseline["video_latents"].detach().float()
                )
                .abs()
                .max()
                .cpu()
            ),
        )
        guided_trace = {
            int(item["step"]): item for item in guided["action_flow_trace"]
        }
        if insertion_step not in guided_trace:
            raise RuntimeError(f"Missing guided flow trace for insertion step {insertion_step}.")
        if insertion_step not in baseline_trace:
            raise RuntimeError(f"Missing baseline flow trace for insertion step {insertion_step}.")
        guided_item = guided_trace[insertion_step]
        baseline_item = baseline_trace[insertion_step]
        if not guided_item["guidance_applied"]:
            raise RuntimeError(f"Guidance was not applied at step {insertion_step}.")

        diagnostics = guided.get("action_guidance", [])
        if len(diagnostics) != 1:
            raise RuntimeError(
                f"Expected exactly one guidance diagnostic at step {insertion_step}, got {diagnostics}."
            )
        diagnostic = diagnostics[0]
        loss_before = float(diagnostic["loss"])
        loss_after = diagnostic["loss_after_clean_step"]
        if loss_after is None:
            raise RuntimeError("Experiment requires verify_descent to obtain loss_after.")
        loss_after = float(loss_after)
        guided_final = _as_batch(guided["action"])
        loss_final_guided = _guidance_loss(
            guidance_fn,
            guided_final,
            device=model_device,
            dtype=ac_ranker.dtype,
        )
        instant_delta = _normalized_rms(
            guided_item["x_after"] - guided_item["x_before"],
            scale=scale,
            action_std=action_std,
            horizon=executed_horizon,
        )
        final_delta = _normalized_rms(
            guided_final - baseline_final,
            scale=scale,
            action_std=action_std,
            horizon=executed_horizon,
        )
        eps = 1e-8
        retention_ratio = final_delta / (instant_delta + eps)
        final_loss_gain = loss_final_base - loss_final_guided
        # Causal retention compares the paired guided-vs-baseline final gain
        # with the immediate JEPA improvement caused by this guidance.
        causal_loss_retention = final_loss_gain / (
            loss_before - loss_after + eps
        )
        # Keep the original field for compatibility; it is the guided final loss.
        loss_final = loss_final_guided
        loss_retention = (loss_before - loss_final) / (loss_before - loss_after + eps)
        baseline_boundary_error = float(
            np.abs(
                guided_item["x_base"].detach().float().numpy()
                - baseline_item["x_base"].detach().float().numpy()
            ).max()
        )
        row = {
            "step": insertion_step,
            "flow_boundary": f"{insertion_step}->{insertion_step + 1}",
            "instant_delta": instant_delta,
            "final_delta": final_delta,
            "retention_ratio": retention_ratio,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "loss_final_base": loss_final_base,
            "loss_final_guided": loss_final_guided,
            "final_loss_gain": final_loss_gain,
            "causal_loss_retention": causal_loss_retention,
            "loss_final": loss_final,
            "loss_retention": loss_retention,
            "executed_horizon": executed_horizon,
            "x_shape": str(list(guided_item["x_after"].shape)),
            "video_condition_max_abs_diff": max_video_diff,
            "baseline_boundary_max_abs_diff": baseline_boundary_error,
        }
        rows.append(row)
        x_before_records.append(guided_item["x_before"].numpy())
        x_after_records.append(guided_item["x_after"].numpy())
        x_base_records.append(baseline_item["x_base"].numpy())
        clean_estimate_records.append(guided_item["clean_action_estimate"].numpy())
        guided_final_records.append(guided_final)
        baseline_final_records.append(baseline_final)

    output_override = cfg.EVALUATION.get("guidance_retention_output_dir", None)
    output_dir = (
        Path(os.path.expanduser(os.path.expandvars(str(output_override)))).resolve()
        if output_override is not None
        else Path(os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.output_dir)))).resolve()
        / "guidance_retention"
    )
    paths = _write_outputs(
        rows=rows,
        traces={
            "x_before": x_before_records,
            "x_after": x_after_records,
            "x_base": x_base_records,
            "clean_action_estimate": clean_estimate_records,
            "guided_final_action": guided_final_records,
            "baseline_final_action": baseline_final_records,
        },
        output_dir=output_dir,
    )
    report = {
        "task_suite": suite_name,
        "task_id": task_id,
        "task_description": task_description,
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "replan_steps": replan_steps,
        "executed_horizon": executed_horizon,
        "guidance_step_size": 0.02,
        "guidance_horizon": guidance_horizon,
        "flow_state_representation": "normalized action-flow state; delta converted to raw action units then divided by global dataset action std",
        "flow_boundary_definition": "step i means the i -> i+1 scheduler boundary; x_before is ordinary scheduler output before additive correction",
        "paired_checks": {
            "same_initial_action_latents": True,
            "same_initial_video_latents": True,
            "same_context": True,
            "max_video_condition_abs_diff": max_video_diff,
        },
        "paths": paths,
        "rows": rows,
    }
    (output_dir / "guidance_retention_metadata.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    close_fn = getattr(env, "close", None)
    if close_fn is not None:
        close_fn()
    return report


if __name__ == "__main__":
    run_experiment()
