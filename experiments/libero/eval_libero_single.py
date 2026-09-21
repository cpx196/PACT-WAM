import gc
import inspect
import json
import logging
import os
import pickle
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

# try:
#     import rootutils

#     rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)
# except ModuleNotFoundError:
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    apply_libero_camera,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_prediction_video,
    save_rollout_video,
)
from experiments.libero.worker_pool import pop_task, write_worker_status
from experiments.libero.action_denoise_convergence import ActionDenoiseConvergenceLogger
from experiments.libero.vjepa2_ac_ranker import LiberoACAdapter, VJEPA2ACRanker
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from fastwam.utils.pytorch_utils import set_global_seed
from libero.libero import benchmark, get_libero_path

DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _cuda_memory_snapshot(device: str) -> Optional[dict[str, float]]:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return None
    torch.cuda.synchronize(resolved)
    return {
        "allocated_gib": torch.cuda.memory_allocated(resolved) / 1024**3,
        "reserved_gib": torch.cuda.memory_reserved(resolved) / 1024**3,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(resolved) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(resolved) / 1024**3,
    }


def _log_cuda_memory(label: str, device: str) -> None:
    snapshot = _cuda_memory_snapshot(device)
    if snapshot is None:
        return
    logging.info(
        "%s CUDA memory: allocated=%.3f GiB reserved=%.3f GiB peak=%.3f GiB",
        label,
        snapshot["allocated_gib"],
        snapshot["reserved_gib"],
        snapshot["peak_allocated_gib"],
    )


def _synchronize_cuda(device: str) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)


def _evaluation_task_keys(cfg: DictConfig) -> list[tuple[str, int]]:
    if os.environ.get("LIBERO_WORKER_MODE") == "1":
        value = os.environ.get("LIBERO_WORKER_TASK_FILE")
        if value is None or not value.strip():
            raise ValueError("LIBERO_WORKER_TASK_FILE must be set in worker mode.")
        task_file = Path(os.path.expanduser(os.path.expandvars(value))).resolve()
        lines = task_file.read_text(encoding="utf-8").splitlines()
        keys = []
        for line in lines:
            line = line.strip()
            if line:
                suite_name, task_id = line.split(",", 1)
                keys.append((suite_name, int(task_id)))
        return keys
    return [(str(cfg.EVALUATION.task_suite_name), int(cfg.EVALUATION.task_id))]


def _evaluation_prompts(cfg: DictConfig) -> list[str]:
    benchmark_dict = benchmark.get_benchmark_dict()
    prompts = []
    seen = set()
    suites = {}
    for suite_name, task_id in _evaluation_task_keys(cfg):
        if suite_name not in suites:
            suites[suite_name] = benchmark_dict[suite_name]()
        prompt = DEFAULT_PROMPT.format(task=suites[suite_name].get_task(task_id).language)
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
    if not prompts:
        raise ValueError("No LIBERO task prompts were found for text pre-encoding.")
    return prompts


def _preencode_prompt_contexts(
    cfg: DictConfig,
    *,
    device: str,
    dtype: torch.dtype,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Encode all worker prompts with standalone T5, then release it completely."""
    prompts = _evaluation_prompts(cfg)
    model_cfg = cfg.model
    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=str(model_cfg.model_id),
        tokenizer_model_id=str(model_cfg.tokenizer_model_id),
        redirect_common_files=bool(model_cfg.get("redirect_common_files", True)),
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()
    context_len = int(model_cfg.get("tokenizer_max_len", cfg.data.train.get("context_len", 128)))
    batch_size = int(cfg.EVALUATION.get("text_encoder_batch_size", 8))
    if batch_size <= 0:
        raise ValueError("EVALUATION.text_encoder_batch_size must be positive.")

    logging.info(
        "Loading standalone T5 on %s to pre-encode %d LIBERO prompts.",
        device,
        len(prompts),
    )
    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=dtype,
        device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=context_len,
        clean="whitespace",
    )
    cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
            ids = ids.to(device)
            mask = mask.to(device=device, dtype=torch.bool)
            context = text_encoder(ids, mask)
            seq_lens = mask.gt(0).sum(dim=1).long()
            for index, prompt in enumerate(batch_prompts):
                context[index, seq_lens[index] :] = 0
                cache[prompt] = (
                    context[index : index + 1].to(device="cpu", dtype=dtype).contiguous(),
                    torch.ones_like(mask[index : index + 1], device="cpu"),
                )

    _log_cuda_memory("Standalone T5 peak", device)
    del context, ids, mask, text_encoder, tokenizer
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    logging.info("Released standalone T5; cached %d prompt embeddings on CPU.", len(cache))
    _log_cuda_memory("After releasing standalone T5", device)
    return cache


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    msg = (
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )
    raise FileNotFoundError(msg)


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    model.load_checkpoint(ckpt)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)
    return

    # deprecated legacy checkpoint loading
    payload = torch.load(ckpt, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Legacy checkpoint payload must be dict, got: {type(payload)}")

    if "mot" in payload and hasattr(model, "mot"):
        missing, unexpected = model.mot.load_state_dict(payload["mot"], strict=False)
        logging.warning(
            "Loaded fallback `mot` state_dict with strict=False. Missing=%d Unexpected=%d",
            len(missing),
            len(unexpected),
        )
        return

    state_dict = None
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            state_dict = value
            break
    if state_dict is None and all(torch.is_tensor(v) for v in payload.values()):
        state_dict = payload
    if state_dict is None:
        raise ValueError(f"Cannot parse legacy checkpoint keys from: {ckpt}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.warning(
        "Loaded fallback model state_dict with strict=False. Missing=%d Unexpected=%d",
        len(missing),
        len(unexpected),
    )


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _normalize_proprio(
    proprio: np.ndarray,
    processor: FastWAMProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]

    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    camera_cfg = cfg.EVALUATION.get("vjepa2_ac", {}).get("camera", {})
    image_rotation_degrees = int(camera_cfg.get("image_rotation_degrees", 180))
    imgs = get_libero_image(obs, image_rotation_degrees=image_rotation_degrees)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}")
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    num_cameras = processor.num_output_cameras
    if num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}.")

    actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
    expected_h, expected_w = int(height), int(width)
    image_shapes = [meta["shape"] for meta in image_meta]
    assert actual_h == expected_h and actual_w == expected_w, (
        "Input image size mismatch after per-camera resize + concat: "
        f"got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) "
        f"from data.train.video_size={[expected_h, expected_w]}; "
        f"shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
    )

    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0

    proprio = _normalize_proprio(_extract_sim_state(obs), processor)

    return x, proprio, imgs


def _agentview_for_jepa(imgs: dict[str, Any], processor: FastWAMProcessor) -> np.ndarray:
    """Apply the same primary-camera crop used by the FastWAM input path."""
    image_meta = processor.shape_meta["images"]
    if not image_meta:
        raise ValueError("shape_meta.images must contain a primary camera for V-JEPA2-AC.")
    shape = image_meta[0]["shape"]
    if len(shape) != 3:
        raise ValueError(f"Primary camera shape must be [C,H,W], got {shape}")
    return _center_crop_resize(imgs["image"], width=int(shape[2]), height=int(shape[1]))


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state


def _denormalize_action(action: torch.Tensor, processor: FastWAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _build_jepa_action_guidance(
    *,
    obs: dict,
    previous_jepa_image: Any,
    current_jepa_image: Any,
    predicted_future_frames: list[Image.Image],
    processor: FastWAMProcessor,
    ac_ranker: VJEPA2ACRanker,
    ac_adapter: LiberoACAdapter,
    cfg: DictConfig,
    model_device: str,
) -> tuple[Any, int]:
    """Build a differentiable normalized-action -> JEPA-energy closure."""
    resolved_model_device = torch.device(model_device)

    def canonical_device(device: torch.device) -> tuple[str, Optional[int]]:
        if device.type != "cuda":
            return device.type, device.index
        index = torch.cuda.current_device() if device.index is None else device.index
        return device.type, index

    if canonical_device(resolved_model_device) != canonical_device(ac_ranker.device):
        raise ValueError(
            "JEPA action guidance requires FastWAM and V-JEPA2-AC on the same device; "
            f"got {resolved_model_device} and {ac_ranker.device}."
        )

    guidance_cfg = cfg.EVALUATION.vjepa2_ac.guidance
    num_ac_steps = int(guidance_cfg.get("ac_steps", 1))
    stride = ac_adapter.low_level_steps_per_ac_step
    guidance_horizon = num_ac_steps * stride
    if len(predicted_future_frames) <= num_ac_steps:
        raise ValueError(
            f"Guidance needs {num_ac_steps + 1} future frames, got "
            f"{len(predicted_future_frames)}."
        )
    target_future_clips = [
        [predicted_future_frames[step], predicted_future_frames[step + 1]]
        for step in range(num_ac_steps)
    ]
    current_rep, target_reps = ac_ranker.encode_guidance_context(
        current_clip=[previous_jepa_image, current_jepa_image],
        target_future_clips=target_future_clips,
    )
    initial_state = torch.as_tensor(
        ac_adapter.state_from_observation(obs),
        device=ac_ranker.device,
        dtype=ac_ranker.dtype,
    )

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("JEPA guidance expects one merged action normalizer.")
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    scale = normalizer.scale.to(device=ac_ranker.device, dtype=ac_ranker.dtype)
    offset = normalizer.offset.to(device=ac_ranker.device, dtype=ac_ranker.dtype)

    def guidance_energy(normalized_clean_action: torch.Tensor) -> torch.Tensor:
        clean = normalized_clean_action.to(device=ac_ranker.device, dtype=ac_ranker.dtype)
        denormalized = (clean[:, :guidance_horizon] - offset) / scale
        # Match the existing evaluation postprocess without the final discrete
        # gripper binarization: dataset 0=close/1=open -> LIBERO +1=close/-1=open.
        executable = torch.cat(
            (denormalized[..., :6], 1.0 - 2.0 * denormalized[..., 6:7]),
            dim=-1,
        )
        ac_batch = ac_adapter.convert_torch(executable, initial_state)
        return ac_ranker.differentiable_energy(
            current_rep=current_rep,
            target_reps=target_reps,
            ac_actions=ac_batch.actions,
            ac_states=ac_batch.states,
        ).mean()

    return guidance_energy, guidance_horizon


def _compute_trigger_jepa_loss(
    *,
    obs: dict,
    previous_jepa_image: Any,
    current_jepa_image: Any,
    predicted_future_frames: list[Image.Image],
    executable_action: np.ndarray,
    ac_ranker: VJEPA2ACRanker,
    ac_adapter: LiberoACAdapter,
    cfg: DictConfig,
) -> float:
    """Compute one frozen JEPA-vs-WAM consistency loss for a planning point."""
    probe_cfg = cfg.EVALUATION.trigger_probe
    num_ac_steps = int(probe_cfg.get("ac_steps", 2))
    stride = int(ac_adapter.low_level_steps_per_ac_step)
    horizon = num_ac_steps * stride
    if len(predicted_future_frames) <= num_ac_steps:
        raise ValueError(
            f"Trigger probe needs {num_ac_steps + 1} WAM frames, got "
            f"{len(predicted_future_frames)}."
        )
    if executable_action.shape[0] < horizon:
        raise ValueError(
            f"Trigger probe needs {horizon} low-level actions, got "
            f"{executable_action.shape[0]}."
        )
    target_future_clips = [
        [predicted_future_frames[step], predicted_future_frames[step + 1]]
        for step in range(num_ac_steps)
    ]
    current_rep, target_reps = ac_ranker.encode_guidance_context(
        current_clip=[previous_jepa_image, current_jepa_image],
        target_future_clips=target_future_clips,
    )
    initial_state = ac_adapter.state_from_observation(obs)
    ac_batch = ac_adapter.convert(executable_action[None, :horizon], initial_state)
    with torch.no_grad():
        energy = ac_ranker.differentiable_energy(
            current_rep=current_rep,
            target_reps=target_reps,
            ac_actions=torch.from_numpy(ac_batch.actions),
            ac_states=torch.from_numpy(ac_batch.states),
        )
    return float(energy[0].detach().cpu())


def _get_num_video_frames(cfg: DictConfig) -> int:
    return (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1


def _get_replan_steps(cfg: DictConfig) -> int:
    ac_cfg = cfg.EVALUATION.get("vjepa2_ac", {})
    if bool(ac_cfg.get("enabled", False)):
        return int(ac_cfg.get("replan_steps", cfg.EVALUATION.get("replan_steps", 5)))
    return int(cfg.EVALUATION.get("replan_steps", 5))


def _validate_visualize_future_video_cfg(cfg: DictConfig) -> None:
    if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
        return

    action_conditioned = cfg.model.video_dit_config.get("action_conditioned", None)
    if action_conditioned is not False:
        raise ValueError(
            "EVALUATION.visualize_future_video=true requires "
            "model.video_dit_config.action_conditioned=false."
        )

    ac_cfg = cfg.EVALUATION.get("vjepa2_ac", {})
    if bool(ac_cfg.get("enabled", False)):
        guidance_cfg = ac_cfg.get("guidance", {})
        guidance_enabled = bool(guidance_cfg.get("enabled", False))
        if not bool(cfg.EVALUATION.get("offload_text_encoder", False)):
            raise ValueError(
                "V-JEPA2-AC requires EVALUATION.offload_text_encoder=true "
                "to release T5 before loading the inference models."
            )
        if not guidance_enabled and int(ac_cfg.get("num_action_candidates", 8)) < 2:
            raise ValueError("V-JEPA2-AC ranking requires at least two action candidates.")
        candidate_generation_mode = str(
            ac_cfg.get("candidate_generation_mode", "separate")
        ).lower()
        if candidate_generation_mode not in {"separate", "joint"}:
            raise ValueError(
                "EVALUATION.vjepa2_ac.candidate_generation_mode must be "
                f"'separate' or 'joint', got {candidate_generation_mode!r}."
            )
        if guidance_enabled and candidate_generation_mode != "separate":
            raise ValueError(
                "V-JEPA2-AC action guidance requires candidate_generation_mode='separate' "
                "because the target future must exist before guided action sampling."
            )
        if guidance_enabled:
            last_flow_steps = int(guidance_cfg.get("last_flow_steps", 2))
            after_flow_steps_cfg = guidance_cfg.get("after_flow_steps", None)
            after_flow_steps = (
                None
                if after_flow_steps_cfg is None
                else [int(step) for step in after_flow_steps_cfg]
            )
            step_size = float(guidance_cfg.get("step_size", 0.02))
            ac_steps = int(guidance_cfg.get("ac_steps", 1))
            if after_flow_steps is not None:
                if not after_flow_steps:
                    raise ValueError(
                        "vjepa2_ac.guidance.after_flow_steps must not be empty when set."
                    )
                if len(set(after_flow_steps)) != len(after_flow_steps):
                    raise ValueError(
                        "vjepa2_ac.guidance.after_flow_steps must not contain duplicates."
                    )
            elif last_flow_steps <= 0:
                raise ValueError(
                    "vjepa2_ac.guidance.last_flow_steps must be positive when "
                    "after_flow_steps is not set."
                )
            if step_size <= 0:
                raise ValueError("vjepa2_ac.guidance.step_size must be positive.")
            if ac_steps <= 0:
                raise ValueError("vjepa2_ac.guidance.ac_steps must be positive.")
        replan_steps = _get_replan_steps(cfg)
        interval = int(cfg.data.train.action_video_freq_ratio)
        if replan_steps <= 0 or replan_steps % interval != 0:
            raise ValueError(
                "EVALUATION.vjepa2_ac.replan_steps must be a positive multiple of "
                f"data.train.action_video_freq_ratio={interval}, got {replan_steps}."
            )
        ac_steps = replan_steps // interval
        max_ac_steps = int(ac_cfg.get("max_ac_steps", 8))
        if ac_steps > max_ac_steps:
            raise ValueError(
                f"V-JEPA2-AC ranking needs {ac_steps} predictor steps, but "
                f"max_ac_steps={max_ac_steps}."
            )
        if guidance_enabled and int(guidance_cfg.get("ac_steps", 1)) > ac_steps:
            raise ValueError(
                "vjepa2_ac.guidance.ac_steps cannot exceed the number of future "
                f"targets ({ac_steps})."
            )


def _select_predicted_future_frames(pred_video: list[Image.Image], cfg: DictConfig) -> list[Image.Image]:
    if len(pred_video) == 0:
        raise ValueError("`infer_joint` returned an empty predicted video.")

    replan_steps = _get_replan_steps(cfg)
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    keep_frames = 1 + num_future_frames
    return list(pred_video[:keep_frames])


def _get_future_frame_capture_steps(cfg: DictConfig) -> list[int]:
    replan_steps = _get_replan_steps(cfg)
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    return [step_idx * action_video_freq_ratio for step_idx in range(num_future_frames + 1)]


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        images = []
        for value in frame.values():
            value_array = np.array(value) if isinstance(value, Image.Image) else np.array(value, copy=True)
            images.append(value_array)
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def _compute_clip_mean_psnr(
    gt_frames: list[Any],
    pred_frames: list[Any],
    eps: float = 1e-8,
) -> Optional[float]:
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        return None
    assert len(gt_frames) == len(pred_frames), (
        "GT/pred frame count mismatch for PSNR: "
        f"len(gt_frames)={len(gt_frames)} len(pred_frames)={len(pred_frames)}. "
        "This indicates temporal misalignment in future-video capture."
    )
    num_frames = len(gt_frames)

    frame_psnr_values = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        gt_image = _frame_to_rgb_array(gt_frame)
        pred_image = _frame_to_rgb_array(pred_frame)
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_f32 = gt_image.astype(np.float32)
        pred_f32 = pred_image.astype(np.float32)
        mse = float(np.mean((pred_f32 - gt_f32) ** 2))
        psnr = 10.0 * np.log10((255.0 * 255.0) / max(mse, eps))
        frame_psnr_values.append(float(psnr))

    if len(frame_psnr_values) == 0:
        return None
    return float(np.mean(frame_psnr_values))


def _predict_action_chunk(
    obs: dict,
    previous_jepa_image: Any,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    ac_ranker: Optional[VJEPA2ACRanker] = None,
    ac_adapter: Optional[LiberoACAdapter] = None,
    executed_horizon: Optional[int] = None,
    action_convergence_logger: Optional[ActionDenoiseConvergenceLogger] = None,
) -> tuple[np.ndarray, dict, Optional[list[Image.Image]], Optional[dict[str, Any]]]:
    timing_enabled = bool(cfg.EVALUATION.get("timing_enabled", False))
    action_denoise_trace_enabled = bool(cfg.EVALUATION.get("action_denoise_trace", False))
    trigger_probe_enabled = bool(
        cfg.EVALUATION.get("trigger_probe", {}).get("enabled", False)
    )
    ranking_enabled = bool(cfg.EVALUATION.vjepa2_ac.get("enabled", False))
    timings: Optional[dict[str, Any]] = (
        {} if timing_enabled or action_denoise_trace_enabled or trigger_probe_enabled else None
    )
    total_start = (
        time.perf_counter()
        if timing_enabled or action_denoise_trace_enabled or trigger_probe_enabled
        else 0.0
    )
    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    if num_inference_steps_cfg is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
    else:
        num_inference_steps = int(num_inference_steps_cfg)
    prompt_template = DEFAULT_PROMPT
    prompt = prompt_template.format(task=task_description)

    stage_start = time.perf_counter() if timing_enabled else 0.0
    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )
    if timing_enabled:
        _synchronize_cuda(model_device)
        timings["preprocess_s"] = time.perf_counter() - stage_start

    infer_kwargs = {
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    if bool(cfg.EVALUATION.get("offload_text_encoder", False)):
        cache = getattr(model, "_libero_prompt_context_cache", None)
        if cache is None or prompt not in cache:
            raise KeyError(
                "LIBERO prompt was not pre-encoded before FastWAM loading: "
                f"{prompt!r}"
            )
        cached_context, cached_mask = cache[prompt]
        infer_kwargs["context"] = cached_context.to(
            device=model_device,
            dtype=model.torch_dtype,
            non_blocking=True,
        )
        infer_kwargs["context_mask"] = cached_mask.to(
            device=model_device,
            dtype=torch.bool,
            non_blocking=True,
        )
        infer_kwargs["prompt"] = None
    else:
        infer_kwargs["prompt"] = prompt
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    predicted_future_frames = None
    trigger_future_frames = None
    if visualize_future_video:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
    elif "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)

    compile_action_infer = bool(cfg.EVALUATION.get("compile_action_infer", False))
    candidate_generation_mode = str(
        cfg.EVALUATION.vjepa2_ac.get("candidate_generation_mode", "separate")
    ).lower()
    guidance_cfg = cfg.EVALUATION.vjepa2_ac.get("guidance", {})
    guidance_enabled = (
        ranking_enabled
        and ac_ranker is not None
        and bool(guidance_cfg.get("enabled", False))
    )
    split_idm_inference = (
        visualize_future_video
        and candidate_generation_mode == "separate"
        and hasattr(model, "infer_video")
        and hasattr(model, "infer_action_from_video")
    )
    infer_method = (
        model.infer_video
        if split_idm_inference
        else (model.infer_joint if visualize_future_video else model.infer_action)
    )
    infer_signature = inspect.signature(infer_method).parameters
    if "video_sigma_shift" in infer_signature:
        infer_kwargs["video_sigma_shift"] = float(
            cfg.EVALUATION.get("video_sigma_shift", 5.0)
        )
    if "action_sigma_shift" in infer_signature:
        infer_kwargs["action_sigma_shift"] = float(
            cfg.EVALUATION.get("action_sigma_shift", 1.0)
        )
    if action_denoise_trace_enabled and "action_denoise_trace" in inspect.signature(infer_method).parameters:
        infer_kwargs["action_denoise_trace"] = True
    if not visualize_future_video and "action_infer_mode" in inspect.signature(infer_method).parameters:
        infer_kwargs["action_infer_mode"] = str(
            cfg.EVALUATION.get("action_infer_mode", "idm")
        )
    compile_arg = "compile_video_infer" if split_idm_inference else "compile_action_infer"
    if compile_action_infer and compile_arg not in infer_signature:
        raise ValueError(
            f"{type(model).__name__}.{infer_method.__name__} does not support `{compile_arg}`."
        )

    with torch.no_grad():
        if visualize_future_video:
            # `infer_joint` defaults to an expensive debug equivalence check
            # that runs a second action-only inference. Evaluation should only
            # enable it explicitly; otherwise it pollutes rollout latency.
            if not split_idm_inference:
                infer_kwargs["test_action_with_infer_action"] = bool(
                    cfg.EVALUATION.get("test_action_with_infer_action", False)
            ) and not ranking_enabled
            if ranking_enabled and candidate_generation_mode == "joint":
                if "num_action_candidates" not in inspect.signature(model.infer_joint).parameters:
                    raise TypeError(
                        f"{type(model).__name__}.infer_joint does not support batched action candidates."
                    )
                infer_kwargs["num_action_candidates"] = int(
                    cfg.EVALUATION.vjepa2_ac.get("num_action_candidates", 8)
                )
            stage_start = time.perf_counter() if timing_enabled else 0.0
            if split_idm_inference:
                video_signature = inspect.signature(model.infer_video).parameters
                video_infer_kwargs = {
                    key: value for key, value in infer_kwargs.items() if key in video_signature
                }
                pred = model.infer_video(
                    **video_infer_kwargs,
                    compile_video_infer=compile_action_infer,
                    decode_video=False,
                )
            else:
                pred = model.infer_joint(
                    **infer_kwargs,
                    compile_action_infer=compile_action_infer,
                )
            generation_ids = []
            if action_convergence_logger is not None:
                generation_id = action_convergence_logger.record(
                    trace=pred.get("action_denoise_trace", []),
                    clean_action_estimates=pred.get("action_denoise_estimates", []),
                    final_action=pred.get("action_denoise_final_action"),
                    executed_horizon=(
                        action_horizon if executed_horizon is None else executed_horizon
                    ),
                )
                if generation_id is not None:
                    generation_ids.append(generation_id)
            if timing_enabled:
                _synchronize_cuda(model_device)
                timings[
                    "infer_video_s" if split_idm_inference else "infer_joint_s"
                ] = time.perf_counter() - stage_start
            # Preserve the original IDM execution order: video denoise, action
            # denoise, then VAE decode. Guidance is the only exception because
            # its loss explicitly consumes decoded future frames.
            if split_idm_inference and guidance_enabled:
                stage_start = time.perf_counter() if timing_enabled else 0.0
                pred["video"] = model.decode_video_latents(
                    pred["video_latents"],
                    tiled=bool(cfg.EVALUATION.get("tiled", False)),
                )
                if timing_enabled:
                    _synchronize_cuda(model_device)
                    timings["decode_video_s"] = time.perf_counter() - stage_start
            if "video" in pred:
                trigger_future_frames = list(pred["video"])
                predicted_future_frames = _select_predicted_future_frames(pred["video"], cfg)
            if split_idm_inference or (
                ranking_enabled and candidate_generation_mode == "separate"
            ):
                action_signature = inspect.signature(model.infer_action).parameters
                if "num_action_candidates" not in action_signature:
                    raise TypeError(
                        f"{type(model).__name__}.infer_action does not support batched action candidates."
                    )
                action_infer_kwargs = {
                    key: value for key, value in infer_kwargs.items() if key in action_signature
                }
                if "video_sigma_shift" in action_signature:
                    action_infer_kwargs["video_sigma_shift"] = float(
                        cfg.EVALUATION.get("video_sigma_shift", 5.0)
                    )
                if "action_sigma_shift" in action_signature:
                    action_infer_kwargs["action_sigma_shift"] = float(
                        cfg.EVALUATION.get("action_sigma_shift", 1.0)
                    )
                if guidance_enabled:
                    required_guidance_args = {
                        "action_guidance_fn",
                        "action_guidance_last_steps",
                        "action_guidance_after_steps",
                        "action_guidance_step_size",
                        "action_guidance_horizon",
                        "action_guidance_verify_descent",
                    }
                    missing_guidance_args = required_guidance_args - set(action_signature)
                    if missing_guidance_args:
                        raise TypeError(
                            f"{type(model).__name__}.infer_action does not support JEPA "
                            f"guidance arguments: {sorted(missing_guidance_args)}"
                        )
                    if ac_adapter is None:
                        raise RuntimeError("JEPA action guidance requires LiberoACAdapter.")
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
                    action_infer_kwargs.update(
                        num_action_candidates=1,
                        action_guidance_fn=guidance_fn,
                        action_guidance_last_steps=int(
                            guidance_cfg.get("last_flow_steps", 2)
                        ),
                        action_guidance_after_steps=guidance_cfg.get(
                            "after_flow_steps", None
                        ),
                        action_guidance_step_size=float(guidance_cfg.get("step_size", 0.02)),
                        action_guidance_horizon=guidance_horizon,
                        action_guidance_verify_descent=bool(
                            guidance_cfg.get("verify_descent", True)
                        ),
                    )
                else:
                    action_infer_kwargs["num_action_candidates"] = (
                        1
                        if not ranking_enabled
                        else int(cfg.EVALUATION.vjepa2_ac.get("num_action_candidates", 8))
                    )
                if "action_infer_mode" in action_signature:
                    action_infer_kwargs["action_infer_mode"] = str(
                        cfg.EVALUATION.get("action_infer_mode", "idm")
                    )
                if "action_denoise_trace" in action_signature:
                    action_infer_kwargs["action_denoise_trace"] = action_denoise_trace_enabled
                if "action_response_trace" in action_signature:
                    action_infer_kwargs["action_response_trace"] = trigger_probe_enabled
                stage_start = time.perf_counter() if timing_enabled else 0.0
                if split_idm_inference:
                    action_pred = model.infer_action_from_video(
                        video_latents=pred["video_latents"],
                        **action_infer_kwargs,
                        compile_action_infer=compile_action_infer,
                    )
                else:
                    action_pred = model.infer_action(
                        **action_infer_kwargs,
                        compile_action_infer=compile_action_infer,
                    )
                if action_convergence_logger is not None:
                    generation_id = action_convergence_logger.record(
                        trace=action_pred.get("action_denoise_trace", []),
                        clean_action_estimates=action_pred.get("action_denoise_estimates", []),
                        final_action=action_pred.get("action_denoise_final_action"),
                        executed_horizon=(
                            action_horizon if executed_horizon is None else executed_horizon
                        ),
                    )
                    if generation_id is not None:
                        generation_ids.append(generation_id)
                if timing_enabled:
                    _synchronize_cuda(model_device)
                    timings["infer_action_s"] = time.perf_counter() - stage_start
                pred["action"] = action_pred["action"]
                if action_denoise_trace_enabled and timings is not None:
                    timings["action_denoise_trace"] = action_pred.get(
                        "action_denoise_trace", []
                    )
                    timings["action_denoise_generation_ids"] = generation_ids
                if trigger_probe_enabled and timings is not None:
                    timings["action_response_trace"] = action_pred.get(
                        "action_response_trace", []
                    )
                if guidance_enabled:
                    logging.info(
                        "V-JEPA2-AC action guidance diagnostics=%s",
                        action_pred.get("action_guidance", []),
                    )
            if split_idm_inference and "video" not in pred:
                stage_start = time.perf_counter() if timing_enabled else 0.0
                pred["video"] = model.decode_video_latents(
                    pred["video_latents"],
                    tiled=bool(cfg.EVALUATION.get("tiled", False)),
                )
                if timing_enabled:
                    _synchronize_cuda(model_device)
                    timings["decode_video_s"] = time.perf_counter() - stage_start
                predicted_future_frames = _select_predicted_future_frames(pred["video"], cfg)
                trigger_future_frames = list(pred["video"])
        else:
            if "action_denoise_trace" in inspect.signature(model.infer_action).parameters:
                infer_kwargs["action_denoise_trace"] = action_denoise_trace_enabled
            if "action_response_trace" in inspect.signature(model.infer_action).parameters:
                infer_kwargs["action_response_trace"] = trigger_probe_enabled
            stage_start = time.perf_counter() if timing_enabled else 0.0
            pred = model.infer_action(
                **infer_kwargs,
                compile_action_infer=compile_action_infer,
            )
            generation_ids = []
            if action_convergence_logger is not None:
                generation_id = action_convergence_logger.record(
                    trace=pred.get("action_denoise_trace", []),
                    clean_action_estimates=pred.get("action_denoise_estimates", []),
                    final_action=pred.get("action_denoise_final_action"),
                    executed_horizon=(
                        action_horizon if executed_horizon is None else executed_horizon
                    ),
                )
                if generation_id is not None:
                    generation_ids.append(generation_id)
            if action_denoise_trace_enabled and timings is not None:
                timings["action_denoise_trace"] = pred.get(
                    "action_denoise_trace", []
                )
                timings["action_denoise_generation_ids"] = generation_ids
            if trigger_probe_enabled and timings is not None:
                timings["action_response_trace"] = pred.get(
                    "action_response_trace", []
                )
            if timing_enabled:
                _synchronize_cuda(model_device)
                timings["infer_action_s"] = time.perf_counter() - stage_start
    action = pred["action"]  # [T, D]

    stage_start = time.perf_counter() if timing_enabled else 0.0
    action = _denormalize_action(action, processor)

    # The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    if timing_enabled:
        timings["action_postprocess_s"] = time.perf_counter() - stage_start
    if ranking_enabled and ac_ranker is not None and not guidance_enabled:
        if ac_adapter is None or predicted_future_frames is None:
            raise RuntimeError("V-JEPA2-AC ranking requires its adapter and a predicted future frame.")
        rank_horizon = min(
            action.shape[1],
            (len(predicted_future_frames) - 1) * int(cfg.data.train.action_video_freq_ratio),
        )
        if rank_horizon <= 0:
            raise ValueError("The FastWAM future clip does not contain a future target frame.")
        initial_state = ac_adapter.state_from_observation(obs)
        ac_batch = ac_adapter.convert(action[:, :rank_horizon], initial_state)
        current_jepa_image = _agentview_for_jepa(imgs, processor)
        num_ac_steps = ac_batch.actions.shape[1]
        target_future_clips = [
            [predicted_future_frames[step], predicted_future_frames[step + 1]]
            for step in range(num_ac_steps)
        ]
        stage_start = time.perf_counter() if timing_enabled else 0.0
        selected, energies = ac_ranker.rank(
            current_clip=[previous_jepa_image, current_jepa_image],
            target_future_clips=target_future_clips,
            ac_actions=ac_batch.actions,
            ac_states=ac_batch.states,
            timing_enabled=timing_enabled,
        )
        if timing_enabled:
            _synchronize_cuda(str(ac_ranker.device))
            timings["jepa_rank_s"] = time.perf_counter() - stage_start
            timings.update(ac_ranker.last_timing)
        if not getattr(ac_ranker, "_libero_logged_inference_memory", False):
            _log_cuda_memory("FastWAM + V-JEPA2-AC first inference", model_device)
            ac_ranker._libero_logged_inference_memory = True
        logging.info(
            "V-JEPA2-AC selected candidate=%d energies=%s rank_horizon=%d",
            selected,
            [round(float(value), 6) for value in energies],
            rank_horizon,
        )
        action = action[selected]
    else:
        action = action[0]
    if trigger_probe_enabled:
        if ac_ranker is None or ac_adapter is None or trigger_future_frames is None:
            raise RuntimeError(
                "Trigger probe requires a loaded V-JEPA2-AC model, adapter, and WAM future."
            )
        trigger_start = time.perf_counter()
        jepa_loss = _compute_trigger_jepa_loss(
            obs=obs,
            previous_jepa_image=previous_jepa_image,
            current_jepa_image=_agentview_for_jepa(imgs, processor),
            predicted_future_frames=trigger_future_frames,
            executable_action=action,
            ac_ranker=ac_ranker,
            ac_adapter=ac_adapter,
            cfg=cfg,
        )
        assert timings is not None
        timings["trigger_probe_jepa_loss"] = jepa_loss
        timings["trigger_probe_jepa_s"] = time.perf_counter() - trigger_start
    if timing_enabled or action_denoise_trace_enabled or trigger_probe_enabled:
        timings["total_s"] = time.perf_counter() - total_start
    return action, imgs, predicted_future_frames, timings


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    ac_ranker: Optional[VJEPA2ACRanker] = None,
    ac_adapter: Optional[LiberoACAdapter] = None,
    action_convergence_logger: Optional[ActionDenoiseConvergenceLogger] = None,
) -> tuple[bool, list, list[dict[str, Any]], Optional[float], Optional[dict[str, Any]]]:
    max_steps_override = cfg.EVALUATION.get("max_steps_override", None)
    max_steps = (
        _get_max_steps(cfg.EVALUATION.task_suite_name)
        if max_steps_override is None
        else int(max_steps_override)
    )
    replan_steps = _get_replan_steps(cfg)
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    capture_steps = set(_get_future_frame_capture_steps(cfg)[1:])
    jepa_frame_interval = int(cfg.data.train.action_video_freq_ratio)
    timing_enabled = bool(cfg.EVALUATION.get("timing_enabled", False))
    action_denoise_trace_enabled = bool(cfg.EVALUATION.get("action_denoise_trace", False))
    trigger_probe_enabled = bool(
        cfg.EVALUATION.get("trigger_probe", {}).get("enabled", False)
    )
    episode_start = (
        time.perf_counter()
        if timing_enabled or action_denoise_trace_enabled or trigger_probe_enabled
        else 0.0
    )
    replan_timings: list[dict[str, Any]] = []
    env_step_seconds = 0.0

    env.reset()
    camera_cfg = cfg.EVALUATION.get("vjepa2_ac", {}).get("camera", None)
    apply_libero_camera(env, camera_cfg)
    image_rotation_degrees = int(
        (camera_cfg or {}).get("image_rotation_degrees", 180)
    )
    obs = env.set_init_state(initial_state)
    observation_history = deque(maxlen=jepa_frame_interval + 1)
    observation_history.append(
        get_libero_image(obs, image_rotation_degrees=image_rotation_degrees)
    )
    if use_action_ensembler:
        from action_ensembler import ActionEnsembler

        ensembler = ActionEnsembler()
        ensembler.reset()

    replay_images = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    pending_actions: list[list[float]] = []
    current_predicted_future_clip: Optional[dict[str, Any]] = None
    current_replan_step = 0
    current_replan_idx = -1

    t = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            env_start = time.perf_counter() if timing_enabled else 0.0
            obs, _, done, _ = env.step(get_libero_dummy_action())
            if timing_enabled:
                env_step_seconds += time.perf_counter() - env_start
            observation_history.append(
                get_libero_image(obs, image_rotation_degrees=image_rotation_degrees)
            )
            t += 1
            continue

        if len(pending_actions) == 0:
            action_chunk, imgs, predicted_future_frames, replan_timing = _predict_action_chunk(
                obs=obs,
                previous_jepa_image=_agentview_for_jepa(
                    observation_history[0],
                    processor,
                ),
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
                ac_ranker=ac_ranker,
                ac_adapter=ac_adapter,
                executed_horizon=replan_steps,
                action_convergence_logger=action_convergence_logger,
            )
            if replan_timing is not None:
                if bool(cfg.EVALUATION.get("record_first_action_chunk", False)) and not replan_timings:
                    replan_timing["first_action_chunk"] = action_chunk.tolist()
                replan_timings.append(replan_timing)
                logging.info(
                    "replan timing episode=%d index=%d total=%.3fs video=%.3fs "
                    "joint=%.3fs action=%.3fs decode=%.3fs jepa_rank=%.3fs "
                    "encoder=%.3fs predictor=%.3fs",
                    episode_idx,
                    len(replan_timings) - 1,
                    replan_timing["total_s"],
                    replan_timing.get("infer_video_s", 0.0),
                    replan_timing.get("infer_joint_s", 0.0),
                    replan_timing.get("infer_action_s", 0.0),
                    replan_timing.get("decode_video_s", 0.0),
                    replan_timing.get("jepa_rank_s", 0.0),
                    replan_timing.get("jepa_encoder_phase_s", 0.0),
                    replan_timing.get("jepa_predictor_total_s", 0.0),
                )
            if predicted_future_frames is not None:
                current_replan_idx += 1
                current_predicted_future_clip = {
                    "replan_idx": current_replan_idx,
                    "gt_frames": [imgs.copy()],
                    "pred_frames": predicted_future_frames,
                }
            else:
                current_predicted_future_clip = None
            current_replan_step = 0
            if use_action_ensembler:
                ensembler.add_actions(action_chunk, t)
                pending_actions = [ensembler.get_action(ts).tolist() for ts in range(t, t + replan_steps)]
            else:
                pending_actions = action_chunk[:replan_steps].tolist()
            replay_images.append(imgs.copy())
        else:
            imgs = get_libero_image(obs, image_rotation_degrees=image_rotation_degrees)
            replay_images.append(imgs.copy())

        env_start = time.perf_counter() if timing_enabled else 0.0
        obs, _, done, _ = env.step(pending_actions.pop(0))
        if timing_enabled:
            env_step_seconds += time.perf_counter() - env_start
        observation_history.append(
            get_libero_image(obs, image_rotation_degrees=image_rotation_degrees)
        )
        if visualize_future_video and current_predicted_future_clip is not None:
            current_replan_step += 1
            if current_replan_step in capture_steps:
                current_predicted_future_clip["gt_frames"].append(
                    get_libero_image(obs, image_rotation_degrees=image_rotation_degrees)
                )
            if done or len(pending_actions) == 0:
                expected_frame_count = 1 + sum(
                    1 for capture_step in capture_steps if capture_step <= current_replan_step
                )
                gt_len = len(current_predicted_future_clip["gt_frames"])
                pred_len = len(current_predicted_future_clip["pred_frames"])
                assert gt_len == expected_frame_count, (
                    "GT future frames do not match expected capture count: "
                    f"gt_len={gt_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']} "
                    f"current_replan_step={current_replan_step} capture_steps={sorted(capture_steps)}."
                )
                assert pred_len >= expected_frame_count, (
                    "Predicted future frames shorter than expected capture count: "
                    f"pred_len={pred_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                if pred_len != expected_frame_count:
                    logging.info(
                        "Align predicted clip length to executed steps: "
                        "episode=%s replan=%s done=%s expected=%s pred_full=%s",
                        episode_idx,
                        current_predicted_future_clip["replan_idx"],
                        done,
                        expected_frame_count,
                        pred_len,
                    )
                current_predicted_future_clip["pred_frames"] = current_predicted_future_clip["pred_frames"][
                    :expected_frame_count
                ]
                assert len(current_predicted_future_clip["gt_frames"]) == len(
                    current_predicted_future_clip["pred_frames"]
                ), (
                    "GT/pred frame count mismatch after alignment: "
                    f"len(gt_frames)={len(current_predicted_future_clip['gt_frames'])} "
                    f"len(pred_frames)={len(current_predicted_future_clip['pred_frames'])} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                clip_psnr = _compute_clip_mean_psnr(
                    current_predicted_future_clip["gt_frames"],
                    current_predicted_future_clip["pred_frames"],
                )
                if clip_psnr is not None:
                    episode_future_clip_psnr.append(clip_psnr)
                predicted_future_video_clips.append(current_predicted_future_clip)
                current_predicted_future_clip = None
        if done:
            break
        t += 1
    pbar.close()

    episode_mean_psnr = (
        float(np.mean(episode_future_clip_psnr)) if len(episode_future_clip_psnr) > 0 else None
    )
    episode_timing: Optional[dict[str, Any]] = None
    if timing_enabled or action_denoise_trace_enabled or trigger_probe_enabled:
        total_replan_seconds = float(sum(item["total_s"] for item in replan_timings))
        episode_timing = {
            "episode_wall_s": time.perf_counter() - episode_start,
            "replan_count": len(replan_timings),
            "policy_replan_total_s": total_replan_seconds,
            "policy_replan_mean_s": (
                total_replan_seconds / len(replan_timings) if replan_timings else None
            ),
            "env_step_total_s": env_step_seconds,
            "low_level_action_steps": len(replay_images),
            "per_replan": replan_timings,
        }
    return bool(done), replay_images, predicted_future_video_clips, episode_mean_psnr, episode_timing


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    ac_ranker: Optional[VJEPA2ACRanker] = None,
    ac_adapter: Optional[LiberoACAdapter] = None,
    action_convergence_logger: Optional[ActionDenoiseConvergenceLogger] = None,
) -> dict:
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
    }
    memory_snapshots = getattr(model, "_libero_memory_snapshots", None)
    if memory_snapshots is not None:
        results["memory_snapshots"] = dict(memory_snapshots)
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None
    if bool(cfg.EVALUATION.get("timing_enabled", False)) or bool(
        cfg.EVALUATION.get("action_denoise_trace", False)
    ) or bool(cfg.EVALUATION.get("trigger_probe", {}).get("enabled", False)):
        results["episode_timings"] = []

    for trial_idx in range(int(cfg.EVALUATION.num_trials)):
        success, replay_images, predicted_future_video_clips, episode_mean_psnr, episode_timing = run_single_episode(
            env=env,
            initial_state=initial_states[trial_idx],
            task_description=task_description,
            model=model,
            processor=processor,
            cfg=cfg,
            episode_idx=trial_idx,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
            ac_ranker=ac_ranker,
            ac_adapter=ac_adapter,
            action_convergence_logger=action_convergence_logger,
        )
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)
        if visualize_future_video:
            results["episode_future_video_psnr"].append(episode_mean_psnr)
        if episode_timing is not None:
            results["episode_timings"].append(episode_timing)

        save_rollout_video(
            video_dir,
            replay_images,
            f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
            success=success,
            task_description=task_description,
        )
        if visualize_future_video:
            if len(predicted_future_video_clips) == 0:
                logging.warning(
                    "No predicted future frames collected for task %s trial %s.",
                    cfg.EVALUATION.task_id,
                    trial_idx,
                )
            else:
                all_gt_frames = []
                all_pred_frames = []
                for clip in predicted_future_video_clips:
                    all_gt_frames.extend(clip["gt_frames"])
                    all_pred_frames.extend(clip["pred_frames"])
                    save_prediction_video(
                        predicted_video_dir,
                        clip["gt_frames"],
                        clip["pred_frames"],
                        f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                        clip["replan_idx"],
                        success=success,
                        task_description=task_description,
                    )
                save_prediction_video(
                    predicted_video_dir,
                    all_gt_frames,
                    all_pred_frames,
                    f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                    "all",
                    success=success,
                    task_description=task_description,
                )

    close_fn = getattr(env, "close", None)
    if close_fn is not None:
        close_fn()

    if visualize_future_video:
        valid_episode_psnr = [x for x in results["episode_future_video_psnr"] if x is not None]
        if len(valid_episode_psnr) > 0:
            results["future_video_psnr_mean"] = float(np.mean(valid_episode_psnr))
    return results


def _required_worker_path(name: str) -> Path:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        raise ValueError(f"{name} must be set in worker mode.")
    return Path(os.path.expanduser(os.path.expandvars(value))).resolve()


def _result_file(output_root: Path, suite_name: str, task_id: int) -> Path | None:
    matches = sorted((output_root / suite_name).glob(f"gpu*_task{task_id}_results.json"))
    return matches[0] if matches else None


def _run_task_to_file(
    *,
    cfg: DictConfig,
    suite_name: str,
    task_id: int,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    output_root: Path,
    worker_id: str | None,
    ac_ranker: Optional[VJEPA2ACRanker] = None,
    ac_adapter: Optional[LiberoACAdapter] = None,
) -> tuple[Path, dict]:
    task_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    task_cfg.EVALUATION.task_suite_name = suite_name
    task_cfg.EVALUATION.task_id = int(task_id)

    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)
    # Let the active LIBERO benchmark resolve init-state storage. LIBERO-Plus
    # keeps layout-perturbation states under init_files/libero_newobj/<suite>,
    # while vanilla LIBERO stores them directly under init_files/<suite>.
    try:
        initial_states = task_suite.get_task_init_states(task_id)
    except pickle.UnpicklingError as exc:
        # Vanilla LIBERO predates PyTorch 2.6, where torch.load changed the
        # default to weights_only=True. These benchmark init-state files are
        # trusted local assets and contain NumPy arrays rather than weights.
        if "Weights only load failed" not in str(exc):
            raise
        init_states_path = Path(get_libero_path("init_states")) / (
            task.problem_folder
        ) / task.init_states_file
        logging.info(
            "Reloading trusted vanilla LIBERO init states with weights_only=False: %s",
            init_states_path,
        )
        initial_states = torch.load(init_states_path, weights_only=False)
    while len(initial_states) < int(task_cfg.EVALUATION.num_trials):
        initial_states.extend(
            initial_states[: int(task_cfg.EVALUATION.num_trials) - len(initial_states)]
        )

    video_dir = output_root / suite_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = output_root / suite_name / "predicted_videos"
    if bool(task_cfg.EVALUATION.get("visualize_future_video", False)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)

    convergence_logger = None
    if bool(task_cfg.EVALUATION.get("action_denoise_trace", False)):
        convergence_dir = (
            output_root
            / suite_name
            / f"task{task_id}_gpu{int(task_cfg.gpu_id)}"
            / "action_denoise_convergence"
        )
        convergence_logger = ActionDenoiseConvergenceLogger(convergence_dir)

    start_time = time.time()
    results = {
        "task_suite": suite_name,
        "task_id": task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": int(task_cfg.EVALUATION.num_trials),
        "gpu_id": int(task_cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }
    if worker_id is not None:
        results["worker_id"] = worker_id
    results.update(
        run_single_task(
            task=task,
            initial_states=initial_states,
            model=model,
            processor=processor,
            cfg=task_cfg,
            video_dir=video_dir,
            predicted_video_dir=predicted_video_dir,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
            ac_ranker=ac_ranker,
            ac_adapter=ac_adapter,
            action_convergence_logger=convergence_logger,
        )
    )
    if convergence_logger is not None:
        results["action_denoise_convergence"] = convergence_logger.save()
    results["duration"] = time.time() - start_time
    if "memory_snapshots" in results:
        results["memory_snapshots"]["after_rollout"] = _cuda_memory_snapshot(model_device)

    result_dir = output_root / suite_name
    result_dir.mkdir(parents=True, exist_ok=True)
    output_file = result_dir / f"gpu{task_cfg.gpu_id}_task{task_id}_results.json"
    temp_output_file = result_dir / f".{output_file.name}.{os.getpid()}.tmp"
    temp_output_file.write_text(
        json.dumps(results, indent=4, cls=NumpyEncoder),
        encoding="utf-8",
    )
    os.replace(temp_output_file, output_file)
    return output_file, results


def _run_worker_loop(
    *,
    cfg: DictConfig,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    output_root: Path,
    ac_ranker: Optional[VJEPA2ACRanker] = None,
    ac_adapter: Optional[LiberoACAdapter] = None,
) -> None:
    pending_file = _required_worker_path("LIBERO_WORKER_PENDING_FILE")
    lock_file = _required_worker_path("LIBERO_WORKER_LOCK_FILE")
    status_dir = _required_worker_path("LIBERO_WORKER_STATUS_DIR")
    failed_file = _required_worker_path("LIBERO_WORKER_FAILED_FILE")
    stop_file = _required_worker_path("LIBERO_WORKER_STOP_FILE")
    worker_id = os.environ.get("LIBERO_WORKER_ID", str(cfg.gpu_id))

    write_worker_status(status_dir, worker_id, "idle", "model loaded")
    completed = 0
    skipped = 0
    while not stop_file.exists():
        task = pop_task(pending_file, lock_file, status_dir, worker_id)
        if task is None:
            time.sleep(0.2)
            continue

        suite_name, task_id = task
        if _result_file(output_root, suite_name, task_id) is not None:
            skipped += 1
            continue

        try:
            output_file, _ = _run_task_to_file(
                cfg=cfg,
                suite_name=suite_name,
                task_id=task_id,
                model=model,
                processor=processor,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
                output_root=output_root,
                worker_id=worker_id,
                ac_ranker=ac_ranker,
                ac_adapter=ac_adapter,
            )
            completed += 1
            print(f"worker {worker_id} completed {suite_name},{task_id}: {output_file}")
        except Exception as exc:
            with failed_file.open("a", encoding="utf-8") as f:
                f.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')},{suite_name},{task_id},"
                    f"gpu={cfg.gpu_id},error={exc!r}\n"
                )
            write_worker_status(
                status_dir,
                worker_id,
                "failed",
                f"{suite_name},{task_id}: {exc!r}",
            )
            raise

    write_worker_status(
        status_dir,
        worker_id,
        "done",
        f"completed={completed} skipped={skipped}",
    )
    print(f"worker {worker_id} done: completed={completed} skipped={skipped}")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_single_process(cfg: DictConfig):
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. "
            "Use run_libero_manager.py for multi-GPU task parallelism."
        )

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    prompt_cache = None
    model_cfg = cfg.model
    if bool(cfg.EVALUATION.get("offload_text_encoder", False)):
        if not bool(cfg.model.get("load_text_encoder", True)):
            raise ValueError(
                "EVALUATION.offload_text_encoder requires model.load_text_encoder=true "
                "so the standalone T5 configuration is available."
            )
        prompt_cache = _preencode_prompt_contexts(
            cfg,
            device=model_device,
            dtype=model_dtype,
        )
        # Resolve interpolations while the model node is still attached to the
        # root config; a detached copy cannot resolve references such as
        # `${data.train.processor.proprio_output_dim}`.
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.load_text_encoder = False

    model = instantiate(model_cfg, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()
    if prompt_cache is not None:
        model._libero_prompt_context_cache = prompt_cache
    _log_cuda_memory("FastWAM loaded", model_device)
    model._libero_memory_snapshots = {
        "fastwam_only_after_load": _cuda_memory_snapshot(model_device),
    }

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    output_root = Path(
        os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.output_dir)))
    ).resolve()
    ac_ranker = None
    ac_adapter = None
    ac_cfg = cfg.EVALUATION.get("vjepa2_ac", {})
    trigger_probe_cfg = cfg.EVALUATION.get("trigger_probe", {})
    trigger_probe_enabled = bool(trigger_probe_cfg.get("enabled", False))
    if bool(ac_cfg.get("enabled", False)) or trigger_probe_enabled:
        if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
            raise ValueError(
                "V-JEPA2-AC ranking/trigger probing requires "
                "EVALUATION.visualize_future_video=true."
            )
        ac_adapter = LiberoACAdapter(
            low_level_steps_per_ac_step=int(
                ac_cfg.get("low_level_steps_per_ac_step", cfg.data.train.action_video_freq_ratio)
            ),
            max_gripper_width=float(ac_cfg.get("max_gripper_width", 0.08)),
            osc_position_scale=float(ac_cfg.get("osc_position_scale", 0.05)),
            osc_rotation_scale=float(ac_cfg.get("osc_rotation_scale", 0.5)),
        )
        ac_device_cfg = ac_cfg.get("device", None)
        ac_device = model_device if ac_device_cfg is None else str(ac_device_cfg)
        ac_dtype_name = str(ac_cfg.get("dtype", "float32")).strip().lower()
        ac_dtypes = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        if ac_dtype_name not in ac_dtypes:
            raise ValueError(
                "EVALUATION.vjepa2_ac.dtype must be one of: "
                "fp32, float32, fp16, float16, bf16, bfloat16."
            )
        max_ac_steps = max(
            int(ac_cfg.get("max_ac_steps", 8)),
            int(trigger_probe_cfg.get("ac_steps", 1)) if trigger_probe_enabled else 1,
        )
        ac_ranker = VJEPA2ACRanker(
            checkpoint_path=os.path.expanduser(os.path.expandvars(str(ac_cfg.checkpoint_path))),
            vjepa2_repo=os.path.expanduser(os.path.expandvars(str(ac_cfg.repo_path))),
            device=ac_device,
            dtype=ac_dtypes[ac_dtype_name],
            max_ac_steps=max_ac_steps,
        )
        if trigger_probe_enabled and not bool(ac_cfg.get("enabled", False)):
            logging.info(
                "Enabled V-JEPA2-AC trigger verifier on %s with %s (%d AC steps).",
                ac_device,
                ac_dtypes[ac_dtype_name],
                int(trigger_probe_cfg.get("ac_steps", 1)),
            )
        elif bool(ac_cfg.get("guidance", {}).get("enabled", False)):
            logging.info(
                "Enabled V-JEPA2-AC action-flow guidance on %s with %s.",
                ac_device,
                ac_dtypes[ac_dtype_name],
            )
        else:
            logging.info(
                "Enabled V-JEPA2-AC Best-of-%d ranking on %s with %s.",
                int(ac_cfg.get("num_action_candidates", 8)),
                ac_device,
                ac_dtypes[ac_dtype_name],
            )
        _log_cuda_memory("FastWAM + V-JEPA2-AC loaded", model_device)
        model._libero_memory_snapshots["fastwam_plus_jepa_after_load"] = _cuda_memory_snapshot(
            model_device
        )
    if os.environ.get("LIBERO_WORKER_MODE") == "1":
        _run_worker_loop(
            cfg=cfg,
            model=model,
            processor=processor,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
            output_root=output_root,
            ac_ranker=ac_ranker,
            ac_adapter=ac_adapter,
        )
        return None

    _, results = _run_task_to_file(
        cfg=cfg,
        suite_name=str(cfg.EVALUATION.task_suite_name),
        task_id=int(cfg.EVALUATION.task_id),
        model=model,
        processor=processor,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device=model_device,
        output_root=output_root,
        worker_id=None,
        ac_ranker=ac_ranker,
        ac_adapter=ac_adapter,
    )

    print(
        f"Task {cfg.EVALUATION.task_id} completed: "
        f"{results['successes']}/{cfg.EVALUATION.num_trials} successes"
    )
    if results.get("future_video_psnr_mean") is not None:
        print(f"Task {cfg.EVALUATION.task_id} future-video PSNR mean: {results['future_video_psnr_mean']:.4f}")
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


if __name__ == "__main__":
    eval_single_process()
