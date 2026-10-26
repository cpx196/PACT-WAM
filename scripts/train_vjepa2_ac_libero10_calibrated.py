"""LIBERO10 Predictor fine-tuning with train-only OSC response calibration.

The historical entrypoint and online adapter defaults are unchanged. Matching
calibration must be applied explicitly when this checkpoint is used online.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA_FORMAT_VERSION = 'libero_ac_euler_response_calibrated_v1'


def libero_ee_state_as_euler(ee):
    """LIBERO HDF5 ee_states = XYZ + rotvec; AC state = XYZ + Euler XYZ."""
    state = np.asarray(ee, dtype=np.float32).copy()
    if state.shape != (6,):
        raise ValueError(f'Expected LIBERO ee_states [6], got {state.shape}')
    if not np.isfinite(state).all():
        raise ValueError('Non-finite LIBERO EEF state')
    state[3:6] = Rotation.from_rotvec(state[3:6]).as_euler('xyz', degrees=False)
    return state


class LiberoWindows(Dataset):
    def __init__(self, root: Path, *, split: str, repeats: int, seed: int,
                 image_rotation_degrees: int = 0, calibration: dict,
                 manifest: Path):
        self.rows: list[tuple[str, str, int]] = []
        self.seed = seed
        self.repeats = repeats
        self.split = split
        self.calibration = calibration
        if image_rotation_degrees not in (0, 180):
            raise ValueError("Only 0° and 180° image rotations are supported")
        self.image_rotation_degrees = image_rotation_degrees
        files = sorted(root.glob("*.hdf5"))
        if not files:
            raise FileNotFoundError(f"No LIBERO10 HDF5 files in {root}")
        manifest_rows = json.loads(manifest.read_text())
        split_pairs = {(r['task_file'], r['demo']): r['split'] for r in manifest_rows}
        assert len(split_pairs) == 500, 'Expected 500 LIBERO10 demonstrations'
        for path in files:
            with h5py.File(path, "r") as handle:
                demos = sorted(handle["data"], key=lambda name: int(name.split("_")[-1]))
                for i, demo in enumerate(demos):
                    if split_pairs[(path.name, demo)] != split:
                        continue
                    length = len(handle[f"data/{demo}/actions"])
                    if length >= 17:
                        self.rows.append((str(path), demo, length))
        if not self.rows:
            raise RuntimeError(f"No {split} windows found")
        self.files = len(files)
        assert self.files == 10 and len(self.rows) == (450 if split == 'train' else 50)

    def __len__(self) -> int:
        return len(self.rows) * self.repeats

    def __getitem__(self, index: int):
        path, demo, length = self.rows[index // self.repeats]
        # Fixed validation windows: do not confuse resampling noise with progress.
        rng = np.random.default_rng(self.seed + index + (random.randrange(2**20) if self.split == 'train' else 0))
        t = int(rng.integers(8, length - 8))
        # Online context is [previous observation, current observation].
        # Online target clips overlap at the predicted frame boundary.
        frame_ids = np.asarray([t - 8, t, t, t + 4, t + 4, t + 8])
        with h5py.File(path, "r") as handle:
            group = handle[f"data/{demo}"]
            rgb = group["obs/agentview_rgb"]
            frames = np.stack([rgb[int(i)] for i in frame_ids])
            low_actions = np.asarray(group["actions"][t : t + 8], dtype=np.float32)
            ee = np.asarray(group["obs/ee_states"][t], dtype=np.float32)
            fingers = np.asarray(group["obs/gripper_states"][t], dtype=np.float32)

        # FastWAM's online agentview frames and predicted future are both
        # rotated 180°. Apply the same orientation to context and target clips.
        if self.image_rotation_degrees == 180:
            frames = np.rot90(frames, k=2, axes=(1, 2))

        # Match LiberoACAdapter's initial EEF pose and gripper convention.
        closedness = 1.0 - np.clip(abs(float(fingers[0] - fingers[1])) / 0.08, 0.0, 1.0)
        state = np.concatenate((libero_ee_state_as_euler(ee), [closedness])).astype(np.float32)
        from experiments.libero.vjepa2_ac_ranker import LiberoACAdapter

        # Scale clipped XYZ and per-step rotation vectors BEFORE composition.
        # The shared adapter defaults and robot executable commands are untouched.
        converted = LiberoACAdapter(
            low_level_steps_per_ac_step=4,
            osc_position_scale=self.calibration['osc_position_scale'] * self.calibration['translation_response_gain'],
            osc_rotation_scale=self.calibration['osc_rotation_scale'] * self.calibration['rotation_response_gain'],
        ).convert(
            low_actions[None], state
        )
        frames = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
        return frames, torch.from_numpy(converted.actions[0]), torch.from_numpy(converted.states[0])


def make_models(repo: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(repo))
    from src.hub.backbones import vjepa2_ac_vit_giant

    encoder, predictor = vjepa2_ac_vit_giant(pretrained=False, num_frames=6)
    source = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)

    def clean(weights):
        return {key.replace("module.", "").replace("backbone.", ""): value for key, value in weights.items()}

    enc_result = encoder.load_state_dict(clean(source["encoder"]), strict=False)
    if enc_result.missing_keys or enc_result.unexpected_keys:
        raise RuntimeError(f"Encoder checkpoint mismatch: {enc_result}")
    predictor.load_state_dict(clean(source["predictor"]), strict=True)
    del source
    encoder.eval().requires_grad_(False).to(device)
    # LIBERO online guidance does not supply camera extrinsics.  Keeping this
    # unused branch trainable makes DDP expect gradients it cannot receive.
    predictor.extrinsics_encoder.requires_grad_(False)
    predictor.use_activation_checkpointing = True
    predictor.train().to(device)
    return encoder, predictor


def encode(encoder, frames: torch.Tensor, device: torch.device):
    batch = frames.shape[0]
    x = frames.reshape(batch * 6, 3, *frames.shape[-2:]).to(device, non_blocking=True)
    x = F.interpolate(x.float() / 255.0, size=(256, 256), mode="bicubic", align_corners=False, antialias=True)
    mean = x.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = x.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    x = ((x - mean) / std).reshape(batch, 3, 2, 3, 256, 256)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch * 3, 3, 2, 256, 256)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        reps = encoder(x)
        if isinstance(reps, list):
            reps = reps[-1]
        reps = F.layer_norm(reps.float(), (reps.shape[-1],))
    return reps.reshape(batch, 3, reps.shape[1], reps.shape[2]).detach()


def loss_fn(predictor, reps, actions, states):
    batch, _, tokens, width = reps.shape
    teacher = predictor(
        reps[:, :2].reshape(batch, 2 * tokens, width), actions, states
    )
    teacher = F.layer_norm(teacher.float(), (width,)).reshape(batch, 2, tokens, width)
    target = reps[:, 1:].float()
    teacher_loss = F.l1_loss(teacher, target)

    first = predictor(reps[:, 0], actions[:, :1], states[:, :1])[:, -tokens:]
    first = F.layer_norm(first.float(), (width,))
    second = predictor(
        torch.cat((reps[:, 0], first), dim=1), actions, states
    )[:, -tokens:]
    second = F.layer_norm(second.float(), (width,))
    rollout_loss = (F.l1_loss(first, target[:, 0]) + F.l1_loss(second, target[:, 1])) / 2
    return teacher_loss, rollout_loss


def save_checkpoint(path: Path, *, predictor, optimizer, scheduler, step, args):
    payload = {
        "predictor": predictor.module.state_dict() if hasattr(predictor, "module") else predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "config": vars(args),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def validate(encoder, predictor, loader, device, batches=12):
    model = predictor.module if hasattr(predictor, "module") else predictor
    model.eval()
    totals = torch.zeros(3, device=device)
    iterator = iter(loader)
    for _ in range(batches):
        try:
            frames, actions, states = next(iterator)
        except StopIteration:
            break
        reps = encode(encoder, frames, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tf, ar = loss_fn(model, reps, actions.to(device), states.to(device))
        totals += torch.stack((tf.float(), ar.float(), tf.new_ones(()).float()))
    model.train()
    return (totals[:2] / totals[2].clamp_min(1)).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--image-rotation-degrees", type=int, choices=(0, 180), default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.data_format_version = DATA_FORMAT_VERSION
    args.calibration_config = json.loads(args.calibration.read_text())
    if args.calibration_config.get('schema') != 'libero_ac_command_response_v1' or args.calibration_config.get('fitted_split') != 'train_only':
        raise ValueError('Require audited train-only response calibration')
    if Path(args.calibration_config['manifest']).resolve() != args.manifest.resolve():
        raise ValueError('Calibration and training split manifests must match')
    for key in ('translation_response_gain', 'rotation_response_gain', 'osc_position_scale', 'osc_rotation_scale'):
        if not np.isfinite(args.calibration_config[key]) or args.calibration_config[key] <= 0:
            raise ValueError(f'Invalid calibration value: {key}')

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = LiberoWindows(args.data, split="train", repeats=8, seed=args.seed,
                            image_rotation_degrees=args.image_rotation_degrees,
                            calibration=args.calibration_config, manifest=args.manifest)
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed) if distributed else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        shuffle=sampler is None, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = None
    if rank == 0:
        val_data = LiberoWindows(args.data, split="val", repeats=1, seed=args.seed + 1000000,
                                 image_rotation_degrees=args.image_rotation_degrees,
                                 calibration=args.calibration_config, manifest=args.manifest)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
    encoder, predictor = make_models(args.repo, args.checkpoint, device)
    predictor = DistributedDataParallel(predictor, device_ids=[local_rank]) if distributed else predictor
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=0.04)

    def ratio(step):
        if step < args.warmup:
            return max(0.05, (step + 1) / max(1, args.warmup))
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, ratio)
    start_step = 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get('config', {}).get('data_format_version') != DATA_FORMAT_VERSION:
            raise ValueError('Refuse to resume a checkpoint with legacy/unknown EEF conventions')
        if saved['config'].get('calibration_config') != args.calibration_config:
            raise ValueError('Refuse to resume with a different response calibration')
        (predictor.module if distributed else predictor).load_state_dict(saved["predictor"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_step = int(saved["step"])
        del saved
    if rank == 0:
        (args.output / 'config.json').write_text(json.dumps(vars(args), default=str, indent=2))
        print(json.dumps({"event": "start", "rank": rank, "files": dataset.files,
                          "trajectories": len(dataset.rows), "steps": args.steps,
                          "resume_step": start_step, "world_size": dist.get_world_size() if distributed else 1,
                          "image_rotation_degrees": args.image_rotation_degrees,
                          "calibration": args.calibration_config,
                          "eef_state_convention": "LIBERO rotvec converted to Euler XYZ"}), flush=True)

    epoch = 0
    iterator = iter(loader)
    for step in range(start_step + 1, args.steps + 1):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for micro in range(args.accum):
            try:
                frames, actions, states = next(iterator)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                iterator = iter(loader)
                frames, actions, states = next(iterator)
            actions = actions.to(device, non_blocking=True)
            states = states.to(device, non_blocking=True)
            reps = encode(encoder, frames, device)
            sync = micro == args.accum - 1
            context = contextlib.nullcontext() if sync or not distributed else predictor.no_sync()
            with context, torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_loss, rollout_loss = loss_fn(predictor, reps, actions, states)
                loss = (teacher_loss + rollout_loss) / args.accum
            loss.backward()
            losses.append((teacher_loss.detach().float(), rollout_loss.detach().float()))
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step <= 5 or step % 20 == 0 or step % args.save_every == 0:
            torch.cuda.synchronize(device)
            metrics = torch.stack([torch.stack(pair) for pair in losses]).mean(0)
            if distributed:
                dist.all_reduce(metrics)
                metrics /= dist.get_world_size()
            if rank == 0:
                print(json.dumps({"step": step, "tf_loss": float(metrics[0]),
                                  "ar_loss": float(metrics[1]), "lr": scheduler.get_last_lr()[0],
                                  "seconds": round(time.monotonic() - started, 3),
                                  "max_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2)}), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            if rank == 0:
                val_tf, val_ar = validate(encoder, predictor, val_loader, device)
                print(json.dumps({"step": step, "val_tf_loss": val_tf,
                                  "val_ar_loss": val_ar}), flush=True)
                save_checkpoint(args.output / f"step_{step:06d}.pt", predictor=predictor,
                                optimizer=optimizer, scheduler=scheduler, step=step, args=args)
            if distributed:
                dist.barrier()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
