"""V-JEPA 2-AC adapter and Best-of-K latent action ranker for LIBERO.

FastWAM/LIBERO actions use delta translation, delta axis-angle rotation, and
an absolute gripper command.  V-JEPA 2-AC was trained with absolute 7-DoF
robot states and the state-to-state delta between adjacent video frames.  This
module performs that deterministic conversion and exposes the official L1
latent energy used by Meta's planner.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ACTrajectoryBatch:
    actions: np.ndarray  # [K, S, 7], delta xyz/euler/gripper-closedness
    states: np.ndarray  # [K, S, 7], absolute xyz/euler/closedness
    terminal_states: np.ndarray  # [K, 7]


@dataclass(frozen=True)
class ACTrajectoryTensorBatch:
    """Differentiable counterpart of :class:`ACTrajectoryBatch`."""

    actions: torch.Tensor  # [K, S, 7], delta xyz/euler/gripper-closedness
    states: torch.Tensor  # [K, S, 7], absolute xyz/euler/closedness
    terminal_states: torch.Tensor  # [K, 7]


class LiberoACAdapter:
    """Convert executable LIBERO action chunks to V-JEPA 2-AC semantics."""

    def __init__(
        self,
        *,
        low_level_steps_per_ac_step: int = 4,
        max_gripper_width: float = 0.08,
        osc_position_scale: float = 0.05,
        osc_rotation_scale: float = 0.5,
    ):
        if low_level_steps_per_ac_step <= 0:
            raise ValueError("low_level_steps_per_ac_step must be positive")
        if max_gripper_width <= 0:
            raise ValueError("max_gripper_width must be positive")
        if osc_position_scale <= 0:
            raise ValueError("osc_position_scale must be positive")
        if osc_rotation_scale <= 0:
            raise ValueError("osc_rotation_scale must be positive")
        self.low_level_steps_per_ac_step = int(low_level_steps_per_ac_step)
        self.max_gripper_width = float(max_gripper_width)
        self.osc_position_scale = float(osc_position_scale)
        self.osc_rotation_scale = float(osc_rotation_scale)

    def state_from_observation(self, obs: dict[str, Any]) -> np.ndarray:
        position = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(3)
        quaternion_xyzw = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).reshape(4)
        euler_xyz = Rotation.from_quat(quaternion_xyzw).as_euler("xyz", degrees=False)
        gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)
        if gripper_qpos.size < 2:
            raise ValueError("LIBERO observation must contain both gripper joint positions")
        gripper_width = abs(float(gripper_qpos[0] - gripper_qpos[1]))
        closedness = 1.0 - np.clip(gripper_width / self.max_gripper_width, 0.0, 1.0)
        return np.concatenate((position, euler_xyz, [closedness])).astype(np.float32)

    def convert(self, action_chunks: np.ndarray, initial_state: np.ndarray) -> ACTrajectoryBatch:
        chunks = np.asarray(action_chunks, dtype=np.float64)
        if chunks.ndim == 2:
            chunks = chunks[None]
        if chunks.ndim != 3 or chunks.shape[-1] != 7:
            raise ValueError(f"Expected action chunks [K,T,7], got {chunks.shape}")
        state0 = np.asarray(initial_state, dtype=np.float64).reshape(-1)
        if state0.shape != (7,):
            raise ValueError(f"Expected initial V-JEPA state [7], got {state0.shape}")

        candidate_actions: list[np.ndarray] = []
        candidate_states: list[np.ndarray] = []
        terminal_states: list[np.ndarray] = []
        stride = self.low_level_steps_per_ac_step

        for chunk in chunks:
            state = state0.copy()
            ac_actions: list[np.ndarray] = []
            ac_states: list[np.ndarray] = []
            for start in range(0, len(chunk), stride):
                group = chunk[start : start + stride]
                ac_states.append(state.copy())

                # FastWAM emits the normalized OSC_POSE command that is sent
                # to env.step().  Match Robosuite's symmetric input clipping
                # and output scaling before converting to DROID/V-JEPA's
                # metric EEF pose deltas.
                delta_xyz = (
                    np.clip(group[:, :3], -1.0, 1.0) * self.osc_position_scale
                ).sum(axis=0)
                delta_rotation = Rotation.identity()
                scaled_rotvecs = (
                    np.clip(group[:, 3:6], -1.0, 1.0) * self.osc_rotation_scale
                )
                for rotvec in scaled_rotvecs:
                    delta_rotation = Rotation.from_rotvec(rotvec) * delta_rotation
                delta_euler = delta_rotation.as_euler("xyz", degrees=False)

                # The executable LIBERO convention is -1=open, +1=close.
                target_closedness = np.clip((float(group[-1, 6]) + 1.0) / 2.0, 0.0, 1.0)
                delta_closedness = target_closedness - state[6]
                ac_action = np.concatenate((delta_xyz, delta_euler, [delta_closedness]))
                ac_actions.append(ac_action)

                state[:3] += delta_xyz
                absolute_rotation = delta_rotation * Rotation.from_euler(
                    "xyz", state[3:6], degrees=False
                )
                state[3:6] = absolute_rotation.as_euler("xyz", degrees=False)
                state[6] = target_closedness

            candidate_actions.append(np.stack(ac_actions))
            candidate_states.append(np.stack(ac_states))
            terminal_states.append(state)

        return ACTrajectoryBatch(
            actions=np.stack(candidate_actions).astype(np.float32),
            states=np.stack(candidate_states).astype(np.float32),
            terminal_states=np.stack(terminal_states).astype(np.float32),
        )

    @staticmethod
    def _skew_symmetric(vectors: torch.Tensor) -> torch.Tensor:
        x, y, z = vectors.unbind(dim=-1)
        zero = torch.zeros_like(x)
        return torch.stack(
            (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
        ).reshape(*vectors.shape[:-1], 3, 3)

    @classmethod
    def _rotvec_to_matrix(cls, rotvec: torch.Tensor) -> torch.Tensor:
        """Differentiable Rodrigues conversion, including the zero-angle limit."""
        theta_sq = torch.sum(rotvec * rotvec, dim=-1, keepdim=True)
        theta = torch.sqrt(theta_sq.clamp_min(torch.finfo(rotvec.dtype).tiny))
        small = theta_sq < 1e-8
        theta_sq_safe = theta_sq.clamp_min(torch.finfo(rotvec.dtype).eps)
        a = torch.where(
            small,
            1.0 - theta_sq / 6.0 + theta_sq * theta_sq / 120.0,
            torch.sin(theta) / theta,
        )
        b = torch.where(
            small,
            0.5 - theta_sq / 24.0 + theta_sq * theta_sq / 720.0,
            (1.0 - torch.cos(theta)) / theta_sq_safe,
        )
        skew = cls._skew_symmetric(rotvec)
        identity = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)
        identity = identity.expand(*rotvec.shape[:-1], 3, 3)
        return identity + a.unsqueeze(-1) * skew + b.unsqueeze(-1) * (skew @ skew)

    @staticmethod
    def _euler_xyz_to_matrix(euler: torch.Tensor) -> torch.Tensor:
        """Match scipy's lower-case (extrinsic) ``from_euler('xyz')``."""
        x, y, z = euler.unbind(dim=-1)
        cx, cy, cz = torch.cos(x), torch.cos(y), torch.cos(z)
        sx, sy, sz = torch.sin(x), torch.sin(y), torch.sin(z)
        return torch.stack(
            (
                cy * cz,
                sx * sy * cz - cx * sz,
                cx * sy * cz + sx * sz,
                cy * sz,
                sx * sy * sz + cx * cz,
                cx * sy * sz - sx * cz,
                -sy,
                sx * cy,
                cx * cy,
            ),
            dim=-1,
        ).reshape(*euler.shape[:-1], 3, 3)

    @staticmethod
    def _matrix_to_euler_xyz(matrix: torch.Tensor) -> torch.Tensor:
        """Differentiable XYZ extraction away from the usual gimbal singularity."""
        # Clamp only for numerical roundoff. At the singularity the Euler
        # representation itself is non-unique, just as in scipy.
        pitch = torch.asin((-matrix[..., 2, 0]).clamp(-1.0, 1.0))
        roll = torch.atan2(matrix[..., 2, 1], matrix[..., 2, 2])
        yaw = torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0])
        return torch.stack((roll, pitch, yaw), dim=-1)

    def convert_torch(
        self,
        action_chunks: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> ACTrajectoryTensorBatch:
        """Convert executable LIBERO chunks without breaking autograd.

        The existing NumPy/SciPy ``convert`` method remains the inference-only
        ranking path. This method mirrors its semantics for action guidance.
        """
        chunks = action_chunks
        if chunks.ndim == 2:
            chunks = chunks.unsqueeze(0)
        if chunks.ndim != 3 or chunks.shape[-1] != 7:
            raise ValueError(f"Expected action chunks [K,T,7], got {tuple(chunks.shape)}")
        if not chunks.is_floating_point():
            raise TypeError("action_chunks must be a floating-point tensor")

        state0 = torch.as_tensor(
            initial_state, device=chunks.device, dtype=chunks.dtype
        ).reshape(-1)
        if tuple(state0.shape) != (7,):
            raise ValueError(f"Expected initial V-JEPA state [7], got {tuple(state0.shape)}")

        batch_size = chunks.shape[0]
        state = state0.unsqueeze(0).expand(batch_size, -1).clone()
        candidate_actions = []
        candidate_states = []
        stride = self.low_level_steps_per_ac_step
        identity = torch.eye(3, device=chunks.device, dtype=chunks.dtype)
        identity = identity.unsqueeze(0).expand(batch_size, -1, -1)

        for start in range(0, chunks.shape[1], stride):
            group = chunks[:, start : start + stride]
            candidate_states.append(state)

            delta_xyz = (
                torch.clamp(group[..., :3], -1.0, 1.0) * self.osc_position_scale
            ).sum(dim=1)
            delta_rotation = identity
            scaled_rotvecs = (
                torch.clamp(group[..., 3:6], -1.0, 1.0) * self.osc_rotation_scale
            )
            for low_level_step in range(scaled_rotvecs.shape[1]):
                step_rotation = self._rotvec_to_matrix(
                    scaled_rotvecs[:, low_level_step]
                )
                delta_rotation = step_rotation @ delta_rotation
            delta_euler = self._matrix_to_euler_xyz(delta_rotation)

            target_closedness = torch.clamp((group[:, -1, 6] + 1.0) / 2.0, 0.0, 1.0)
            delta_closedness = target_closedness - state[:, 6]
            candidate_actions.append(
                torch.cat((delta_xyz, delta_euler, delta_closedness.unsqueeze(-1)), dim=-1)
            )

            absolute_rotation = delta_rotation @ self._euler_xyz_to_matrix(state[:, 3:6])
            state = torch.cat(
                (
                    state[:, :3] + delta_xyz,
                    self._matrix_to_euler_xyz(absolute_rotation),
                    target_closedness.unsqueeze(-1),
                ),
                dim=-1,
            )

        return ACTrajectoryTensorBatch(
            actions=torch.stack(candidate_actions, dim=1),
            states=torch.stack(candidate_states, dim=1),
            terminal_states=state,
        )


class VJEPA2ACRanker:
    """Load the official V-JEPA 2-AC model and rank candidate trajectories."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        vjepa2_repo: str,
        device: str = "cuda:1",
        dtype: torch.dtype = torch.float32,
        max_ac_steps: int = 8,
        compile_predictor: bool = False,
    ) -> None:
        repo = str(Path(vjepa2_repo).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from src.hub.backbones import vjepa2_ac_vit_giant

        # One latent frame is represented by a two-frame tubelet.  The causal
        # mask must cover the current frame plus the complete rollout.
        self.encoder, self.predictor = vjepa2_ac_vit_giant(
            pretrained=False,
            num_frames=2 * (int(max_ac_steps) + 1),
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        encoder_sd = self._clean_state_dict(checkpoint["encoder"])
        predictor_sd = self._clean_state_dict(checkpoint["predictor"])
        encoder_load = self.encoder.load_state_dict(encoder_sd, strict=False)
        if encoder_load.missing_keys:
            raise ValueError(f"Missing V-JEPA encoder keys: {encoder_load.missing_keys}")
        self.predictor.load_state_dict(predictor_sd, strict=True)

        self.device = torch.device(device)
        self.dtype = dtype
        self.max_ac_steps = int(max_ac_steps)
        self.encoder.to(device=self.device, dtype=dtype).eval().requires_grad_(False)
        self.predictor.to(device=self.device, dtype=dtype).eval().requires_grad_(False)
        if compile_predictor:
            self.predictor = torch.compile(
                self.predictor,
                mode="reduce-overhead",
                fullgraph=True,
            )
        self.tokens_per_frame = (256 // int(self.encoder.patch_size)) ** 2
        self.last_timing: dict[str, float] = {}

    @staticmethod
    def _clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in state_dict.items()
        }

    @staticmethod
    def _as_agentview(image: Any) -> np.ndarray:
        if isinstance(image, dict):
            image = image["image"]
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"Expected RGB image [H,W,3], got {array.shape}")
        # FastWAM future frames concatenate agent and wrist views horizontally.
        if array.shape[1] == 2 * array.shape[0]:
            array = array[:, : array.shape[0]]
        # PIL-backed arrays can be read-only; make an owned writable copy
        # before torch.from_numpy to avoid undefined-behavior warnings.
        return np.array(array, copy=True, order="C")

    def encode_clips(self, clips: Sequence[Sequence[Any]]) -> torch.Tensor:
        """Encode genuine two-frame clips into one latent visual state each."""
        clip_tensors = []
        for clip in clips:
            if len(clip) != 2:
                raise ValueError(f"Each V-JEPA2-AC clip must contain exactly 2 frames, got {len(clip)}")
            frames = []
            for image in clip:
                array = self._as_agentview(image)
                tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
                tensor = F.interpolate(
                    tensor.unsqueeze(0),
                    size=(256, 256),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )[0]
                frames.append(tensor)
            clip_tensors.append(torch.stack(frames, dim=1))  # [C, T=2, H, W]
        batch = torch.stack(clip_tensors).to(device=self.device)
        mean = torch.tensor((0.485, 0.456, 0.406), device=self.device).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=self.device).view(1, 3, 1, 1)
        batch = ((batch - mean.unsqueeze(2)) / std.unsqueeze(2)).to(dtype=self.dtype)
        reps = self.encoder(batch)
        if isinstance(reps, list):
            reps = reps[-1]
        return F.layer_norm(reps, (reps.shape[-1],))

    def encode_guidance_context(
        self,
        *,
        current_clip: Sequence[Any],
        target_future_clips: Sequence[Sequence[Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode fixed visual context once; returned tensors are detached."""
        # Use no_grad rather than inference_mode: these fixed tensors are fed
        # into a later autograd-enabled predictor call for action gradients.
        with torch.no_grad():
            encoded = self.encode_clips([current_clip, *target_future_clips])
        return encoded[0].detach(), encoded[1:].detach()

    def differentiable_energy(
        self,
        *,
        current_rep: torch.Tensor,
        target_reps: torch.Tensor,
        ac_actions: torch.Tensor,
        ac_states: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-candidate energy while retaining gradients to actions/states."""
        actions = ac_actions.to(device=self.device, dtype=self.dtype)
        states = ac_states.to(device=self.device, dtype=self.dtype)
        if actions.shape != states.shape or actions.ndim != 3 or actions.shape[-1] != 7:
            raise ValueError(
                f"AC actions/states must both be [K,S,7], got {actions.shape} and {states.shape}"
            )
        num_candidates, num_steps, _ = actions.shape
        if num_steps <= 0 or num_steps > self.max_ac_steps:
            raise ValueError(
                f"V-JEPA2-AC rollout steps must be in [1, {self.max_ac_steps}], got {num_steps}."
            )
        if target_reps.shape[0] != num_steps:
            raise ValueError(
                f"Expected {num_steps} target representations, got {target_reps.shape[0]}."
            )

        current = current_rep.to(device=self.device, dtype=self.dtype)
        targets = target_reps.to(device=self.device, dtype=self.dtype)
        rep_history = current.unsqueeze(0).expand(num_candidates, -1, -1)
        per_step_energies = []
        for step in range(num_steps):
            prediction = self.predictor(
                rep_history,
                actions[:, : step + 1],
                states[:, : step + 1],
            )
            predicted_rep = F.layer_norm(
                prediction[:, -self.tokens_per_frame :],
                (prediction.shape[-1],),
            )
            rep_history = torch.cat((rep_history, predicted_rep), dim=1)
            per_step_energies.append(
                torch.mean(
                    torch.abs(predicted_rep.float() - targets[step].unsqueeze(0).float()),
                    dim=(1, 2),
                )
            )
        return torch.stack(per_step_energies, dim=1).mean(dim=1)

    @torch.inference_mode()
    def rank(
        self,
        *,
        current_clip: Sequence[Any],
        target_future_clips: Sequence[Sequence[Any]],
        ac_actions: np.ndarray,
        ac_states: np.ndarray,
        timing_enabled: bool = False,
    ) -> tuple[int, torch.Tensor]:
        actions = torch.as_tensor(ac_actions, device=self.device, dtype=self.dtype)
        states = torch.as_tensor(ac_states, device=self.device, dtype=self.dtype)
        if actions.shape != states.shape or actions.ndim != 3 or actions.shape[-1] != 7:
            raise ValueError(
                f"AC actions/states must both be [K,S,7], got {actions.shape} and {states.shape}"
            )

        num_candidates, num_steps, _ = actions.shape
        if num_steps <= 0 or num_steps > self.max_ac_steps:
            raise ValueError(
                f"V-JEPA2-AC rollout steps must be in [1, {self.max_ac_steps}], got {num_steps}."
            )
        if len(target_future_clips) != num_steps:
            raise ValueError(
                "V-JEPA2-AC needs one two-frame target clip per action step, got "
                f"{len(target_future_clips)} clips for {num_steps} steps."
            )
        if timing_enabled:
            torch.cuda.synchronize(self.device)
        encoder_start = time.perf_counter() if timing_enabled else 0.0
        encoded = self.encode_clips([current_clip, *target_future_clips])
        if timing_enabled:
            torch.cuda.synchronize(self.device)
            encoder_phase_seconds = time.perf_counter() - encoder_start
        current_rep = encoded[0]
        target_reps = encoded[1:]
        rep_history = current_rep.unsqueeze(0).expand(num_candidates, -1, -1)
        predicted_rep = rep_history
        per_step_energies = []
        predictor_step_seconds: list[float] = []
        for step in range(num_steps):
            if timing_enabled:
                torch.cuda.synchronize(self.device)
            predictor_start = time.perf_counter() if timing_enabled else 0.0
            prediction = self.predictor(
                rep_history,
                actions[:, : step + 1],
                states[:, : step + 1],
            )
            predicted_rep = F.layer_norm(
                prediction[:, -self.tokens_per_frame :],
                (prediction.shape[-1],),
            )
            rep_history = torch.cat((rep_history, predicted_rep), dim=1)
            per_step_energies.append(
                torch.mean(
                    torch.abs(predicted_rep.float() - target_reps[step].unsqueeze(0).float()),
                    dim=(1, 2),
                )
            )
            if timing_enabled:
                torch.cuda.synchronize(self.device)
                predictor_step_seconds.append(time.perf_counter() - predictor_start)

        # Apply the official normalized-latent L1 objective at every future
        # step, then average across the complete FastWAM trajectory.
        energies = torch.stack(per_step_energies, dim=1).mean(dim=1)
        selected = int(torch.argmin(energies).item())
        if timing_enabled:
            self.last_timing = {
                "jepa_encoder_phase_s": encoder_phase_seconds,
                "jepa_predictor_total_s": float(sum(predictor_step_seconds)),
                "jepa_predictor_step_mean_s": float(np.mean(predictor_step_seconds)),
                "jepa_predictor_step_0_s": predictor_step_seconds[0],
                "jepa_predictor_step_1_s": predictor_step_seconds[1]
                if len(predictor_step_seconds) > 1
                else 0.0,
            }
        else:
            self.last_timing = {}
        return selected, energies.detach().cpu()
