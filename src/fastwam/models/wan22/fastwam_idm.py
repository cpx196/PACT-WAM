from typing import Any, Callable, Optional, Sequence

import torch
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from .fastwam_joint import FastWAMJoint

logger = get_logger(__name__)


class FastWAMIDM(FastWAMJoint):
    """IDM variant with teacher-forcing video conditioning for action denoising."""

    video_cond_noise_prob: float

    @classmethod
    def from_wan22_pretrained(cls, *, video_cond_noise_prob: float = 0.5, **kwargs):
        prob = float(video_cond_noise_prob)
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"`video_cond_noise_prob` must be in [0, 1], got {prob}.")
        model = super().from_wan22_pretrained(**kwargs)
        model.video_cond_noise_prob = prob
        return model

    @torch.no_grad()
    def _build_teacher_forcing_attention_mask(
        self,
        noisy_video_seq_len: int,
        cond_video_seq_len: int,
        action_seq_len: int,
        noisy_video_tokens_per_frame: int,
        cond_video_tokens_per_frame: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        del batch_size
        if noisy_video_tokens_per_frame != cond_video_tokens_per_frame:
            raise ValueError(
                "Teacher-forcing requires identical `tokens_per_frame` for noisy and cond video branches, "
                f"got {noisy_video_tokens_per_frame} and {cond_video_tokens_per_frame}."
            )

        noisy_end = noisy_video_seq_len
        cond_end = noisy_video_seq_len + cond_video_seq_len
        total_seq_len = cond_end + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # noisy_video -> noisy_video
        mask[:noisy_end, :noisy_end] = self.video_expert.build_video_to_video_mask(
            video_seq_len=noisy_video_seq_len,
            video_tokens_per_frame=noisy_video_tokens_per_frame,
            device=device,
        )
        # cond_video -> cond_video
        mask[noisy_end:cond_end, noisy_end:cond_end] = self.video_expert.build_video_to_video_mask(
            video_seq_len=cond_video_seq_len,
            video_tokens_per_frame=cond_video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[cond_end:, cond_end:] = True
        # action -> cond_video only
        mask[cond_end:, noisy_end:cond_end] = True
        return mask

    def _teacher_forcing_training_denoise_core(
        self,
        latents_noisy: torch.Tensor,
        latents_cond: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_video_cond: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (
            noisy_video_tokens,
            t_video_noisy,
            t_mod_video_noisy,
            context_video_noisy,
            context_mask_video_noisy,
            freqs_video_noisy,
            f_noisy,
            h_noisy,
            w_noisy,
            noisy_video_tokens_per_frame,
        ) = self.video_expert.prepare(
            x=latents_noisy,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        (
            cond_video_tokens,
            _t_video_cond,
            t_mod_video_cond,
            _context_video_cond,
            context_mask_video_cond,
            freqs_video_cond,
            _f_cond,
            _h_cond,
            _w_cond,
            cond_video_tokens_per_frame,
        ) = self.video_expert.prepare(
            x=latents_cond,
            timestep=timestep_video_cond,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        (
            action_tokens,
            _t_action,
            t_mod_action,
            context_action,
            context_mask_action,
            freqs_action,
        ) = self.action_expert.prepare(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        noisy_video_seq_len = int(noisy_video_tokens.shape[1])
        cond_video_seq_len = int(cond_video_tokens.shape[1])
        merged_video_tokens = torch.cat([noisy_video_tokens, cond_video_tokens], dim=1)
        merged_video_freqs = torch.cat([freqs_video_noisy, freqs_video_cond], dim=0)
        merged_video_t_mod = torch.cat([t_mod_video_noisy, t_mod_video_cond], dim=1)
        merged_video_context_mask = torch.cat(
            [context_mask_video_noisy, context_mask_video_cond], dim=1
        )
        attention_mask = self._build_teacher_forcing_attention_mask(
            noisy_video_seq_len=noisy_video_seq_len,
            cond_video_seq_len=cond_video_seq_len,
            action_seq_len=action_tokens.shape[1],
            noisy_video_tokens_per_frame=noisy_video_tokens_per_frame,
            cond_video_tokens_per_frame=cond_video_tokens_per_frame,
            batch_size=action_tokens.shape[0],
            device=merged_video_tokens.device,
        )

        merged_video_out, action_out = self.mot.forward_joint_core(
            video_tokens=merged_video_tokens,
            action_tokens=action_tokens,
            video_freqs=merged_video_freqs,
            action_freqs=freqs_action,
            video_t_mod=merged_video_t_mod,
            action_t_mod=t_mod_action,
            video_context=context_video_noisy,
            video_context_mask=merged_video_context_mask,
            action_context=context_action,
            action_context_mask=context_mask_action,
            attention_mask=attention_mask,
        )

        pred_video_tokens = merged_video_out[:, :noisy_video_seq_len]
        pred_video = self.video_expert.post(
            pred_video_tokens,
            t_video_noisy,
            f_noisy,
            h_noisy,
            w_noisy,
        )
        pred_action = self.action_expert.post(action_out)
        return pred_video, pred_action

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        fuse_flag = inputs["fuse_vae_embedding_in_latents"]
        if not bool(getattr(self.video_expert, "seperated_timestep", False)) or not fuse_flag:
            raise ValueError(
                "Teacher-forcing requires token-wise `t_mod`; "
                "ensure `seperated_timestep=true` and `fuse_vae_embedding_in_latents=true`."
            )

        # Branch A: noisy video (for video denoising target).
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents_noisy = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        if inputs["first_frame_latents"] is not None:
            latents_noisy[:, :, 0:1] = inputs["first_frame_latents"]

        # Branch B: noisy action.
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        # Branch C: teacher-forcing cond-video.
        # Each sample is independently noised with probability `video_cond_noise_prob`.
        cond_noise_mask = torch.rand((batch_size,), device=self.device) < float(self.video_cond_noise_prob)
        timestep_video_cond = torch.zeros_like(timestep_video, dtype=input_latents.dtype, device=self.device)
        latents_cond = input_latents
        if bool(cond_noise_mask.any()):
            timestep_video_cond_sampled = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=input_latents.dtype,
            )
            timestep_video_cond = torch.where(cond_noise_mask, timestep_video_cond_sampled, timestep_video_cond)
            noise_video_cond = torch.randn_like(input_latents)
            latents_cond_noisy = self.train_video_scheduler.add_noise(
                input_latents, noise_video_cond, timestep_video_cond_sampled
            )
            cond_noise_selector = cond_noise_mask.view(batch_size, 1, 1, 1, 1)
            latents_cond = torch.where(cond_noise_selector, latents_cond_noisy, input_latents)
        if inputs["first_frame_latents"] is not None:
            latents_cond = latents_cond.clone()
            latents_cond[:, :, 0:1] = inputs["first_frame_latents"]

        pred_video, pred_action = self._teacher_forcing_training_denoise_core(
            latents_noisy=latents_noisy,
            latents_cond=latents_cond,
            noisy_action=noisy_action,
            timestep_video=timestep_video,
            timestep_video_cond=timestep_video_cond,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
        )

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    def _denoise_video(
        self,
        latents_video: torch.Tensor,
        timestep_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_self_attn_mask: Optional[torch.Tensor],
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        x_tokens, t, t_mod, context_emb, context_attn_mask, freqs, f, h, w, _ = (
            self.video_expert.prepare(
                x=latents_video,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            )
        )
        for block in self.video_expert.blocks:
            x_tokens = block(
                x_tokens,
                context_emb,
                t_mod,
                freqs,
                context_mask=context_attn_mask,
                self_attn_mask=video_self_attn_mask,
            )
        x = self.video_expert.head(x_tokens, t)
        return self.video_expert.unpatchify(x, (f, h, w))

    @torch.no_grad()
    def infer_video(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        video_sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        compile_video_infer: bool = False,
        initial_video_latents: Optional[torch.Tensor] = None,
        decode_video: bool = True,
    ) -> dict[str, Any]:
        """Generate only the future video; no action latent is allocated or denoised."""
        self.eval()

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(
            height, width, num_video_frames
        )
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                "`input_image` must be resized before infer, expected multiples of 16 "
                f"but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim != 2 or proprio.shape[0] != 1:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        video_shape = (1, self.vae.model.z_dim, latent_t, latent_h, latent_w)
        if initial_video_latents is None:
            generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
            latents_video = torch.randn(
                video_shape,
                generator=generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)
        else:
            if tuple(initial_video_latents.shape) != video_shape:
                raise ValueError(
                    "initial_video_latents has shape "
                    f"{tuple(initial_video_latents.shape)}, expected {video_shape}."
                )
            latents_video = initial_video_latents.detach().to(
                device=self.device, dtype=self.torch_dtype
            ).clone()

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image, tiled=tiled
        )
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    "`context/context_mask` must be [B,L,D]/[B,L], got "
                    f"{tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio
            )

        patch_t, patch_h, patch_w = (int(value) for value in self.video_expert.patch_size)
        video_tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
        video_seq_len = (latent_t // patch_t) * video_tokens_per_frame
        video_self_attn_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=self.device,
        )
        if compile_video_infer:
            if not hasattr(self, "_denoise_video_compiled"):
                self._denoise_video_compiled = torch.compile(self._denoise_video, fullgraph=True)
            denoise_video = self._denoise_video_compiled
        else:
            denoise_video = self._denoise_video

        resolved_video_shift = video_sigma_shift if video_sigma_shift is not None else sigma_shift
        timesteps, deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=resolved_video_shift,
        )
        for step_t, step_delta in zip(timesteps, deltas):
            timestep = step_t.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            pred_video = denoise_video(
                latents_video=latents_video,
                timestep_video=timestep,
                context=context,
                context_mask=context_mask,
                video_self_attn_mask=video_self_attn_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            latents_video = self.infer_video_scheduler.step(
                pred_video, step_delta, latents_video
            )
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        result: dict[str, Any] = {"video_latents": latents_video}
        if decode_video:
            result["video"] = self.decode_video_latents(latents_video, tiled=tiled)
        return result

    @torch.no_grad()
    def decode_video_latents(
        self,
        video_latents: torch.Tensor,
        *,
        tiled: bool = False,
        compile_vae_decode: bool = False,
    ) -> list[Any]:
        """Decode an already generated video without running either denoiser."""
        if compile_vae_decode and not tiled:
            if not hasattr(self, "_decode_latents_compiled"):
                self._decode_latents_compiled = torch.compile(
                    self._decode_latents,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
            return self._decode_latents_compiled(video_latents, tiled=False)
        return self._decode_latents(video_latents, tiled=tiled)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        num_action_candidates: int = 1,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        video_sigma_shift: Optional[float] = None,
        action_sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
        compile_action_infer: bool = False,
    ) -> dict[str, Any]:
        del test_action_with_infer_action
        if action is not None:
            logger.warning(
                "`FastWAMIDM.infer_joint` ignores `action` input; "
                "video is denoised in a standalone first stage."
            )

        out = self.infer_action(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            num_action_candidates=num_action_candidates,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            video_sigma_shift=video_sigma_shift,
            action_sigma_shift=action_sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            compile_action_infer=compile_action_infer,
        )
        return {
            "video": self._decode_latents(out["video_latents"], tiled=tiled),
            "action": out["action"],
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        num_action_candidates: int = 1,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        video_sigma_shift: Optional[float] = None,
        action_sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        compile_action_infer: bool = False,
        action_guidance_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        action_guidance_last_steps: int = 0,
        action_guidance_after_steps: Optional[Sequence[int]] = None,
        action_guidance_step_size: float = 0.0,
        action_guidance_horizon: Optional[int] = None,
        action_guidance_normalize_gradient: bool = True,
        action_guidance_max_delta_rms: Optional[float] = None,
        action_guidance_verify_descent: bool = False,
        action_guidance_trigger_reference_step: Optional[int] = None,
        action_guidance_trigger_decision_step: Optional[int] = None,
        action_guidance_trigger_ratio_threshold: Optional[float] = None,
        initial_video_latents: Optional[torch.Tensor] = None,
        precomputed_video_latents: Optional[torch.Tensor] = None,
        initial_action_latents: Optional[torch.Tensor] = None,
        action_flow_trace: bool = False,
        action_response_trace: bool = False,
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale
        self.eval()

        if num_action_candidates <= 0:
            raise ValueError(
                f"num_action_candidates must be positive, got {num_action_candidates}."
            )
        if initial_video_latents is not None and precomputed_video_latents is not None:
            raise ValueError(
                "Pass only one of initial_video_latents or precomputed_video_latents."
            )
        if action_guidance_last_steps < 0:
            raise ValueError("action_guidance_last_steps must be non-negative.")
        explicit_guidance_steps = (
            None
            if action_guidance_after_steps is None
            else tuple(int(step) for step in action_guidance_after_steps)
        )
        if explicit_guidance_steps is not None and len(set(explicit_guidance_steps)) != len(
            explicit_guidance_steps
        ):
            raise ValueError("action_guidance_after_steps must not contain duplicates.")
        if action_guidance_step_size < 0:
            raise ValueError("action_guidance_step_size must be non-negative.")
        if action_guidance_max_delta_rms is not None and action_guidance_max_delta_rms <= 0:
            raise ValueError(
                "action_guidance_max_delta_rms must be positive when set."
            )
        if action_response_trace and compile_action_infer:
            raise ValueError(
                "action_response_trace requires compile_action_infer=False because it "
                "collects per-layer diagnostic scalars."
            )
        guidance_enabled = (
            action_guidance_fn is not None
            and (
                action_guidance_last_steps > 0
                or bool(explicit_guidance_steps)
            )
            and action_guidance_step_size > 0
        )
        trigger_values = (
            action_guidance_trigger_reference_step,
            action_guidance_trigger_decision_step,
            action_guidance_trigger_ratio_threshold,
        )
        trigger_enabled = all(value is not None for value in trigger_values)
        if any(value is not None for value in trigger_values) and not trigger_enabled:
            raise ValueError(
                "Online action-guidance trigger requires reference_step, "
                "decision_step, and ratio_threshold together."
            )
        trigger_reference_step = (
            None
            if action_guidance_trigger_reference_step is None
            else int(action_guidance_trigger_reference_step)
        )
        trigger_decision_step = (
            None
            if action_guidance_trigger_decision_step is None
            else int(action_guidance_trigger_decision_step)
        )
        trigger_ratio_threshold = (
            None
            if action_guidance_trigger_ratio_threshold is None
            else float(action_guidance_trigger_ratio_threshold)
        )
        if trigger_enabled:
            if not guidance_enabled:
                raise ValueError("Online action-guidance trigger requires guidance to be enabled.")
            if trigger_reference_step < 0 or trigger_decision_step <= trigger_reference_step:
                raise ValueError(
                    "Online action-guidance trigger requires 0 <= reference_step < decision_step."
                )
            if trigger_ratio_threshold <= 0:
                raise ValueError("Online action-guidance trigger ratio_threshold must be positive.")
        if guidance_enabled and num_action_candidates != 1:
            raise ValueError("Action-flow guidance currently requires num_action_candidates=1.")
        if action_guidance_horizon is not None and not 0 < action_guidance_horizon <= action_horizon:
            raise ValueError(
                "action_guidance_horizon must be in [1, action_horizon], got "
                f"{action_guidance_horizon}."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        video_shape = (1, self.vae.model.z_dim, latent_t, latent_h, latent_w)
        action_shape = (
            num_action_candidates,
            action_horizon,
            self.action_expert.action_dim,
        )
        video_is_precomputed = precomputed_video_latents is not None
        if precomputed_video_latents is not None:
            if tuple(precomputed_video_latents.shape) != video_shape:
                raise ValueError(
                    "precomputed_video_latents has shape "
                    f"{tuple(precomputed_video_latents.shape)}, expected {video_shape}."
                )
            latents_video = precomputed_video_latents.detach().to(
                device=self.device, dtype=self.torch_dtype
            ).clone()
        elif initial_video_latents is None:
            latents_video = torch.randn(
                video_shape,
                generator=video_generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)
        else:
            if tuple(initial_video_latents.shape) != video_shape:
                raise ValueError(
                    "initial_video_latents has shape "
                    f"{tuple(initial_video_latents.shape)}, expected {video_shape}."
                )
            latents_video = initial_video_latents.detach().to(
                device=self.device, dtype=self.torch_dtype
            ).clone()
        if initial_action_latents is None:
            latents_action = torch.randn(
                action_shape,
                generator=action_generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)
        else:
            if tuple(initial_action_latents.shape) != action_shape:
                raise ValueError(
                    "initial_action_latents has shape "
                    f"{tuple(initial_action_latents.shape)}, expected {action_shape}."
                )
            latents_action = initial_action_latents.detach().to(
                device=self.device, dtype=self.torch_dtype
            ).clone()

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        if not video_is_precomputed:
            latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        patch_t, patch_h, patch_w = (int(value) for value in self.video_expert.patch_size)
        video_tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
        video_seq_len = (latent_t // patch_t) * video_tokens_per_frame
        video_self_attn_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=self.device,
        )
        if compile_action_infer:
            if not hasattr(self, "_denoise_video_compiled"):
                self._denoise_video_compiled = torch.compile(
                    self._denoise_video,
                    fullgraph=True,
                )
            denoise_video = self._denoise_video_compiled
        else:
            denoise_video = self._denoise_video

        # Stage 1: denoise video only, unless the caller already generated it.
        if not video_is_precomputed:
            resolved_video_shift = (
                video_sigma_shift if video_sigma_shift is not None else sigma_shift
            )
            infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_video.dtype,
                shift_override=resolved_video_shift,
            )
            for step_t_video, step_delta_video in zip(infer_timesteps_video, infer_deltas_video):
                timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
                pred_video = denoise_video(
                    latents_video=latents_video,
                    timestep_video=timestep_video,
                    context=context,
                    context_mask=context_mask,
                    video_self_attn_mask=video_self_attn_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                )
                latents_video = self.infer_video_scheduler.step(
                    pred_video, step_delta_video, latents_video
                )
                latents_video[:, :, 0:1] = first_frame_latents.clone()

        # Stage 2: freeze denoised video as cond and denoise action via video K/V cache.
        timestep_video_cond = torch.zeros(
            (latents_video.shape[0],), dtype=latents_video.dtype, device=self.device
        )
        (
            video_cond_tokens,
            _t_video_cond,
            video_cond_t_mod,
            video_cond_context,
            video_cond_context_mask,
            video_cond_freqs,
            _f_cond,
            _h_cond,
            _w_cond,
            video_tokens_per_frame,
        ) = self.video_expert.prepare(
            x=latents_video,
            timestep=timestep_video_cond,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_cond_tokens.shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=video_tokens_per_frame,
            device=video_cond_tokens.device,
        )
        video_attention_mask = attention_mask[:video_seq_len, :video_seq_len]
        action_attention_mask = attention_mask[video_seq_len:, :]
        if compile_action_infer:
            if not hasattr(self, "_prefill_video_cache_compiled"):
                self._prefill_video_cache_compiled = torch.compile(
                    self.mot.prefill_video_cache_tensor,
                    fullgraph=True,
                )
            if not hasattr(self, "_denoise_action_with_video_cache_compiled"):
                self._denoise_action_with_video_cache_compiled = torch.compile(
                    self._denoise_action_with_video_cache,
                    fullgraph=True,
                )
            prefill_video_cache = self._prefill_video_cache_compiled
            denoise_action_with_video_cache = self._denoise_action_with_video_cache_compiled
        else:
            prefill_video_cache = self.mot.prefill_video_cache_tensor
            denoise_action_with_video_cache = self._denoise_action_with_video_cache
        video_cache_k, video_cache_v = prefill_video_cache(
            video_tokens=video_cond_tokens,
            video_freqs=video_cond_freqs,
            video_t_mod=video_cond_t_mod,
            video_context=video_cond_context,
            video_context_mask=video_cond_context_mask,
            video_attention_mask=video_attention_mask,
        )
        if compile_action_infer:
            video_cache_k = [cache.clone() for cache in video_cache_k]
            video_cache_v = [cache.clone() for cache in video_cache_v]

        # The expensive future imagination and video prefill above are shared by
        # every candidate.  Only the action noise is sampled independently.
        # Expand the frozen conditioning tensors without materializing K copies
        # so Best-of-K action sampling remains substantially cheaper than K
        # complete video rollouts.
        if num_action_candidates > 1:
            context = context.expand(num_action_candidates, -1, -1)
            context_mask = context_mask.expand(num_action_candidates, -1)
            video_cache_k = [
                cache.expand(num_action_candidates, *cache.shape[1:]) for cache in video_cache_k
            ]
            video_cache_v = [
                cache.expand(num_action_candidates, *cache.shape[1:]) for cache in video_cache_v
            ]

        resolved_action_shift = (
            action_sigma_shift if action_sigma_shift is not None else sigma_shift
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=resolved_action_shift,
        )
        guidance_diagnostics = []
        total_action_steps = len(infer_timesteps_action)
        if explicit_guidance_steps is None:
            guidance_after_steps = set(
                range(
                    max(0, total_action_steps - action_guidance_last_steps),
                    total_action_steps,
                )
            )
        else:
            invalid_guidance_steps = [
                step
                for step in explicit_guidance_steps
                if step < 0 or step >= total_action_steps
            ]
            if invalid_guidance_steps:
                raise ValueError(
                    "action_guidance_after_steps entries must be valid zero-based "
                    f"flow-step indices in [0, {total_action_steps - 1}], got "
                    f"{invalid_guidance_steps}."
                )
            guidance_after_steps = set(explicit_guidance_steps)

        if trigger_enabled:
            if trigger_decision_step >= total_action_steps:
                raise ValueError(
                    "Online action-guidance trigger decision_step must be smaller than "
                    f"the number of action flow steps ({total_action_steps})."
                )
            if guidance_after_steps and trigger_decision_step >= min(guidance_after_steps):
                raise ValueError(
                    "Online action-guidance trigger decision_step must precede every "
                    "configured guidance step."
                )

        flow_trace = []
        response_trace = []
        trigger_reference_response = None
        trigger_decision_response = None
        trigger_response_ratio = None
        guidance_triggered = not trigger_enabled
        for step_index, (step_t_action, step_delta_action) in enumerate(
            zip(infer_timesteps_action, infer_deltas_action)
        ):
            state_before_flow_step = latents_action
            timestep_action = step_t_action.expand(num_action_candidates).to(
                dtype=latents_action.dtype,
                device=self.device,
            )
            collect_trigger_response = trigger_enabled and step_index in {
                trigger_reference_step,
                trigger_decision_step,
            }
            response_layers = [] if action_response_trace or collect_trigger_response else None
            pred_action = denoise_action_with_video_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_cache_k=video_cache_k,
                video_cache_v=video_cache_v,
                action_attention_mask=action_attention_mask,
                response_metrics=response_layers,
                video_tokens_per_frame=video_tokens_per_frame,
            )
            if response_layers is not None:
                r_future_layer_mean = float(
                    sum(item["r_future"] for item in response_layers) / len(response_layers)
                )
                if action_response_trace:
                    response_trace.append(
                        {
                            "denoise_step": int(step_index),
                            "timestep": float(step_t_action.float().item()),
                            "r_future_layer_mean": r_future_layer_mean,
                            "r_video_layer_mean": float(
                                sum(item["r_video"] for item in response_layers)
                                / len(response_layers)
                            ),
                            "layers": response_layers,
                        }
                    )
                if trigger_enabled and step_index == trigger_reference_step:
                    trigger_reference_response = r_future_layer_mean
                if trigger_enabled and step_index == trigger_decision_step:
                    trigger_decision_response = r_future_layer_mean
                    if trigger_reference_response is None:
                        raise RuntimeError("Missing online trigger reference response.")
                    trigger_response_ratio = trigger_decision_response / max(
                        trigger_reference_response, 1e-12
                    )
                    guidance_triggered = trigger_response_ratio <= trigger_ratio_threshold

            guidance_applied = (
                guidance_enabled
                and step_index in guidance_after_steps
                and guidance_triggered
            )

            clean_action_estimate = None
            if action_flow_trace or guidance_applied:
                clean_action_estimate = self.infer_action_scheduler.estimate_clean_sample(
                    model_output=pred_action,
                    sample=latents_action,
                    timestep=step_t_action,
                ).detach()

            # Keep the scheduler ordinary update as the reference boundary.
            # The existing additive guidance implementation applies its
            # correction after this ordinary flow update.
            base_step = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action
            )
            if guidance_applied:
                # FastWAM uses v = epsilon - a0 and
                # a_sigma = (1-sigma) * a0 + sigma * epsilon.
                # Therefore a0_hat = a_sigma - sigma * v.
                sigma = (
                    step_t_action.float()
                    / float(self.infer_action_scheduler.num_train_timesteps)
                ).reshape(1, 1, 1)
                with torch.enable_grad():
                    clean_action = (
                        latents_action.float() - sigma * pred_action.float()
                    ).detach().requires_grad_(True)
                    guidance_loss = action_guidance_fn(clean_action)
                    if guidance_loss.numel() != 1:
                        guidance_loss = guidance_loss.mean()
                    loss_gradient = torch.autograd.grad(
                        guidance_loss, clean_action, only_inputs=True
                    )[0]

                horizon = (
                    latents_action.shape[1]
                    if action_guidance_horizon is None
                    else int(action_guidance_horizon)
                )
                prefix_gradient = loss_gradient[:, :horizon]
                gradient_rms = prefix_gradient.square().mean().sqrt().clamp_min(1e-8)
                guidance_gradient = torch.zeros_like(loss_gradient)
                guidance_gradient[:, :horizon] = (
                    prefix_gradient / gradient_rms
                    if action_guidance_normalize_gradient
                    else prefix_gradient
                )
                unclipped_delta_rms = (
                    float(action_guidance_step_size)
                    * guidance_gradient[:, :horizon].square().mean().sqrt()
                )
                if action_guidance_max_delta_rms is not None:
                    clip_scale = torch.clamp(
                        float(action_guidance_max_delta_rms)
                        / unclipped_delta_rms.clamp_min(1e-8),
                        max=1.0,
                    )
                    guidance_gradient = guidance_gradient * clip_scale
                correction_rms = (
                    float(action_guidance_step_size)
                    * guidance_gradient[:, :horizon].square().mean().sqrt()
                )
                if not torch.isfinite(guidance_gradient).all():
                    raise FloatingPointError("Non-finite JEPA action guidance gradient.")

                verified_loss = None
                if action_guidance_verify_descent:
                    with torch.no_grad():
                        verified_loss = action_guidance_fn(
                            clean_action.detach()
                            - float(action_guidance_step_size) * guidance_gradient.detach()
                        )
                        if verified_loss.numel() != 1:
                            verified_loss = verified_loss.mean()

                latents_action = self.infer_action_scheduler.step_with_loss_guidance(
                    pred_action,
                    step_delta_action,
                    latents_action,
                    guidance_gradient,
                    action_guidance_step_size,
                )
                guidance_diagnostics.append(
                    {
                        "flow_step": int(step_index),
                        "after_flow_step": int(step_index),
                        "loss": float(guidance_loss.detach().cpu()),
                        "loss_after_clean_step": (
                            None
                            if verified_loss is None
                            else float(verified_loss.detach().cpu())
                        ),
                        "gradient_rms": float(gradient_rms.detach().cpu()),
                        "normalize_gradient": bool(action_guidance_normalize_gradient),
                        "unclipped_delta_rms": float(unclipped_delta_rms.detach().cpu()),
                        "correction_rms": float(correction_rms.detach().cpu()),
                    }
                )
            else:
                latents_action = base_step

            if action_flow_trace:
                flow_trace.append(
                    {
                        "step": int(step_index),
                        "timestep": float(step_t_action.float().item()),
                        "state_before_flow_step": state_before_flow_step.detach().float().cpu().clone(),
                        "x_before": base_step.detach().float().cpu().clone(),
                        "x_after": latents_action.detach().float().cpu().clone(),
                        "x_base": base_step.detach().float().cpu().clone(),
                        "clean_action_estimate": clean_action_estimate.float().cpu().clone(),
                        "guidance_applied": bool(guidance_applied),
                    }
                )

        action_out = latents_action.detach().to(device="cpu", dtype=torch.float32)
        result = {
            "video_latents": latents_video,
            "action": action_out[0] if num_action_candidates == 1 else action_out,
            "action_guidance": guidance_diagnostics,
            "action_guidance_trigger": {
                "enabled": bool(trigger_enabled),
                "reference_step": trigger_reference_step,
                "decision_step": trigger_decision_step,
                "ratio_threshold": trigger_ratio_threshold,
                "reference_response": trigger_reference_response,
                "decision_response": trigger_decision_response,
                "response_ratio": trigger_response_ratio,
                "triggered": bool(guidance_triggered) if trigger_enabled else None,
            },
        }
        if action_flow_trace:
            result["action_flow_trace"] = flow_trace
            result["action_flow_initial_latents"] = (
                initial_action_latents.detach().float().cpu().clone()
                if initial_action_latents is not None
                else None
            )
            result["action_flow_final_action"] = action_out
        if action_response_trace:
            result["action_response_trace"] = response_trace
        return result

    @torch.no_grad()
    def infer_action_from_video(
        self,
        *,
        video_latents: torch.Tensor,
        **kwargs,
    ) -> dict[str, Any]:
        """Denoise only actions while attending to an already generated future video."""
        action_infer_mode = kwargs.pop("action_infer_mode", "idm")
        if action_infer_mode != "idm":
            raise ValueError("infer_action_from_video requires action_infer_mode='idm'.")
        kwargs["precomputed_video_latents"] = video_latents
        return FastWAMIDM.infer_action(self, **kwargs)
