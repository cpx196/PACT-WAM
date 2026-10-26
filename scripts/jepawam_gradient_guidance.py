"""Runtime-only JEPA-WAM flow adapter. No native policy files or weights changed."""
from types import MethodType
import torch
import torch.nn.functional as F


class JEPAGuidance:
    def __init__(self, model, predictor, stats, unnorm_key, scale=4., cap=.02,
                 steps=(2, 3), verify=True):
        self.model, self.predictor = model, predictor
        self.scale, self.cap, self.steps, self.verify = scale, cap, tuple(steps), verify
        action_stats = model.get_action_stats(unnorm_key)
        self.low = torch.tensor(action_stats['q01'], device='cuda', dtype=torch.float32)
        self.high = torch.tensor(action_stats['q99'], device='cuda', dtype=torch.float32)
        self.mask = torch.tensor(action_stats.get('mask', [True]*7), device='cuda', dtype=torch.bool)
        self.stats = {k: (torch.tensor(v['mean'], device='cuda', dtype=torch.float32),
                          torch.tensor(v['std'], device='cuda', dtype=torch.float32).clamp_min(1e-6))
                      for k, v in stats.items()}
        self.context = None
        self.records = []
        self.check_gradient = False
        self.original = model.action_head.predict_action
        self.hook = model.register_forward_hook(self._capture)

    def _capture(self, module, args, output):
        if self.context is not None:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                self.context['W'] = self.model.visual_token_cosine_head.align_dimension(
                    output['llm_hidden'][:, 1:577]).float()

    def install(self):
        owner = self

        @torch.no_grad()
        def guided(head, vl_embs, proprio):
            if owner.context is None or owner.scale == 0:
                return owner.original(vl_embs, proprio)
            state = head._prepare_state(proprio).to(dtype=vl_embs.dtype)
            x = torch.randn((vl_embs.shape[0], head.action_horizon, head.action_dim),
                            device=vl_embs.device, dtype=vl_embs.dtype)
            n = head.num_inference_timesteps
            assert n == 4 and x.shape[1:] == (20, 7), (n, x.shape)
            # FP32 accumulation avoids BF16 swallowing small gradient updates.
            for step in range(n):
                time = step / n
                bucket = torch.full((x.shape[0],), int(time*head.num_timestep_buckets),
                                    device=x.device, dtype=torch.long)
                velocity = head._predict_velocity(vl_embs, x, bucket, state)
                clean = x.float() + (1-time)*velocity.float()
                if step in owner.steps:
                    next_x = x.float() + velocity.float()/n
                    delta, record = owner.correct(clean)
                    next_x = next_x - delta
                    record['flow_step_zero_based'] = step
                    owner.records.append(record)
                else:
                    # Exactly native arithmetic until the first correction.
                    next_x = x + (1.0/n)*velocity
                x = next_x
            return x

        self.model.action_head.predict_action = MethodType(guided, self.model.action_head)
        return self

    def prepare(self, image, state):
        # Match the predictor's separate batch-one current-frame encoder cache.
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            z = self.model.vision_backbone(
                self.model.vision_backbone.image_transform(image)[None].cuda())
        assert z.shape == (1, 576, 1024)
        self.context = dict(z=z, state=torch.as_tensor(state, device='cuda', dtype=torch.float32)[None])
        self.records = []

    def correct(self, clean):
        # OpenVLA.predict_action has @inference_mode; enable_grad alone is insufficient.
        with torch.inference_mode(False), torch.enable_grad(), torch.autocast('cuda', enabled=False):
            a = clean.detach().float().clone().requires_grad_(True)
            z = self.context['z'].detach().float().clone()
            w = self.context['W'].detach().float().clone()
            state = self.context['state'].detach().float().clone()
            assert z.shape == w.shape == (1, 576, 1024)
            am, ast = [v.clone() for v in self.stats['action']]
            sm, sst = [v.clone() for v in self.stats['state']]
            low, high, mask = self.low.clone(), self.high.clone(), self.mask.clone()

            def energy(normalized):
                executable = torch.where(mask, .5*(normalized+1)*(high-low+1e-8)+low, normalized)
                # Native LIBERO gripper binarize/invert ONCE; no gradient to gripper.
                grip = -torch.sign(2*executable[..., 6:7].detach()-1)
                executable = torch.cat((executable[..., :6], grip), dim=-1)
                prediction = self.predictor(z, (executable-am)/ast, (state-sm)/sst)
                # Released Qwen alignment is trained with cosine only: its raw
                # norm is not calibrated to encoder/predictor latent magnitude.
                return (1-F.cosine_similarity(prediction.float(), w, dim=-1)).mean()

            loss = energy(a)
            gradient = torch.autograd.grad(loss, a, only_inputs=True)[0]
            gradient = torch.cat((gradient[..., :6], torch.zeros_like(gradient[..., 6:])), dim=-1)
            assert torch.isfinite(loss) and torch.isfinite(gradient).all()
            raw_delta = self.scale*gradient
            rms = raw_delta[..., :6].square().mean(dim=(1, 2), keepdim=True).sqrt()
            factor = (self.cap/rms.clamp_min(1e-12)).clamp(max=1) if self.cap is not None else torch.ones_like(rms)
            delta = (raw_delta*factor).detach()
            after = None
            if self.verify:
                with torch.no_grad():
                    after = float(energy(a.detach()-delta))
            record = dict(loss_before=float(loss.detach()), loss_after=after,
                          gradient_rms=float(gradient[..., :6].square().mean().sqrt()),
                          unclipped_delta_rms=float(rms.mean()),
                          correction_rms=float(delta[..., :6].square().mean().sqrt()),
                          capped=bool((factor < 1).any()),
                          gripper_delta_max=float(delta[..., 6].abs().max()))
            if self.check_gradient:
                direction = gradient.detach()/gradient.square().mean().sqrt().clamp_min(1e-12)
                eps = .002
                with torch.no_grad():
                    fd = float((energy(a.detach()+eps*direction)-energy(a.detach()-eps*direction))/(2*eps))
                analytic = float((gradient*direction).sum())
                relative_error = abs(fd-analytic)/max(abs(analytic),1e-8)
                record.update(finite_difference=fd,analytic_directional_derivative=analytic,
                              finite_difference_relative_error=relative_error)
                assert fd > 0 and relative_error < .15, record
                self.check_gradient = False
            assert record['gripper_delta_max'] == 0
            assert self.cap is None or record['correction_rms'] <= self.cap+1e-6
            return delta, record

    def close(self):
        self.model.action_head.predict_action = self.original
        self.hook.remove()
