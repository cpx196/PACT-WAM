"""Measure guidance and predictor gradient scales without updating weights."""

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/data/pxchen/PACT-WAM')
CACHE = ROOT / 'artifacts/libero10_predictor_action_oneepoch_20260930'
OUT = ROOT / 'artifacts/libero10_guidance_backward_scale_20260930'
sys.path.insert(0, str(ROOT))


def emit(event, **values):
    row = dict(time=time.strftime('%F %T'), event=event, **values)
    print(json.dumps(row), flush=True)
    return row


def main():
    from experiments.libero.vjepa2_ac_ranker import VJEPA2ACRanker, LiberoACAdapter
    from scripts.smoke_guidance_three_step_joint import make_train_method
    from scripts.train_guidance_libero10_one_epoch import make_wam, tasks

    OUT.mkdir(parents=True, exist_ok=True)
    rows = tasks()
    train_ids = [i for i, row in enumerate(rows) if row['split'] == 'train']
    order = np.random.default_rng(239).permutation(train_ids).tolist()
    sample_ids = order[: int(os.environ.get('PACT_DIAGNOSTIC_WINDOWS', '6'))]

    model, _, _ = make_wam(need_prompts=False)
    model.requires_grad_(False)
    train_method = make_train_method(model)
    ranker = VJEPA2ACRanker(
        checkpoint_path=str(ROOT / 'checkpoints/vjepa2-ac-libero90-euler-fixed-rot180-step2000-inference.pt'),
        vjepa2_repo='/data/pxchen/JEPA_WAM', device='cpu', dtype=torch.float32,
        max_ac_steps=2,
    )
    del ranker.encoder
    ranker.device = torch.device('cuda')
    predictor = ranker.predictor.cuda().eval().requires_grad_(True)
    predictor.extrinsics_encoder.requires_grad_(False)
    predictor.use_activation_checkpointing = True
    params = [(name, p) for name, p in predictor.named_parameters() if p.requires_grad]
    adapter = LiberoACAdapter(low_level_steps_per_ac_step=4)
    records = []

    for sample_index, window in enumerate(sample_ids, 1):
        record = torch.load(CACHE / 'wam' / f'{window:04d}.pt', map_location='cpu', weights_only=False)
        reps = torch.load(CACHE / 'latent' / f'{window:04d}.pt', map_location='cpu', weights_only=True).cuda()
        kwargs = record['kw'].copy()
        kwargs['precomputed_video_latents'] = kwargs.pop('video_latents')
        assert kwargs.pop('action_infer_mode', 'idm') == 'idm'
        kwargs.pop('compile_action_infer', None)
        kwargs = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in kwargs.items()}
        state = torch.as_tensor(adapter.state_from_observation(record['obs']), device='cuda')
        scale, offset = record['scale'].cuda(), record['offset'].cuda()

        def energy(action):
            ac = adapter.convert_torch((action[:, :8] - offset) / scale, state)
            return ranker.differentiable_energy(
                current_rep=reps[0], target_reps=reps[1:],
                ac_actions=ac.actions, ac_states=ac.states,
            ).mean()

        kwargs.update(
            action_guidance_fn=energy,
            action_guidance_after_steps=[7, 8, 9],
            action_guidance_step_size=4.0,
            action_guidance_horizon=8,
            action_guidance_normalize_gradient=False,
            action_guidance_max_delta_rms=0.02,
            action_guidance_verify_descent=False,
            action_flow_trace=False,
        )
        predictor.zero_grad(set_to_none=True)
        with (
            torch.enable_grad(),
            torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False),
            torch.autograd.graph.save_on_cpu(pin_memory=True),
        ):
            result = train_method(**kwargs)
            action = result['action']
            if action.ndim == 2:
                action = action.unsqueeze(0)
            loss = F.mse_loss(action[:, :8, :6], record['expert'].cuda()[None, :, :6])
            nodes = result['training_nodes']
            node_grads = torch.autograd.grad(loss, nodes, retain_graph=True)
            loss.backward()

        grad_sq = 0.0
        grad_max = 0.0
        nonzero_tensors = 0
        groups = {}
        for name, parameter in params:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            square = float(gradient.square().sum())
            grad_sq += square
            grad_max = max(grad_max, float(gradient.abs().max()))
            nonzero_tensors += int(square > 0)
            group = name.split('.')[0]
            groups[group] = groups.get(group, 0.0) + square

        diagnostics = result['action_guidance']
        row = emit(
            'BACKWARD_SCALE', sample=sample_index, window=window,
            action_loss=float(loss.detach()),
            guidance=[{
                'step': d['flow_step'],
                'energy': d['loss'],
                'action_gradient_rms': d['gradient_rms'],
                'unclipped_delta_rms': d['unclipped_delta_rms'],
                'correction_rms': d['correction_rms'],
                'cap_active': d['unclipped_delta_rms'] > 0.0200001,
                'loss_to_gradient_node_norm': float(g.detach().float().norm()),
            } for d, g in zip(diagnostics, node_grads)],
            predictor_global_grad_norm=math.sqrt(grad_sq),
            predictor_grad_max_abs=grad_max,
            predictor_nonzero_grad_tensors=nonzero_tensors,
            predictor_group_grad_norms={k: math.sqrt(v) for k, v in groups.items()},
            would_global_clip_at_1=math.sqrt(grad_sq) > 1.0,
        )
        records.append(row)
        del result, action, loss, nodes, node_grads, kwargs, reps
        gc.collect()
        torch.cuda.empty_cache()

    summary = {
        'windows': len(records),
        'predictor_global_grad_norm': {
            'min': min(r['predictor_global_grad_norm'] for r in records),
            'median': float(np.median([r['predictor_global_grad_norm'] for r in records])),
            'max': max(r['predictor_global_grad_norm'] for r in records),
        },
        'cap_active': {
            str(step): sum(d['cap_active'] for r in records for d in r['guidance'] if d['step'] == step)
            for step in (7, 8, 9)
        },
        'guidance_gradient_rms_median': {
            str(step): float(np.median([
                d['action_gradient_rms'] for r in records for d in r['guidance'] if d['step'] == step
            ])) for step in (7, 8, 9)
        },
        'node_grad_norm_median': {
            str(step): float(np.median([
                d['loss_to_gradient_node_norm'] for r in records for d in r['guidance'] if d['step'] == step
            ])) for step in (7, 8, 9)
        },
        'global_clip_count': sum(r['would_global_clip_at_1'] for r in records),
    }
    (OUT / 'records.json').write_text(json.dumps(records, indent=2))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    emit('BACKWARD_SCALE_COMPLETED', **summary)


if __name__ == '__main__':
    main()
