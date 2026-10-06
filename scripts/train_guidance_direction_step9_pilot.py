"""100-step pilot: supervise the direction of the step-9 JEPA correction."""
import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/data/pxchen/PACT-WAM')
CACHE = ROOT/'artifacts/libero10_predictor_action_oneepoch_20260930'
OUT = ROOT/'artifacts/libero10_predictor_direction_step9_pilot100_20260930'
SOURCE = ROOT/'checkpoints/vjepa2-ac-libero90-euler-fixed-rot180-step2000-inference.pt'
sys.path.insert(0, str(ROOT))


def emit(event, **kw):
    print(json.dumps(dict(time=time.strftime('%F %T'), event=event, **kw)), flush=True)


def configure_predictor(predictor):
    """Preserve learned dynamics; adapt action interfaces and the last four blocks."""
    predictor.requires_grad_(False)
    for name in ('action_encoder', 'state_encoder', 'predictor_proj', 'predictor_norm'):
        getattr(predictor, name).requires_grad_(True)
    for block in predictor.predictor_blocks[-4:]:
        block.requires_grad_(True)
    predictor.extrinsics_encoder.requires_grad_(False)
    predictor.use_activation_checkpointing = True
    heads, blocks = [], []
    for name, p in predictor.named_parameters():
        if not p.requires_grad:
            continue
        (blocks if name.startswith('predictor_blocks.') else heads).append(p)
    assert heads and blocks
    return heads, blocks


def rms_clip(x, limit=0.02, eps=1e-8):
    rms = x.float().square().mean(dim=(-2, -1), keepdim=True).sqrt()
    return x * torch.clamp(limit/(rms + eps), max=1.0), rms.squeeze()


def main(smoke=False):
    from experiments.libero.vjepa2_ac_ranker import VJEPA2ACRanker, LiberoACAdapter
    from scripts.smoke_guidance_three_step_joint import make_train_method
    from scripts.train_guidance_libero10_one_epoch import make_wam, tasks

    torch.manual_seed(239)
    np.random.seed(239)
    rows = tasks()
    assert SOURCE.exists() and len(list((CACHE/'wam').glob('*.pt'))) == 500
    model, _, _ = make_wam(need_prompts=False)
    train_method = make_train_method(model)
    ranker = VJEPA2ACRanker(checkpoint_path=str(SOURCE),
        vjepa2_repo='/data/pxchen/vjepa2', device='cpu', dtype=torch.float32,
        max_ac_steps=2)
    del ranker.encoder
    ranker.device = torch.device('cuda')
    predictor = ranker.predictor.cuda().eval()
    heads, blocks = configure_predictor(predictor)
    params = heads + blocks
    optimizer = torch.optim.AdamW([
        dict(params=heads, lr=1e-5),
        dict(params=blocks, lr=3e-6),
    ], weight_decay=0.04)
    adapter = LiberoACAdapter(low_level_steps_per_ac_step=4)
    trainids = [i for i, r in enumerate(rows) if r['split'] == 'train']
    valids = [i for i, r in enumerate(rows) if r['split'] == 'val']
    order = trainids[:2] if smoke else np.random.default_rng(239).permutation(trainids).tolist()[:100]
    runroot = OUT/('smoke' if smoke else 'train100')
    runroot.mkdir(parents=True, exist_ok=False)
    metrics_path = runroot/'metrics.jsonl'

    def one(index, update):
        r = torch.load(CACHE/'wam'/f'{index:04d}.pt', map_location='cpu', weights_only=False)
        reps = torch.load(CACHE/'latent'/f'{index:04d}.pt', map_location='cpu', weights_only=True).cuda()
        kw = r['kw'].copy()
        kw['precomputed_video_latents'] = kw.pop('video_latents')
        assert kw.pop('action_infer_mode', 'idm') == 'idm'
        kw.pop('compile_action_infer', None)
        kw = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in kw.items()}
        state = torch.as_tensor(adapter.state_from_observation(r['obs']), device='cuda')
        scale, offset = r['scale'].cuda(), r['offset'].cuda()

        def energy(a):
            ac = adapter.convert_torch((a[:, :8]-offset)/scale, state)
            return ranker.differentiable_energy(current_rep=reps[0], target_reps=reps[1:],
                ac_actions=ac.actions, ac_states=ac.states).mean()

        kw.update(action_guidance_fn=energy, action_guidance_after_steps=[9],
            action_guidance_step_size=4.0, action_guidance_horizon=8,
            action_guidance_normalize_gradient=False, action_guidance_max_delta_rms=0.02,
            action_guidance_verify_descent=False, action_flow_trace=False)
        optimizer.zero_grad(set_to_none=True)
        started = time.monotonic()
        with torch.enable_grad(), torch.backends.cuda.sdp_kernel(enable_flash=False,
                enable_math=True, enable_mem_efficient=False), \
                torch.autograd.graph.save_on_cpu(pin_memory=True):
            result = train_method(**kw)
            nodes = result['training_nodes']
            clean_nodes = result['training_clean_actions']
            assert len(nodes) == len(clean_nodes) == 1
            gradient = nodes[0][:, :8, :6]
            clean = clean_nodes[0][:, :8, :6]
            expert = r['expert'].cuda()[None, :8, :6]
            desired, desired_raw_rms = rms_clip(expert-clean.detach(), 0.02)
            predicted = -4.0*gradient
            flat_pred = predicted.flatten(1)
            flat_desired = desired.flatten(1)
            cosine_each = F.cosine_similarity(flat_pred, flat_desired, dim=1, eps=1e-8)
            direction_loss = 1.0-cosine_each.mean()
            magnitude_loss = F.smooth_l1_loss(predicted/0.02, desired/0.02, beta=0.5)
            loss = direction_loss + 0.25*magnitude_loss
            corrected = clean.detach()+predicted.detach()
            clean_mse = F.mse_loss(clean.detach(), expert)
            corrected_mse = F.mse_loss(corrected, expert)
            if update:
                loss.backward()
        predicted_rms = float(predicted.detach().float().square().mean().sqrt())
        row = dict(window=index, loss=float(loss.detach()),
            direction_loss=float(direction_loss.detach()), magnitude_loss=float(magnitude_loss.detach()),
            cosine=float(cosine_each.detach().mean()), positive_fraction=float((cosine_each.detach()>0).float().mean()),
            predicted_correction_rms=predicted_rms, desired_raw_rms=float(desired_raw_rms),
            clean_action_mse=float(clean_mse), corrected_action_mse=float(corrected_mse),
            seconds=time.monotonic()-started)
        if update:
            grads = [p.grad for p in params if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            preclip = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            row['predictor_grad_norm'] = preclip
            row['global_clip_active'] = bool(preclip > 1.0)
            optimizer.step()
        del result, nodes, clean_nodes, gradient, clean, expert, desired, predicted, reps, kw, loss
        gc.collect(); torch.cuda.empty_cache()
        return row

    def evaluate(label):
        values = []
        for j, index in enumerate(valids, 1):
            row = one(index, False)
            row.update(split='val', eval_at=label, val_index=j)
            values.append(row)
            with metrics_path.open('a') as f:
                f.write(json.dumps(row)+'\n')
            if j % 10 == 0:
                emit('VAL_PROGRESS', eval_at=label, completed=j,
                    cosine=float(np.mean([x['cosine'] for x in values])))
        return values

    baseline = None if smoke else evaluate('step_0000')
    train_metrics = []
    for step, index in enumerate(order, 1):
        row = one(index, True)
        row.update(split='train', step=step)
        train_metrics.append(row)
        with metrics_path.open('a') as f:
            f.write(json.dumps(row)+'\n')
        if step <= 3 or step % 10 == 0:
            emit('TRAIN_PROGRESS', step=step, **{k:v for k,v in row.items() if k not in ('split','step')})
    if smoke:
        summary = dict(passed=True, steps=2, source_checkpoint=str(SOURCE), metrics=train_metrics,
            peak_gib=torch.cuda.max_memory_allocated()/2**30)
    else:
        final = evaluate('step_0100')
        def aggregate(items):
            keys = ('cosine','positive_fraction','predicted_correction_rms','desired_raw_rms',
                    'clean_action_mse','corrected_action_mse')
            return {k:float(np.mean([x[k] for x in items])) for k in keys}
        b, f = aggregate(baseline), aggregate(final)
        summary = dict(passed=True, steps=100, source_checkpoint=str(SOURCE),
            supervision='step9 direction + clipped magnitude, EEF dims 0:6 only',
            trainable='action/state encoders, predictor proj/norm, blocks 20-23',
            learning_rates=dict(heads=1e-5, last_four_blocks=3e-6),
            baseline=b, final=f, change={k:f[k]-b[k] for k in b},
            train_first10_loss=float(np.mean([x['loss'] for x in train_metrics[:10]])),
            train_last10_loss=float(np.mean([x['loss'] for x in train_metrics[-10:]])),
            clip_rate=float(np.mean([x['global_clip_active'] for x in train_metrics])),
            peak_gib=torch.cuda.max_memory_allocated()/2**30)
        payload = dict(predictor=predictor.state_dict(), optimizer=optimizer.state_dict(),
            step=100, source_checkpoint=str(SOURCE), objective=summary['supervision'],
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        tmp = runroot/'step_0100.tmp'; torch.save(payload, tmp); os.replace(tmp, runroot/'step_0100.pt')
    (runroot/'summary.json').write_text(json.dumps(summary, indent=2))
    emit('SMOKE_COMPLETED' if smoke else 'PILOT_COMPLETED', **summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['smoke','train'], required=True)
    args = parser.parse_args()
    try:
        main(args.phase == 'smoke')
    except Exception:
        traceback.print_exc()
        sys.exit(1)
