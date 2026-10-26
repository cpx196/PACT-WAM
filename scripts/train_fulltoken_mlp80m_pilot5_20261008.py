"""Matched full-token residual MLP / zero-JEPA control, five full epochs.

Frozen calibrated step2000 Predictor and WAM caches. No spatial/time pooling,
IDM, LA, decoder, online gradient guidance or auxiliary losses. Offline MSE only.
"""
import argparse
import fcntl
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path('/data/pxchen/PACT-WAM')
CACHE = ROOT/'artifacts/libero10_idm_cache_stride32_20261004/tokens'
PCACHE = ROOT/'artifacts/libero10_calibrated_idm_arb_joint10_20261007/cache'
MANIFEST = ROOT/'artifacts/libero10_idm_dense_windows_20261004/windows_stride32.json'


def emit(event, **kw):
    print(json.dumps(dict(time=time.strftime('%F %T'), event=event, **kw)), flush=True)


def write_json(value, path):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


class FullTokenMLP(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.token_projection = nn.Linear(1408, 64)
        self.net = nn.Sequential(
            nn.Linear(65536+56+7, 1024), nn.GELU(),
            nn.Linear(1024, 3072), nn.GELU(),
            nn.Linear(3072, 3072), nn.GELU(), nn.Linear(3072, 48))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, action, state, visual=None):
        if visual is not None:
            assert visual.shape[1:] == (2, 2, 256, 1408)
            tokens = self.token_projection(visual).flatten(1)
        else:
            tokens = action.new_zeros(len(action), 65536)
        low = torch.cat((action.flatten(1), state), -1)
        low = (low-self.mean)/self.std
        delta = self.net(torch.cat((low, tokens), -1)).reshape(-1, 8, 6)
        return action[:, :, :6]+delta


def norm(parameters):
    values = [p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def load_data(out):
    started = time.monotonic()
    rows = json.loads(MANIFEST.read_text())
    tr = [i for i, r in enumerate(rows) if r['split'] == 'train']
    va = [i for i, r in enumerate(rows) if r['split'] == 'val']
    assert (len(rows), len(tr), len(va)) == (4307, 3871, 436)
    assert {(rows[i]['task_file'], rows[i]['demo']) for i in tr}.isdisjoint(
        {(rows[i]['task_file'], rows[i]['demo']) for i in va})
    meta = json.loads((PCACHE/'config.json').read_text())
    assert meta['checkpoint_step'] == 2000 and meta['calibration']['fitted_split'] == 'train_only'
    assert meta['manifest'] == str(MANIFEST) and meta['main_view_rotation'] == 180
    visual = torch.empty(len(rows), 2, 2, 256, 1408, dtype=torch.float32)
    aa, ss, yy = [], [], []
    for i, row in enumerate(rows):
        old = torch.load(CACHE/f'{i:05d}.pt', map_location='cpu', weights_only=False, mmap=True)
        new = torch.load(PCACHE/'tokens'/f'{i:05d}.pt', map_location='cpu', weights_only=False, mmap=True)
        assert old['index'] == new['index'] == i and old['row'] == new['row'] == row
        assert new['checkpoint'] == meta['checkpoint'] and new['calibration'] == meta['calibration']
        # Deterministic flatten order: future time, source(P/W), spatial position, channel.
        feature = torch.stack((new['predicted_future'], old['wam_future']), 1).float()
        a, state, y = old['action'].float(), old['state'].float(), old['expert_normalized_first6'].float()
        assert feature.shape == (2, 2, 256, 1408) and a.shape == (8, 7)
        assert state.shape == (7,) and y.shape == (8, 6)
        assert all(torch.isfinite(x).all() for x in (feature, a, state, y))
        assert torch.allclose(y, (old['expert_executable']*old['scale']+old['offset'])[:, :6], atol=1e-6, rtol=1e-6)
        visual[i].copy_(feature)
        aa.append(a); ss.append(state); yy.append(y)
        if (i+1) % 512 == 0:
            emit('CACHE_LOAD_PROGRESS', completed=i+1, total=len(rows), seconds=time.monotonic()-started)
    a, state, y = torch.stack(aa), torch.stack(ss), torch.stack(yy)
    low = torch.cat((a.flatten(1), state), -1)
    mean, std = low[tr].mean(0), low[tr].std(0).clamp_min(1e-4)
    write_json([dict(index=i, **row) for i, row in enumerate(rows)], out/'window_manifest.json')
    emit('DATA_READY', train_windows=len(tr), val_windows=len(va), load_seconds=time.monotonic()-started,
         visual_cache_gib=visual.numel()*4/2**30)
    return rows, tr, va, visual, a.cuda(), state.cuda(), y.cuda(), mean, std, meta


def train_arm(variant, data, args, config, deadline):
    rows, tr, va, visual, a, state, y, mean, std, meta = data
    include = variant == 'full_tokens'
    dest = args.out/variant
    dest.mkdir()
    torch.manual_seed(240)
    model = FullTokenMLP(mean, std).cuda()
    assert sum(p.numel() for p in model.parameters()) == 80_001_136
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=.01)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    def predict(ix):
        return model(a[ix], state[ix], visual[ix].cuda() if include else None)

    @torch.no_grad()
    def evaluate(ix):
        model.eval()
        errors, rms = [], []
        for j in range(0, len(ix), args.batch):
            batch = ix[j:j+args.batch]
            pred = predict(batch)
            errors.append((pred-y[batch]).square().mean((1, 2)).cpu())
            rms.append((pred-a[batch, :, :6]).square().mean((1, 2)).sqrt().cpu())
        raw = float((a[ix, :, :6]-y[ix]).square().mean())
        return dict(mse=float(torch.cat(errors).mean()), baseline_mse=raw,
                    correction_rms=float(torch.cat(rms).mean()))

    # Zero-head initialization must reproduce the original WAM action exactly.
    with torch.no_grad():
        assert torch.equal(predict(tr[:2]), a[tr[:2], :, :6])
    initial = dict(train=evaluate(tr), val=evaluate(va))
    write_json(initial, dest/'initial.json')
    emit('INITIAL', variant=variant, **initial)
    first = model.token_projection.weight.detach().clone()
    smoke_grads = []
    for step in range(5):
        ix = tr[step*args.batch:(step+1)*args.batch]
        model.train(); opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(predict(ix), y[ix]); assert torch.isfinite(loss)
        loss.backward()
        smoke_grads.append(norm(model.token_projection.parameters()))
        g = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
        assert math.isfinite(g)
        opt.step()
    changed = not torch.equal(first, model.token_projection.weight)
    assert changed and max(smoke_grads[1:]) > 0 if include else not changed
    snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    clone = FullTokenMLP(mean, std).cuda(); clone.load_state_dict(snapshot); clone.eval(); model.eval()
    with torch.no_grad():
        ix = tr[:2]; z = visual[ix].cuda() if include else None
        assert torch.equal(clone(a[ix], state[ix], z), model(a[ix], state[ix], z))
    write_json(dict(passed=True, steps=5, visual_grad_norms=smoke_grads,
                    visual_parameter_updated=changed, checkpoint_state_roundtrip=True,
                    peak_gib=torch.cuda.max_memory_allocated()/2**30), dest/'smoke.json')
    emit('SMOKE_PASSED', variant=variant, visual_parameter_updated=changed, grad_norm=smoke_grads[-1])
    del clone, snapshot, first, model, opt
    torch.cuda.empty_cache()
    # Smoke does not warm-start either formal arm: identical fresh weights and order.
    torch.manual_seed(240)
    model = FullTokenMLP(mean, std).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=.01)
    history = []; step = 0; clip_count = 0

    def save(epoch, name):
        tmp = dest/(name+'.tmp')
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), config=config,
                        variant=variant, epoch=epoch, step=step, next_epoch=epoch,
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                        numpy_rng=np.random.get_state()), tmp)
        tmp.replace(dest/(name+'.pt'))
        write_json(dict(checkpoint=str(dest/(name+'.pt')), epoch=epoch, step=step), dest/'latest.json')

    for epoch in range(1, args.epochs+1):
        t0 = time.monotonic(); total = 0.; n = 0
        order = np.random.default_rng(239+epoch).permutation(tr).tolist()
        model.train()
        for j in range(0, len(order), args.batch):
            if time.monotonic() >= deadline:
                save(epoch-1, 'time_budget_partial')
                raise TimeoutError('Fixed wall-time budget reached; saved own partial state, no other jobs touched')
            ix = order[j:j+args.batch]; opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(predict(ix), y[ix]); assert torch.isfinite(loss)
            loss.backward()
            g = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
            clip_count += g > 1.
            opt.step(); step += 1; total += float(loss)*len(ix); n += len(ix)
            if step % 100 == 0:
                emit('TRAIN_PROGRESS', variant=variant, epoch=epoch, step=step, mse=float(loss),
                     peak_gib=torch.cuda.max_memory_allocated()/2**30)
        metrics = dict(epoch=epoch, step=step, optimization_mse=total/n,
                       train=evaluate(tr), val=evaluate(va), clip_rate=clip_count/step,
                       peak_gib=torch.cuda.max_memory_allocated()/2**30,
                       epoch_compute_seconds=time.monotonic()-t0, seconds=time.monotonic()-started)
        save(epoch, f'epoch{epoch:03d}')
        metrics['epoch_seconds_including_save'] = time.monotonic()-t0
        history.append(metrics)
        with (dest/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics)+'\n')
        emit('EPOCH_COMPLETED', variant=variant, **metrics)
    # Real disk checkpoint + optimizer-state audit, not just a cloned in-memory state.
    ck = torch.load(dest/f'epoch{args.epochs:03d}.pt', map_location='cpu', weights_only=False, mmap=True)
    assert ck['step'] == step and len(ck['optimizer']['state']) > 0
    assert all(float(v['step']) == step for v in ck['optimizer']['state'].values())
    assert all(torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all()
               for v in ck['optimizer']['state'].values())
    clone = FullTokenMLP(mean, std).cuda(); clone.load_state_dict(ck['model']); clone.eval(); model.eval()
    with torch.no_grad():
        ix = va[:2]; z = visual[ix].cuda() if include else None
        assert torch.equal(clone(a[ix], state[ix], z), model(a[ix], state[ix], z))
    result = dict(initial=initial, final=history[-1], epochs=history, parameters=80_001_136,
                  checkpoint_forward_roundtrip=True, optimizer_saved_and_checked=True,
                  seconds=time.monotonic()-started, best_epoch=min(history, key=lambda r: r['val']['mse'])['epoch'])
    write_json(result, dest/'summary.json'); emit('ARM_COMPLETED', variant=variant, **result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--max-seconds', type=int, default=1800)
    ap.add_argument('--cpu-preflight', action='store_true')
    args = ap.parse_args(); torch.set_num_threads(4)
    torch.manual_seed(240); np.random.seed(239)
    model = FullTokenMLP(torch.zeros(63), torch.ones(63))
    assert sum(p.numel() for p in model.parameters()) == 80_001_136
    if args.cpu_preflight:
        a, s, z = torch.randn(1, 8, 7), torch.randn(1, 7), torch.randn(1, 2, 2, 256, 1408)
        with torch.no_grad():
            assert torch.equal(model(a, s, z), a[:, :, :6])
            assert torch.equal(model(a, s), a[:, :, :6])
        emit('CPU_PREFLIGHT_PASSED', parameters=80_001_136); return
    del model
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out/'training.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (args.out/'config.json').exists()
    (args.out/'train.pid').write_text(str(os.getpid()))
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    deadline = time.monotonic()+args.max_seconds
    data = load_data(args.out); rows, tr, va, _, _, _, _, _, _, meta = data
    config = dict(parameters=80_001_136, token_projection=[1408, 64], mlp_hidden=[1024, 3072, 3072],
                  feature_shape=[2, 2, 256, 1408], flattened_visual=65536, feature_order='time,source(P/W),space,channel',
                  source_order=['predictor','wam'], spatial_time_pooling=False, channel_projection=True,
                  epochs=args.epochs, batch=args.batch, steps_per_epoch=math.ceil(len(tr)/args.batch),
                  lr=3e-5, weight_decay=.01, grad_clip=1., dropout=0., precision='FP32; TF32 disabled',
                  model_seed=240, data_seed=239, identical_initialization_and_batch_order=True,
                  loss='normalized demonstrated first8x6 action MSE only', residual_head_zero_init=True,
                  gripper='unchanged', rms_cap=False, auxiliary_losses=False, idm=False, la=False, decoder=False,
                  input_normalization='action and Euler-EEF state mean/std from train only; native cached patch-LN visual tokens',
                  frozen='FastWAM, JEPA Encoder, calibrated Predictor step2000; cached outputs only',
                  predictor_metadata=meta, real_future_input=False, train_ids=tr, val_ids=va,
                  train_windows=len(tr), val_windows=len(va), split='450/50 demonstration-disjoint',
                  early_stopping=False, max_run_seconds=args.max_seconds,
                  saves='every full epoch; model, optimizer, RNG', evaluation='full train/val every epoch',
                  caveat='offline action MSE, not rollout success; changes capacity, feature representation and data vs old pooled MLP')
    write_json(config, args.out/'config.json')
    results = {}
    for variant in ('full_tokens', 'zero_jepa'):
        results[variant] = train_arm(variant, data, args, config, deadline)
        torch.cuda.empty_cache()
    write_json(dict(config=config, results=results), args.out/'summary.json')
    emit('ALL_COMPLETED', final_val={k: v['final']['val']['mse'] for k, v in results.items()})


if __name__ == '__main__':
    main()
