"""Matched frozen-IDM LA versus zero-LA residual Transformer pilot."""
import argparse
import fcntl
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path('/data/pxchen/PACT-WAM')
DATA = ROOT/'artifacts/libero10_idm21m_mlp_pilot450_20261004/dataset.pt'
LA = ROOT/'artifacts/libero10_idm_oracle_comparison450_20261004/latent_actions.pt'
OUT = ROOT/'artifacts/libero10_transformer_la450_20261004'
DENSE = ROOT/'artifacts/libero10_idm_cache_stride32_20261004'
MANIFEST = ROOT/'artifacts/libero10_idm_dense_windows_20261004/windows_stride32.json'
IDM_CKPT = ROOT/'artifacts/libero10_idm_pretrain_then_mlp450_20261004/stage1/best.pt'
EXPECTED_IDM_EPOCH = None
IDM_EPOCH = None


def emit(event, **kw):
    print(json.dumps(dict(time=time.strftime('%F %T'), event=event, **kw)), flush=True)


def save(value, path):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_norm = nn.LayerNorm(256)
        self.cross_norm = nn.LayerNorm(256)
        self.ff_norm = nn.LayerNorm(256)
        self.self_attn = nn.MultiheadAttention(256, 8, dropout=.1, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(256, 8, dropout=.1, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Dropout(.1),
                                nn.Linear(1024, 256), nn.Dropout(.1))
        self.drop = nn.Dropout(.1)
        self.gate = nn.Parameter(torch.tensor(.1))

    def forward(self, x, memory):
        q = self.self_norm(x)
        x = x + self.drop(self.self_attn(q, q, q, need_weights=False)[0])
        x = x + self.gate.tanh()*self.drop(self.cross_attn(
            self.cross_norm(x), memory, memory, need_weights=False)[0])
        return x + self.ff(self.ff_norm(x))


class Corrector(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', mean.reshape(8, 7))
        self.register_buffer('std', std.reshape(8, 7))
        self.action_proj = nn.Linear(7, 256)
        self.la_proj = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 256))
        self.action_pos = nn.Parameter(torch.zeros(1, 8, 256))
        self.future_pos = nn.Parameter(torch.zeros(1, 2, 256))
        nn.init.normal_(self.action_pos, std=.02)
        nn.init.normal_(self.future_pos, std=.02)
        self.blocks = nn.ModuleList([Block() for _ in range(4)])
        self.output = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 6))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, action, la):
        x = self.action_proj((action-self.mean)/self.std)+self.action_pos
        memory = self.la_proj(la)+self.future_pos
        for block in self.blocks:
            x = block(x, memory)
        delta = self.output(x)
        return action[:, :, :6]+delta, delta


def dense_features():
    global IDM_EPOCH
    from scripts.train_libero10_idm_two_stage450 import InverseDynamics
    rows = json.loads(MANIFEST.read_text())
    train = [i for i, row in enumerate(rows) if row['split'] == 'train']
    val = [i for i, row in enumerate(rows) if row['split'] == 'val']
    assert (len(rows), len(train), len(val)) == (4307, 3871, 436)
    assert {(rows[i]['task_file'], rows[i]['demo']) for i in train}.isdisjoint(
        {(rows[i]['task_file'], rows[i]['demo']) for i in val})
    dest = OUT/'dense_features.pt'
    if dest.exists():
        data = torch.load(dest, map_location='cpu', weights_only=False)
        assert data['rows'] == rows and data['idm_checkpoint'] == str(IDM_CKPT)
        IDM_EPOCH = data.get('idm_epoch')
        if EXPECTED_IDM_EPOCH is not None:
            assert IDM_EPOCH == EXPECTED_IDM_EPOCH
        return data
    model = InverseDynamics().cuda().eval().requires_grad_(False)
    checkpoint = torch.load(IDM_CKPT, map_location='cpu', weights_only=False)
    IDM_EPOCH = checkpoint['epoch']
    if EXPECTED_IDM_EPOCH is not None:
        assert IDM_EPOCH == EXPECTED_IDM_EPOCH, 'Unexpected IDM checkpoint epoch'
    model.load_state_dict(checkpoint['model'])
    del checkpoint
    actions, experts, latent = [], [], []
    started = time.monotonic()
    emit('DENSE_CACHE_START', total=len(rows), idm_checkpoint=str(IDM_CKPT), idm_epoch=IDM_EPOCH)
    with torch.no_grad():
        for start in range(0, len(rows), 8):
            tokens = []
            for i in range(start, min(start+8, len(rows))):
                item = torch.load(DENSE/'tokens'/f'{i:05d}.pt', map_location='cpu', weights_only=False)
                assert item['index'] == i and item['row'] == rows[i]
                z = torch.cat((item['current'][None], item['wam_future']), 0)
                assert z.shape == (3, 256, 1408) and torch.isfinite(z).all()
                a, y = item['action'].float(), item['expert_normalized_first6'].float()
                assert a.shape == (8, 7) and y.shape == (8, 6)
                expected = (item['expert_executable']*item['scale']+item['offset'])[:, :6]
                assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)
                assert torch.isfinite(a).all() and torch.isfinite(y).all()
                actions.append(a)
                experts.append(y)
                tokens.append(z)
            _, la = model(torch.stack(tokens).cuda())
            assert torch.isfinite(la).all()
            latent.append(la.cpu())
            processed = len(actions)
            if start == 0 or processed % 256 == 0 or processed == len(rows):
                emit('DENSE_CACHE_PROGRESS', processed=processed, total=len(rows),
                     seconds=time.monotonic()-started,
                     seconds_per_window=(time.monotonic()-started)/processed)
    data = dict(action=torch.stack(actions), expert=torch.stack(experts),
                wam_la=torch.cat(latent), train_ids=train, val_ids=val, rows=rows,
                idm_checkpoint=str(IDM_CKPT), idm_epoch=IDM_EPOCH, source_manifest=str(MANIFEST))
    save(data, dest)
    del model
    torch.cuda.empty_cache()
    emit('DENSE_CACHE_COMPLETED', seconds=time.monotonic()-started)
    return data


def main():
    global OUT, IDM_CKPT, EXPECTED_IDM_EPOCH
    parser = argparse.ArgumentParser()
    parser.add_argument('--dense', action='store_true')
    parser.add_argument('--idm-checkpoint', type=Path)
    parser.add_argument('--expected-idm-epoch', type=int)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.dense:
        OUT = ROOT/'artifacts/libero10_transformer_la_stride32_20261004'
    if args.idm_checkpoint is not None:
        assert args.dense and args.out is not None, 'Custom IDM requires dense mode and separate output'
        IDM_CKPT = args.idm_checkpoint.resolve()
    if args.out is not None:
        OUT = args.out.resolve()
    EXPECTED_IDM_EPOCH = args.expected_idm_epoch
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'launcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT/'summary.json').exists(), 'Do not overwrite completed pilot'
    if args.dense:
        data = dense_features()
        cache = data
    else:
        data = torch.load(DATA, map_location='cpu', weights_only=False)
        cache = torch.load(LA, map_location='cpu', weights_only=False)
        assert cache['rows'] == data['rows']
    train, val = data['train_ids'], data['val_ids']
    assert set(train).isdisjoint(val)
    a, y = data['action'].cuda(), data['expert'].cuda()
    la = cache['wam_la'].cuda()
    assert la.shape == (len(train)+len(val), 2, 128) and torch.isfinite(la).all()
    mean = data['action'][train].flatten(1).mean(0)
    std = data['action'][train].flatten(1).std(0).clamp_min(1e-4)
    del data, cache
    config = dict(train=len(train), val=len(val), batch=8, max_epochs=50, patience=10,
                  seed=239, lr=3e-4, weight_decay=.01, grad_clip=1, dropout=.1,
                  layers=4, hidden=256, heads=8, ffn=1024, dtype='float32',
                  frozen_idm=True, latent_source=str(OUT/'dense_features.pt' if args.dense else LA),
                  dataset=str(MANIFEST if args.dense else DATA),
                  idm_checkpoint=str(IDM_CKPT), idm_epoch=IDM_EPOCH, warm_start_transformer=False,
                  inputs='8 raw action tokens + 2 WAM-future IDM latent action tokens',
                  target='normalized demonstrated first 8x6 actions',
                  gripper_unchanged=True, rms_cap=None, residual_penalty=0,
                  matched_initialization=True, matched_batch_order=True)
    torch.manual_seed(239)
    smoke = Corrector(mean, std).cuda()
    opt = torch.optim.AdamW(smoke.parameters(), lr=3e-4, weight_decay=.01)
    ix = train[:8]
    initial = {k: v.detach().cpu().clone() for k, v in smoke.state_dict().items()}
    grads = []
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        pred, delta = smoke(a[ix], la[ix])
        loss = F.mse_loss(pred, y[ix])
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(smoke.parameters(), 1, error_if_nonfinite=True)
        grads.append(dict(step=step+1, loss=float(loss), grad_norm=float(norm),
                          action_projection_grad=float(smoke.action_proj.weight.grad.norm()),
                          la_projection_grad=float(smoke.la_proj[-1].weight.grad.norm())))
        opt.step()
    assert grads[-1]['la_projection_grad'] > 0
    config['parameters'] = sum(p.numel() for p in smoke.parameters())
    (OUT/'config.json').write_text(json.dumps(config, indent=2))
    (OUT/'smoke.json').write_text(json.dumps(grads, indent=2))
    emit('SMOKE_PASSED', parameters=config['parameters'], gradients=grads)
    del smoke, opt
    results = {}
    for variant in ('wam_la', 'zero_la'):
        torch.manual_seed(239)
        model = Corrector(mean, std).cuda()
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        feat = la if variant == 'wam_la' else torch.zeros_like(la)
        dest = OUT/variant
        dest.mkdir(exist_ok=True)

        @torch.no_grad()
        def evaluate(ids, override=None):
            model.eval()
            errors, deltas, cosines = [], [], []
            for start in range(0, len(ids), 32):
                subset = ids[start:start+32]
                f = feat[subset] if override is None else override[start:start+32]
                pred, delta = model(a[subset], f)
                errors.append((pred-y[subset]).square().mean((1, 2)))
                deltas.append(delta.square().mean((1, 2)).sqrt())
                cosines.append(F.cosine_similarity(delta.flatten(1),
                    (y[subset]-a[subset, :, :6]).flatten(1), dim=1))
            error = torch.cat(errors)
            baseline = (a[ids, :, :6]-y[ids]).square().mean((1, 2))
            return dict(mse=float(error.mean()), baseline_mse=float(baseline.mean()),
                        better=int((error<baseline).sum()), worse=int((error>baseline).sum()),
                        correction_rms=float(torch.cat(deltas).mean()),
                        correction_cosine=float(torch.cat(cosines).mean()))

        def checkpoint(name, epoch, step, best):
            save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                      config=config, variant=variant, epoch=epoch, step=step, best_val=best,
                      torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), dest/name)

        best, best_epoch, stale, step = float('inf'), 0, 0, 0
        started = time.monotonic()
        for epoch in range(50):
            ids = np.random.default_rng(239+epoch).permutation(train).tolist()
            model.train()
            losses, clipped = [], 0
            for start in range(0, len(ids), 8):
                ix = ids[start:start+8]
                optimizer.zero_grad(set_to_none=True)
                pred, delta = model(a[ix], feat[ix])
                loss = F.mse_loss(pred, y[ix])
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                clipped += int(float(norm)>1)
                optimizer.step()
                losses.append(float(loss))
                step += 1
            v = evaluate(val)
            row = dict(epoch=epoch+1, step=step, optimization_mse=float(np.mean(losses)),
                       train=evaluate(train), val=v, clip_rate=clipped/len(losses),
                       gates=[float(b.gate.tanh()) for b in model.blocks],
                       seconds=time.monotonic()-started)
            with (dest/'metrics.jsonl').open('a') as file:
                file.write(json.dumps(row)+'\n')
            if v['mse'] < best:
                best, best_epoch, stale = v['mse'], epoch+1, 0
                checkpoint('best.pt', epoch+1, step, best)
            else:
                stale += 1
            checkpoint('last.pt', epoch+1, step, best)
            if (epoch+1)%5 == 0:
                checkpoint(f'epoch{epoch+1:03d}.pt', epoch+1, step, best)
            emit('EPOCH_COMPLETED', variant=variant, **row)
            if stale >= 10:
                break
        model.load_state_dict(torch.load(dest/'best.pt', map_location='cpu', weights_only=False)['model'])
        permutation = torch.randperm(len(val), generator=torch.Generator().manual_seed(239)).cuda()
        checks = dict(normal=evaluate(val), zero_la=evaluate(val, torch.zeros_like(la[val])),
                      shuffled_la=evaluate(val, la[val][permutation]),
                      swapped_la=evaluate(val, la[val].flip(1)),
                      mean_la=evaluate(val, la[train].mean(0, keepdim=True).expand(len(val), -1, -1)))
        results[variant] = dict(best_val_mse=best, best_epoch=best_epoch, final=row,
                                diagnostics=checks, seconds=time.monotonic()-started)
        emit('VARIANT_COMPLETED', variant=variant, **results[variant])
        del model, optimizer
    (OUT/'summary.json').write_text(json.dumps(dict(config=config, variants=results), indent=2))
    emit('ALL_COMPLETED', best_mse={k: v['best_val_mse'] for k, v in results.items()})


if __name__ == '__main__':
    main()
