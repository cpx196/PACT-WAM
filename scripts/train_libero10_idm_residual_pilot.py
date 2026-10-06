"""LaWAM-inspired shared IDM + residual MLP; 450/50 trajectory-separated windows."""
import argparse
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
OUT = ROOT/'artifacts/libero10_idm21m_mlp_pilot450_20261004'
BASE = ROOT/'artifacts/libero10_predictor_action_oneepoch_20260930'
OLD = ROOT/'artifacts/libero10_mlp_residual_pilot_20261004'


def emit(event, **kw):
    print(json.dumps(dict(time=time.strftime('%F %T'), event=event, **kw)), flush=True)


def save(value, path):
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    source = torch.load(OLD/'features.pt', map_location='cpu', weights_only=False)
    rows = json.loads((BASE/'windows.json').read_text())
    assert source['rows'] == rows and len(rows) == 500
    z = []
    for i in range(500):
        item = torch.load(BASE/'latent'/f'{i:04d}.pt', map_location='cpu', weights_only=True)
        assert item.shape == (3, 256, 1408) and item.dtype == torch.float32
        assert torch.isfinite(item).all()
        z.append(item)
    data = dict(tokens=torch.stack(z), action=source['action'], expert=source['expert'],
                rows=rows, train_ids=source['train_ids'], val_ids=source['val_ids'],
                source_metadata=source['metadata'])
    assert len(data['train_ids']) == 450 and len(data['val_ids']) == 50
    train_demos = {(rows[i]['task_file'], rows[i]['demo']) for i in data['train_ids']}
    val_demos = {(rows[i]['task_file'], rows[i]['demo']) for i in data['val_ids']}
    assert not train_demos & val_demos
    assert data['action'].shape == (500, 8, 7) and data['expert'].shape == (500, 8, 6)
    save(data, OUT/'dataset.pt')
    (OUT/'windows.json').write_text(json.dumps(rows, indent=2))
    emit('DATASET_READY', train=450, val=50, token_shape=list(data['tokens'].shape),
         bytes=(OUT/'dataset.pt').stat().st_size)


def spatial_position(dim=512):
    grid = torch.arange(16).float()
    y, x = torch.meshgrid(grid, grid, indexing='ij')
    frequencies = 1 / (10000 ** (torch.arange(dim//4).float()/(dim//4)))
    xx = x.flatten()[:, None] * frequencies[None]
    yy = y.flatten()[:, None] * frequencies[None]
    return torch.cat((xx.sin(), xx.cos(), yy.sin(), yy.cos()), -1)


class IDM(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_norm = nn.LayerNorm(1408)
        self.project = nn.Linear(1408, 512)
        self.register_buffer('spatial', spatial_position())
        self.time_embedding = nn.Parameter(torch.randn(2, 512)*0.02)
        self.query = nn.Parameter(torch.randn(1, 1, 512)*0.02)
        self.cross_norm = nn.LayerNorm(512)
        self.cross_attention = nn.MultiheadAttention(512, 8, dropout=0.1, batch_first=True)
        self.query_norm = nn.LayerNorm(512)
        self.query_ffn = nn.Sequential(nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 512))
        layer = nn.TransformerEncoderLayer(512, 8, 2048, dropout=0.1,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 6, enable_nested_tensor=False)
        # Avoid identical initialization of TransformerEncoder clones.
        for block in self.transformer.layers:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
        self.output_norm = nn.LayerNorm(512)
        self.output = nn.Linear(512, 128)

    def forward(self, pairs):
        # [B,2,256,1408] -> retain all 512 visual tokens, plus one motion query.
        x = self.project(self.input_norm(pairs))
        x = x + self.spatial[None, None] + self.time_embedding[None, :, None]
        context = x.flatten(1, 2)
        q = self.query.expand(x.shape[0], -1, -1)
        kv = self.cross_norm(context)
        q = q + self.cross_attention(q, kv, kv, need_weights=False)[0]
        q = q + self.query_ffn(self.query_norm(q))
        joined = self.transformer(torch.cat((context, q), 1))
        return self.output(self.output_norm(joined[:, -1]))


class Corrector(nn.Module):
    def __init__(self, mean, std, use_jepa=True):
        super().__init__()
        self.use_jepa = use_jepa
        self.idm = IDM() if use_jepa else None
        self.register_buffer('action_mean', mean)
        self.register_buffer('action_std', std)
        self.la_norm = nn.LayerNorm(128)
        self.mlp = nn.Sequential(nn.Linear(312, 512), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(512, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 48))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, action, tokens):
        if self.use_jepa:
            pairs = torch.stack((tokens[:, :2], tokens[:, 1:3]), 1)
            la = self.idm(pairs.flatten(0, 1)).reshape(action.shape[0], 2, 128)
            la = self.la_norm(la).flatten(1)
        else:
            la = action.new_zeros((action.shape[0], 256))
        a = (action.flatten(1)-self.action_mean)/self.action_std
        correction = self.mlp(torch.cat((a, la), -1)).reshape(-1, 8, 6)
        return action[:, :, :6]+correction, correction


def run(smoke=False):
    torch.set_num_threads(4)
    torch.manual_seed(239)
    np.random.seed(239)
    data = torch.load(OUT/'dataset.pt', map_location='cpu', weights_only=False)
    train, val = data['train_ids'], data['val_ids']
    mean = data['action'][train].flatten(1).mean(0)
    std = data['action'][train].flatten(1).std(0).clamp_min(1e-4)
    tensors = {key: data[key].cuda() for key in ('tokens', 'action', 'expert')}
    cfg = dict(train_windows=450, val_windows=50, seed=239, batch_size=8, epochs=50,
               idm=dict(layers=6, width=512, heads=8, ffn=2048, queries=1, latent_dim=128),
               mlp=[312, 512, 512, 48], dropout=0.1, idm_lr=1e-4, mlp_lr=3e-4,
               weight_decay=0.01, grad_clip=1.0, early_stop_patience=10,
               dtype='float32', rms_cap=None, residual_penalty=0.0,
               zero_output_initialization=True, pooling=False, vae=False,
               input='current/WAM future full tokens plus original first8 normalized actions',
               loss='normalized first6 executable demonstrated action MSE',
               gripper='original WAM output, unchanged',
               frozen='FastWAM/JEPA encoder/predictor; only IDM and MLP are trained',
               control='MLP-only with zero LA input; NOT parameter-count matched',
               source=data['source_metadata'])
    phase = OUT/('smoke' if smoke else 'train')
    phase.mkdir(exist_ok=True)
    (phase/'config.json').write_text(json.dumps(cfg, indent=2))
    results = {}
    variants = ['idm'] if smoke else ['idm', 'no_jepa']
    for variant in variants:
        torch.manual_seed(239)
        model = Corrector(mean, std, variant == 'idm').cuda()
        mlp_params = list(model.mlp.parameters())+list(model.la_norm.parameters())
        groups = [dict(params=mlp_params, lr=3e-4)]
        if model.idm is not None:
            groups.append(dict(params=model.idm.parameters(), lr=1e-4))
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        params = sum(p.numel() for p in model.parameters())
        dest = phase/variant
        dest.mkdir(exist_ok=True)
        def batch(ids):
            return tensors['action'][ids], tensors['tokens'][ids], tensors['expert'][ids]
        @torch.no_grad()
        def evaluate(ids):
            model.eval()
            errors, bases, corrections, cosines = [], [], [], []
            for start in range(0, len(ids), 8):
                a, z, y = batch(ids[start:start+8])
                pred, delta = model(a, z)
                errors.append((pred-y).square().mean((1, 2)))
                bases.append((a[:, :, :6]-y).square().mean((1, 2)))
                corrections.append(delta.square().mean((1, 2)).sqrt())
                cosines.append(F.cosine_similarity(delta.flatten(1), (y-a[:, :, :6]).flatten(1), dim=1))
            e, b, c, cos = map(torch.cat, (errors, bases, corrections, cosines))
            return dict(mse=float(e.mean()), baseline_mse=float(b.mean()),
                        better_windows=int((e<b).sum()), worse_windows=int((e>b).sum()),
                        correction_rms_mean=float(c.mean()), correction_cosine=float(cos.mean()))
        def checkpoint(name, epoch, step, best):
            save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=cfg,
                      variant=variant, epoch=epoch, step=step, best_val=best,
                      torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), dest/name)
        baseline = evaluate(val)
        emit('BASELINE', variant=variant, parameters=params, val=baseline,
             idm_parameters=sum(p.numel() for p in model.idm.parameters()) if model.idm else 0)
        if smoke:
            ids = train[:8]
            before = evaluate(ids)['mse']
            started = time.monotonic()
            norms = []
            for step in range(20):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                a, z, y = batch(ids)
                pred, delta = model(a, z)
                loss = F.mse_loss(pred, y)
                loss.backward()
                grads = [p.grad for p in model.idm.parameters() if p.grad is not None]
                assert all(torch.isfinite(g).all() for g in grads)
                norms.append(float(torch.stack([g.square().sum() for g in grads]).sum().sqrt()))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            after = evaluate(ids)['mse']
            assert after < before and max(norms[1:]) > 0, (before, after, norms)
            checkpoint('checkpoint.pt', 0, 20, after)
            ck = torch.load(dest/'checkpoint.pt', map_location='cpu', weights_only=False)
            assert ck['step'] == 20 and len(ck['optimizer']['state']) > 0
            results[variant] = dict(passed=True, initial_mse=before, final_mse=after,
                                    idm_gradient_norm_max=max(norms), seconds=time.monotonic()-started,
                                    peak_gib=torch.cuda.max_memory_allocated()/2**30, parameters=params)
            emit('SMOKE_COMPLETED', **results[variant])
        else:
            started = time.monotonic()
            best, best_epoch, stale, step = float('inf'), 0, 0, 0
            history = []
            for epoch in range(50):
                order = np.random.default_rng(239+epoch).permutation(train).tolist()
                model.train()
                losses, clipped = [], 0
                for start in range(0, len(order), 8):
                    optimizer.zero_grad(set_to_none=True)
                    a, z, y = batch(order[start:start+8])
                    pred, delta = model(a, z)
                    loss = F.mse_loss(pred, y)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    clipped += int(float(norm) > 1)
                    optimizer.step()
                    step += 1
                    losses.append(float(loss))
                v = evaluate(val)
                tr = evaluate(train[:64])
                item = dict(epoch=epoch+1, step=step, optimization_mse=float(np.mean(losses)),
                            train_probe=tr, val=v, clip_fraction=clipped/len(losses),
                            seconds=time.monotonic()-started,
                            peak_gib=torch.cuda.max_memory_allocated()/2**30)
                history.append(item)
                with (dest/'metrics.jsonl').open('a') as file:
                    file.write(json.dumps(item)+'\n')
                if v['mse'] < best:
                    best, best_epoch, stale = v['mse'], epoch+1, 0
                    checkpoint('best.pt', epoch+1, step, best)
                else:
                    stale += 1
                checkpoint('last.pt', epoch+1, step, best)
                if (epoch+1) % 5 == 0:
                    checkpoint(f'epoch{epoch+1:03d}.pt', epoch+1, step, best)
                emit('EPOCH_COMPLETED', variant=variant, **item)
                if stale >= 10:
                    emit('EARLY_STOP', variant=variant, epoch=epoch+1, best_epoch=best_epoch)
                    break
            results[variant] = dict(parameters=params, baseline=baseline, best_val_mse=best,
                                    best_epoch=best_epoch, final=history[-1], steps=step,
                                    seconds=time.monotonic()-started)
        del model, optimizer
        torch.cuda.empty_cache()
    (phase/'summary.json').write_text(json.dumps(results, indent=2))
    emit('ALL_COMPLETED', phase='smoke' if smoke else 'train', summary=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'smoke', 'train'], required=True)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    else:
        run(args.phase == 'smoke')
