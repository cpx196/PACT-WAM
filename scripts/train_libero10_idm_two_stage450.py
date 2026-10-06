"""Stage1 visual inverse dynamics, stage2 frozen IDM + supervised residual MLP."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scripts.train_libero10_idm_residual_pilot import IDM

ROOT = Path('/data/pxchen/PACT-WAM')
OUT = ROOT/'artifacts/libero10_idm_pretrain_then_mlp450_20261004'
PREVIOUS = ROOT/'artifacts/libero10_idm21m_mlp_pilot450_20261004'
REAL = ROOT/'artifacts/libero10_predictor_energy_ranking_pilot100_20260930/real_latent'


def emit(event, **kw):
    print(json.dumps(dict(time=time.strftime('%F %T'), event=event, **kw)), flush=True)


def save(value, path):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


class InverseDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.idm = IDM()
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 512), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(512, 512), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(512, 24))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, tokens):
        pairs = torch.stack((tokens[:, :2], tokens[:, 1:3]), 1)
        raw = self.idm(pairs.flatten(0, 1)).reshape(tokens.shape[0], 2, 128)
        prediction = self.head(raw).reshape(-1, 8, 6)
        return prediction, raw


class ResidualHead(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.la_norm = nn.LayerNorm(128)
        self.mlp = nn.Sequential(nn.Linear(312, 512), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(512, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 48))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, action, la):
        a = (action.flatten(1)-self.mean)/self.std
        delta = self.mlp(torch.cat((a, self.la_norm(la).flatten(1)), -1)).reshape(-1, 8, 6)
        return action[:, :, :6]+delta, delta


def load():
    data = torch.load(PREVIOUS/'dataset.pt', map_location='cpu', weights_only=False)
    real = []
    for i in range(500):
        z = torch.load(REAL/f'{i:04d}.pt', map_location='cpu', weights_only=True)
        assert z.shape == (3, 256, 1408) and torch.isfinite(z).all()
        assert torch.allclose(z[0], data['tokens'][i, 0], atol=2e-5, rtol=2e-5)
        real.append(z)
    data['real_tokens'] = torch.stack(real)
    assert len(data['train_ids']) == 450 and len(data['val_ids']) == 50
    return data


def stage1(smoke=False):
    torch.manual_seed(239)
    data = load()
    train, val = data['train_ids'], data['val_ids']
    z, y = data['real_tokens'].cuda(), data['expert'].cuda()
    # No raw WAM action or proprioception is provided to the inverse model.
    del data['tokens']
    model = InverseDynamics().cuda()
    optimizer = torch.optim.AdamW([
        dict(params=model.idm.parameters(), lr=1e-4),
        dict(params=model.head.parameters(), lr=3e-4)], weight_decay=0.01)
    dest = OUT/('smoke_stage1' if smoke else 'stage1')
    dest.mkdir(parents=True, exist_ok=True)
    cfg = dict(stage='IDM inverse dynamics pretraining', train=450, val=50, batch=8,
               max_epochs=50, patience=10, dtype='float32', dropout=0.1,
               idm_lr=1e-4, head_lr=3e-4, weight_decay=0.01, clip=1,
               target='two chronological 4x6 normalized demonstrated action segments',
               input='real JEPA current/first future/second future full spatial tokens',
               raw_action_input=False, proprio_input=False, seed=239,
               previous_data=str(PREVIOUS/'dataset.pt'), real_latents=str(REAL))
    (dest/'config.json').write_text(json.dumps(cfg, indent=2))
    prior = y[train].mean(0)
    prior_mse = float((y[val]-prior).square().mean())
    emit('STAGE1_START', parameters=sum(p.numel() for p in model.parameters()),
         constant_mean_action_val_mse=prior_mse)
    @torch.no_grad()
    def evaluate(ids):
        model.eval()
        errors, latent = [], []
        for start in range(0, len(ids), 8):
            ix = ids[start:start+8]
            pred, la = model(z[ix])
            errors.append((pred-y[ix]).square().mean((1, 2)))
            latent.append(la)
        la = torch.cat(latent)
        return dict(mse=float(torch.cat(errors).mean()),
                    la1_la2_cosine=float(F.cosine_similarity(la[:, 0], la[:, 1], dim=1).mean()),
                    la_channel_std=float(la.flatten(1).std(0).mean()))
    def checkpoint(name, epoch, step, best):
        save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=cfg,
                  epoch=epoch, step=step, best_val=best, torch_rng=torch.get_rng_state(),
                  cuda_rng=torch.cuda.get_rng_state_all()), dest/name)
    if smoke:
        ids = train[:8]
        before = evaluate(ids)['mse']
        gradnorms = []
        for step in range(30):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pred, la = model(z[ids])
            loss = F.mse_loss(pred, y[ids])
            loss.backward()
            gradnorms.append(float(torch.stack([p.grad.square().sum() for p in model.idm.parameters()
                                                if p.grad is not None]).sum().sqrt()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
        after = evaluate(ids)['mse']
        assert after < before and max(gradnorms[1:]) > 0
        checkpoint('checkpoint.pt', 0, 30, after)
        summary = dict(passed=True, initial_mse=before, final_mse=after,
                       idm_gradient_norm_max=max(gradnorms))
    else:
        best, best_epoch, stale, step = float('inf'), 0, 0, 0
        started = time.monotonic()
        for epoch in range(50):
            ids = np.random.default_rng(239+epoch).permutation(train).tolist()
            model.train()
            losses = []
            for start in range(0, len(ids), 8):
                ix = ids[start:start+8]
                optimizer.zero_grad(set_to_none=True)
                pred, la = model(z[ix])
                loss = F.mse_loss(pred, y[ix])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss))
                step += 1
            v, tr = evaluate(val), evaluate(train[:64])
            row = dict(epoch=epoch+1, step=step, optimization_mse=float(np.mean(losses)),
                       val=v, train_probe=tr, seconds=time.monotonic()-started)
            with (dest/'metrics.jsonl').open('a') as file:
                file.write(json.dumps(row)+'\n')
            if v['mse'] < best:
                best, best_epoch, stale = v['mse'], epoch+1, 0
                checkpoint('best.pt', epoch+1, step, best)
            else:
                stale += 1
            checkpoint('last.pt', epoch+1, step, best)
            if (epoch+1) % 5 == 0:
                checkpoint(f'epoch{epoch+1:03d}.pt', epoch+1, step, best)
            emit('STAGE1_EPOCH', **row)
            if stale >= 10:
                break
        summary = dict(best_val_mse=best, best_epoch=best_epoch,
                       constant_mean_action_val_mse=prior_mse, final=row,
                       seconds=time.monotonic()-started)
    (dest/'summary.json').write_text(json.dumps(summary, indent=2))
    emit('STAGE1_DONE', smoke=smoke, **summary)


def stage2():
    torch.manual_seed(239)
    data = torch.load(PREVIOUS/'dataset.pt', map_location='cpu', weights_only=False)
    train, val = data['train_ids'], data['val_ids']
    a, z, y = (data[k].cuda() for k in ('action', 'tokens', 'expert'))
    inverse = InverseDynamics().cuda()
    source = torch.load(OUT/'stage1/best.pt', map_location='cpu', weights_only=False)
    inverse.load_state_dict(source['model'])
    inverse.eval().requires_grad_(False)
    latents = []
    with torch.no_grad():
        for start in range(0, 500, 8):
            _, la = inverse(z[start:start+8])
            latents.append(la)
    la = torch.cat(latents).detach()
    save(dict(la=la.cpu(), rows=data['rows'], idm_checkpoint=str(OUT/'stage1/best.pt')),
         OUT/'wam_latent_actions.pt')
    # Validate temporal dependence using real future tokens and training-average LA.
    real_z = torch.stack([torch.load(REAL/f'{i:04d}.pt', map_location='cpu', weights_only=True)
                          for i in val]).cuda()
    real_las, real_predictions, repeated_predictions = [], [], []
    with torch.no_grad():
        for start in range(0, len(val), 8):
            p, r = inverse(real_z[start:start+8])
            repeated, _ = inverse(real_z[start:start+8, :1].expand(-1, 3, -1, -1))
            real_las.append(r)
            real_predictions.append(p)
            repeated_predictions.append(repeated)
        all_real_la = []
        for start in range(0, len(train), 8):
            zz = torch.stack([torch.load(REAL/f'{i:04d}.pt', map_location='cpu', weights_only=True)
                              for i in train[start:start+8]]).cuda()
            _, r = inverse(zz)
            all_real_la.append(r)
        average = torch.cat(all_real_la).mean(0, keepdim=True)
        const_pred = inverse.head(average.expand(len(val), -1, -1)).reshape(-1, 8, 6)
        diagnostic = dict(real_future_action_mse=float(F.mse_loss(torch.cat(real_predictions), y[val])),
                          current_repeated_action_mse=float(F.mse_loss(torch.cat(repeated_predictions), y[val])),
                          mean_la_action_mse=float(F.mse_loss(const_pred, y[val])),
                          real_la1_la2_cosine=float(F.cosine_similarity(torch.cat(real_las)[:, 0],
                                                                       torch.cat(real_las)[:, 1], dim=1).mean()))
    (OUT/'stage1/temporal_diagnostic.json').write_text(json.dumps(diagnostic, indent=2))
    emit('STAGE1_TEMPORAL_DIAGNOSTIC', **diagnostic)
    del inverse, z, real_z, source
    torch.cuda.empty_cache()
    mean = data['action'][train].flatten(1).mean(0)
    std = data['action'][train].flatten(1).std(0).clamp_min(1e-4)
    results = {}
    initial_state = None
    for variant in ('frozen_idm', 'no_jepa'):
        torch.manual_seed(239)
        model = ResidualHead(mean, std).cuda()
        if initial_state is None:
            initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            model.load_state_dict(initial_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        features = la if variant == 'frozen_idm' else torch.zeros_like(la)
        dest = OUT/'stage2'/variant
        dest.mkdir(parents=True, exist_ok=True)
        cfg = dict(stage='frozen pretrained IDM -> residual MLP', variant=variant,
                   train=450, val=50, batch=8, max_epochs=50, patience=10, lr=3e-4,
                   weight_decay=0.01, dtype='float32', frozen_idm=True,
                   source_idm=str(OUT/'stage1/best.pt'), mlp=[312, 512, 512, 48],
                   rms_cap=None, residual_penalty=0, gripper_unchanged=True,
                   matched_mlp_initialization=True, seed=239)
        (dest/'config.json').write_text(json.dumps(cfg, indent=2))
        @torch.no_grad()
        def evaluate(ids, feature_override=None):
            model.eval()
            pred, delta = model(a[ids], features[ids] if feature_override is None else feature_override)
            errors = (pred-y[ids]).square().mean((1, 2))
            base = (a[ids, :, :6]-y[ids]).square().mean((1, 2))
            return dict(mse=float(errors.mean()), baseline_mse=float(base.mean()),
                        better=int((errors<base).sum()), worse=int((errors>base).sum()),
                        correction_rms=float(delta.square().mean((1, 2)).sqrt().mean()))
        def checkpoint(name, epoch, step, best):
            save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=cfg,
                      epoch=epoch, step=step, best_val=best, torch_rng=torch.get_rng_state(),
                      cuda_rng=torch.cuda.get_rng_state_all()), dest/name)
        started = time.monotonic()
        best, best_epoch, stale, step = float('inf'), 0, 0, 0
        baseline = evaluate(val)
        for epoch in range(50):
            ids = np.random.default_rng(239+epoch).permutation(train).tolist()
            model.train()
            losses = []
            for start in range(0, len(ids), 8):
                ix = ids[start:start+8]
                optimizer.zero_grad(set_to_none=True)
                pred, delta = model(a[ix], features[ix])
                loss = F.mse_loss(pred, y[ix])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss))
                step += 1
            v = evaluate(val)
            row = dict(epoch=epoch+1, step=step, optimization_mse=float(np.mean(losses)),
                       val=v, train=evaluate(train), seconds=time.monotonic()-started)
            with (dest/'metrics.jsonl').open('a') as file:
                file.write(json.dumps(row)+'\n')
            if v['mse'] < best:
                best, best_epoch, stale = v['mse'], epoch+1, 0
                checkpoint('best.pt', epoch+1, step, best)
            else:
                stale += 1
            checkpoint('last.pt', epoch+1, step, best)
            if (epoch+1) % 5 == 0:
                checkpoint(f'epoch{epoch+1:03d}.pt', epoch+1, step, best)
            if epoch < 3 or (epoch+1) % 5 == 0:
                emit('STAGE2_EPOCH', variant=variant, **row)
            if stale >= 10:
                break
        ck = torch.load(dest/'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        checks = dict(normal=evaluate(val),
                      train_mean_la=evaluate(val, features[train].mean(0, keepdim=True).expand(len(val), -1, -1)),
                      swapped_la=evaluate(val, features[val].flip(1)))
        results[variant] = dict(best_val_mse=best, best_epoch=best_epoch, baseline=baseline,
                                final=row, best_checks=checks, seconds=time.monotonic()-started)
        emit('STAGE2_VARIANT_DONE', variant=variant, **results[variant])
        del model, optimizer
        torch.cuda.empty_cache()
    (OUT/'summary.json').write_text(json.dumps(dict(stage1_diagnostic=diagnostic, stage2=results), indent=2))
    emit('ALL_COMPLETED', summary=results)


if __name__ == '__main__':
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['smoke', 'stage1', 'stage2'], required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == 'stage2':
        stage2()
    else:
        stage1(args.phase == 'smoke')
