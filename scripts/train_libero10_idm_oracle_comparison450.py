"""Matched MLP oracle-real/WAM/no-JEPA experiment; frozen identical pretrained IDM."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scripts.train_libero10_idm_two_stage450 import InverseDynamics, ResidualHead, REAL

ROOT = Path('/data/pxchen/PACT-WAM')
SOURCE = ROOT/'artifacts/libero10_idm_pretrain_then_mlp450_20261004'
DATA = ROOT/'artifacts/libero10_idm21m_mlp_pilot450_20261004/dataset.pt'
OUT = ROOT/'artifacts/libero10_idm_oracle_comparison450_20261004'


def emit(event, **kw):
    print(json.dumps(dict(time=time.strftime('%F %T'), event=event, **kw)), flush=True)


def save(value, path):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    data = torch.load(DATA, map_location='cpu', weights_only=False)
    train, val = data['train_ids'], data['val_ids']
    assert len(train) == 450 and len(val) == 50
    a, y = data['action'].cuda(), data['expert'].cuda()
    wam = torch.load(SOURCE/'wam_latent_actions.pt', map_location='cpu', weights_only=False)
    assert wam['rows'] == data['rows']
    model = InverseDynamics().cuda()
    ck = torch.load(SOURCE/'stage1/best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval().requires_grad_(False)
    real_la, real_predictions = [], []
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, 500, 8):
            tokens = []
            for i in range(start, min(start+8, 500)):
                z = torch.load(REAL/f'{i:04d}.pt', map_location='cpu', weights_only=True)
                assert torch.allclose(z[0], data['tokens'][i, 0], atol=2e-5, rtol=2e-5)
                tokens.append(z)
            pred, la = model(torch.stack(tokens).cuda())
            real_la.append(la)
            real_predictions.append(pred)
        real_la = torch.cat(real_la)
        wam_la = wam['la'].cuda()
        # Recheck the cached WAM LA against the exact same frozen IDM.
        _, check = model(data['tokens'][:8].cuda())
        assert torch.allclose(check, wam_la[:8], atol=2e-5, rtol=2e-5)
        real_pred = torch.cat(real_predictions)
        wam_pred = model.head(wam_la).reshape(-1, 8, 6)
        inverse_diagnostic = dict(real_future_val_mse=float(F.mse_loss(real_pred[val], y[val])),
                                  wam_future_val_mse=float(F.mse_loss(wam_pred[val], y[val])),
                                  real_wam_la_cosine=float(F.cosine_similarity(
                                      real_la[val].flatten(1), wam_la[val].flatten(1), dim=1).mean()))
    save(dict(real_la=real_la.cpu(), wam_la=wam_la.cpu(), rows=data['rows'],
              idm_checkpoint=str(SOURCE/'stage1/best.pt')), OUT/'latent_actions.pt')
    emit('LA_CACHE_READY', seconds=time.monotonic()-started, inverse_diagnostic=inverse_diagnostic)
    del model, ck, data['tokens']
    torch.cuda.empty_cache()
    features = dict(real_oracle=real_la, wam_future=wam_la, no_jepa=torch.zeros_like(wam_la))
    mean = data['action'][train].flatten(1).mean(0)
    std = data['action'][train].flatten(1).std(0).clamp_min(1e-4)
    config = dict(train=450, val=50, batch=8, max_epochs=50, patience=10, seed=239,
                  lr=3e-4, weight_decay=0.01, dropout=0.1, grad_clip=1,
                  mlp=[312, 512, 512, 48], dtype='float32', frozen_idm=True,
                  rms_cap=None, residual_penalty=0, gripper_unchanged=True,
                  oracle_warning='real future is unavailable online; diagnostic upper-bound only',
                  source_idm=str(SOURCE/'stage1/best.pt'), source_data=str(DATA),
                  matched_initialization=True, matched_batch_order=True)
    (OUT/'config.json').write_text(json.dumps(config, indent=2))
    results = {}
    initial_state = None
    for variant, feat in features.items():
        torch.manual_seed(239)
        model = ResidualHead(mean, std).cuda()
        if initial_state is None:
            initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            model.load_state_dict(initial_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        dest = OUT/variant
        dest.mkdir(exist_ok=True)
        @torch.no_grad()
        def evaluate(ids, f=None):
            model.eval()
            pred, delta = model(a[ids], feat[ids] if f is None else f)
            error = (pred-y[ids]).square().mean((1, 2))
            baseline = (a[ids, :, :6]-y[ids]).square().mean((1, 2))
            return dict(mse=float(error.mean()), baseline_mse=float(baseline.mean()),
                        better=int((error<baseline).sum()), worse=int((error>baseline).sum()),
                        correction_rms=float(delta.square().mean((1, 2)).sqrt().mean()),
                        correction_cosine=float(F.cosine_similarity(delta.flatten(1),
                            (y[ids]-a[ids, :, :6]).flatten(1), dim=1).mean()))
        def checkpoint(name, epoch, step, best):
            save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                      variant=variant, epoch=epoch, step=step, best_val=best,
                      torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), dest/name)
        best, best_epoch, stale, step = float('inf'), 0, 0, 0
        baseline = evaluate(val)
        started = time.monotonic()
        for epoch in range(50):
            ids = np.random.default_rng(239+epoch).permutation(train).tolist()
            model.train()
            losses = []
            for start in range(0, len(ids), 8):
                ix = ids[start:start+8]
                optimizer.zero_grad(set_to_none=True)
                pred, delta = model(a[ix], feat[ix])
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
            if epoch < 3 or (epoch+1) % 10 == 0:
                emit('EPOCH_COMPLETED', variant=variant, **row)
            if stale >= 10:
                break
        ck = torch.load(dest/'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        checks = {name: evaluate(val, value[val]) for name, value in features.items()}
        checks['train_mean_la'] = evaluate(val, feat[train].mean(0, keepdim=True).expand(len(val), -1, -1))
        checks['swapped_la'] = evaluate(val, feat[val].flip(1))
        results[variant] = dict(best_val_mse=best, best_epoch=best_epoch, baseline=baseline,
                                final=row, cross_domain_validation=checks,
                                seconds=time.monotonic()-started)
        emit('VARIANT_COMPLETED', variant=variant, **results[variant])
        del model, optimizer
        torch.cuda.empty_cache()
    summary = dict(inverse_diagnostic=inverse_diagnostic, variants=results, config=config)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2))
    emit('ALL_COMPLETED', inverse_diagnostic=inverse_diagnostic,
         best_mse={k: v['best_val_mse'] for k, v in results.items()})


if __name__ == '__main__':
    main()
