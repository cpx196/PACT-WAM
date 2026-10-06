"""Warm-start frozen-feature inverse dynamics on all stride-32 real futures."""
import fcntl
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scripts.train_libero10_idm_two_stage450 import InverseDynamics

ROOT = Path('/data/pxchen/PACT-WAM')
OUT = ROOT/'artifacts/libero10_idm_real_stride32_30epoch_20261004'
CACHE = ROOT/'artifacts/libero10_idm_cache_stride32_20261004/tokens'
MANIFEST = ROOT/'artifacts/libero10_idm_dense_windows_20261004/windows_stride32.json'
SOURCE = ROOT/'artifacts/libero10_idm_pretrain_then_mlp450_20261004/stage1/best.pt'


def emit(event, **kw):
    print(json.dumps(dict(time=time.strftime('%F %T'), event=event, **kw)), flush=True)


def save(value, path):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def load_data():
    rows = json.loads(MANIFEST.read_text())
    train = [i for i, row in enumerate(rows) if row['split'] == 'train']
    val = [i for i, row in enumerate(rows) if row['split'] == 'val']
    assert (len(rows), len(train), len(val)) == (4307, 3871, 436)
    assert {(rows[i]['task_file'], rows[i]['demo']) for i in train}.isdisjoint(
        {(rows[i]['task_file'], rows[i]['demo']) for i in val})
    # Preallocate on CPU. Only a minibatch is transferred to GPU at a time.
    z = torch.empty(len(rows), 3, 256, 1408)
    y = torch.empty(len(rows), 8, 6)
    wam_val = torch.empty(len(val), 3, 256, 1408)
    val_positions = {index: pos for pos, index in enumerate(val)}
    started = time.monotonic()
    for i, row in enumerate(rows):
        item = torch.load(CACHE/f'{i:05d}.pt', map_location='cpu', weights_only=False)
        assert item['index'] == i and item['row'] == row
        real = torch.cat((item['current'][None], item['real_future']), 0)
        assert real.shape == (3, 256, 1408) and torch.isfinite(real).all()
        target = item['expert_normalized_first6'].float()
        expected = (item['expert_executable']*item['scale']+item['offset'])[:, :6]
        assert target.shape == (8, 6) and torch.isfinite(target).all()
        assert torch.allclose(target, expected, atol=1e-6, rtol=1e-6)
        z[i].copy_(real)
        y[i].copy_(target)
        if i in val_positions:
            wam = torch.cat((item['current'][None], item['wam_future']), 0)
            assert torch.isfinite(wam).all()
            wam_val[val_positions[i]].copy_(wam)
        if (i+1) % 512 == 0 or i+1 == len(rows):
            emit('DATA_LOAD_PROGRESS', loaded=i+1, total=len(rows), seconds=time.monotonic()-started)
    return rows, train, val, z, y, wam_val


def main():
    torch.set_num_threads(4)
    torch.manual_seed(239)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'launcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT/'summary.json').exists(), 'Completed run already exists'
    rows, train, val, z, y, wam_val = load_data()
    model = InverseDynamics().cuda()
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)
    model.load_state_dict(source['model'])
    del source
    config = dict(train=len(train), val=len(val), stride=32, epochs=30,
                  early_stopping=False, batch=8, steps_per_epoch=math.ceil(len(train)/8),
                  seed=239, dtype='float32', dropout=.1, idm_lr=1e-4, head_lr=3e-4,
                  weight_decay=.01, grad_clip=1, warm_start=str(SOURCE),
                  optimizer='fresh AdamW for expanded dataset',
                  encoder_frozen=True, predictor_not_used=True, raw_action_input=False,
                  training_input='cached real JEPA current and two future clips; all spatial tokens',
                  target='two 4-step normalized demonstrated 6D action segments; no gripper',
                  wam_future='validation diagnostic only; never used for training',
                  manifest=str(MANIFEST), token_cache=str(CACHE),
                  idm_parameters=sum(p.numel() for p in model.idm.parameters()),
                  total_parameters=sum(p.numel() for p in model.parameters()))
    (OUT/'config.json').write_text(json.dumps(config, indent=2))
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    def optimizer_for_model():
        return torch.optim.AdamW([dict(params=model.idm.parameters(), lr=1e-4),
                                  dict(params=model.head.parameters(), lr=3e-4)], weight_decay=.01)
    optimizer = optimizer_for_model()
    smoke_rows = []
    for step in range(3):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        ids = train[:8]
        pred, la = model(z[ids].cuda())
        loss = F.mse_loss(pred, y[ids].cuda())
        assert torch.isfinite(loss)
        loss.backward()
        idm_grad = torch.stack([p.grad.square().sum() for p in model.idm.parameters()
                                if p.grad is not None]).sum().sqrt()
        assert torch.isfinite(idm_grad) and float(idm_grad) > 0
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        smoke_rows.append(dict(step=step+1, loss=float(loss), idm_grad_norm=float(idm_grad),
                               grad_norm=float(norm)))
    assert not torch.equal(model.idm.output.weight.detach().cpu(), initial['idm.output.weight'])
    (OUT/'smoke.json').write_text(json.dumps(smoke_rows, indent=2))
    emit('SMOKE_PASSED', gradients=smoke_rows, peak_gib=torch.cuda.max_memory_allocated()/2**30)
    model.load_state_dict(initial)
    del initial, optimizer
    optimizer = optimizer_for_model()
    torch.manual_seed(239)
    probe = np.asarray(train)[np.linspace(0, len(train)-1, 256, dtype=int)].tolist()

    @torch.no_grad()
    def evaluate(ids, features=None, repeated=False):
        model.eval()
        errors, latent = [], []
        for start in range(0, len(ids), 8):
            subset = ids[start:start+8]
            zz = z[subset] if features is None else features[start:start+8]
            if repeated:
                zz = zz[:, :1].expand(-1, 3, -1, -1)
            pred, la = model(zz.cuda())
            errors.append((pred-y[subset].cuda()).square().mean((1, 2)).cpu())
            latent.append(la.cpu())
        la = torch.cat(latent)
        return dict(mse=float(torch.cat(errors).mean()),
                    la1_la2_cosine=float(F.cosine_similarity(la[:, 0], la[:, 1], dim=1).mean()),
                    la_channel_std=float(la.flatten(1).std(0).mean()))

    initial_eval = dict(real_val=evaluate(val), wam_val=evaluate(val, wam_val),
                        repeated_current=evaluate(val, repeated=True), train_probe=evaluate(probe))
    emit('INITIAL_EVALUATION', **initial_eval)
    (OUT/'initial_evaluation.json').write_text(json.dumps(initial_eval, indent=2))
    best, best_epoch, step = float('inf'), 0, 0
    started = time.monotonic()
    def checkpoint(name, epoch):
        save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                  config=config, epoch=epoch, step=step, best_val=best, best_epoch=best_epoch,
                  torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), OUT/name)
    for epoch in range(30):
        ids = np.random.default_rng(239+epoch).permutation(train).tolist()
        model.train()
        losses, count, weighted, clipped = [], 0, 0., 0
        epoch_started = time.monotonic()
        for start in range(0, len(ids), 8):
            ix = ids[start:start+8]
            optimizer.zero_grad(set_to_none=True)
            pred, la = model(z[ix].cuda())
            loss = F.mse_loss(pred, y[ix].cuda())
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            clipped += int(float(norm)>1)
            optimizer.step()
            weighted += float(loss)*len(ix)
            count += len(ix)
            losses.append(float(loss))
            step += 1
            if len(losses) % 100 == 0:
                emit('TRAIN_PROGRESS', epoch=epoch+1, batch=len(losses), step=step,
                     mse=weighted/count, seconds=time.monotonic()-epoch_started,
                     seconds_per_batch=(time.monotonic()-epoch_started)/len(losses))
        row = dict(epoch=epoch+1, step=step, optimization_mse=weighted/count,
                   real_val=evaluate(val), wam_val=evaluate(val, wam_val),
                   repeated_current=evaluate(val, repeated=True), train_probe=evaluate(probe),
                   clip_rate=clipped/len(losses), peak_gib=torch.cuda.max_memory_allocated()/2**30,
                   epoch_seconds=time.monotonic()-epoch_started, seconds=time.monotonic()-started)
        with (OUT/'metrics.jsonl').open('a') as file:
            file.write(json.dumps(row)+'\n')
        if row['real_val']['mse'] < best:
            best, best_epoch = row['real_val']['mse'], epoch+1
            checkpoint('best.pt', epoch+1)
        checkpoint('last.pt', epoch+1)
        if (epoch+1) % 5 == 0:
            checkpoint(f'epoch{epoch+1:03d}.pt', epoch+1)
        emit('EPOCH_COMPLETED', **row)
    summary = dict(config=config, initial=initial_eval, best_val_mse=best,
                   best_epoch=best_epoch, final=row, seconds=time.monotonic()-started)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2))
    emit('ALL_COMPLETED', best_val_mse=best, best_epoch=best_epoch, seconds=summary['seconds'])


if __name__ == '__main__':
    main()
