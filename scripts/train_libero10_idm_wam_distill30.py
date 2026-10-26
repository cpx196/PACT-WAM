"""Fixed-budget WAM-to-real inverse-dynamics latent distillation (no early stop)."""
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
OUT = ROOT/'artifacts/libero10_idm_wam_teacher_distill30_20261007'
CACHE = ROOT/'artifacts/libero10_idm_cache_stride32_20261004/tokens'
MANIFEST = ROOT/'artifacts/libero10_idm_dense_windows_20261004/windows_stride32.json'
SOURCE = ROOT/'artifacts/libero10_idm_real_stride32_30epoch_20261004/best.pt'
TEACHER_CACHE = ROOT/'artifacts/libero10_transformer_real_oracle_idm29_20261007/dense_features.pt'


def emit(event, **kw):
    print(json.dumps(dict(time=time.strftime('%F %T'), event=event, **kw)), flush=True)


def save(value, path):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(239)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'launcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT/'summary.json').exists(), 'Completed run already exists'
    assert not (OUT/'metrics.jsonl').exists(), 'Existing training: use a new directory or explicit resume'
    rows = json.loads(MANIFEST.read_text())
    train = [i for i, row in enumerate(rows) if row['split'] == 'train']
    val = [i for i, row in enumerate(rows) if row['split'] == 'val']
    assert (len(rows), len(train), len(val)) == (4307, 3871, 436)
    assert {(rows[i]['task_file'], rows[i]['demo']) for i in train}.isdisjoint(
        {(rows[i]['task_file'], rows[i]['demo']) for i in val})
    cached = torch.load(TEACHER_CACHE, map_location='cpu', weights_only=False)
    assert cached['rows'] == rows and cached['future_source'] == 'real'
    assert cached['idm_checkpoint'] == str(SOURCE) and int(cached['idm_epoch']) == 29
    target_la = cached['la'].float().clone()
    y = cached['expert'].float().clone()
    assert target_la.shape == (4307, 2, 128) and y.shape == (4307, 8, 6)
    assert torch.isfinite(target_la).all() and torch.isfinite(y).all()
    del cached
    # Fixed train-only channel scaling preserves the teacher's LA coordinates and magnitude.
    scale = target_la[train].flatten(0, 1).std(0)
    scale = scale.clamp_min(float(scale.mean())*.05).clamp_min(1e-4).cuda()
    z = torch.empty(len(rows), 3, 256, 1408)
    real_val = torch.empty(len(val), 3, 256, 1408)
    val_positions = {i: j for j, i in enumerate(val)}
    audit_ids = [train[0], train[len(train)//2], train[-1], val[0], val[-1]]
    real_audit = []
    started = time.monotonic()
    for i, row in enumerate(rows):
        item = torch.load(CACHE/f'{i:05d}.pt', map_location='cpu', weights_only=False)
        assert item['index'] == i and item['row'] == row
        wam = torch.cat((item['current'][None], item['wam_future']), 0).float()
        assert wam.shape == (3, 256, 1408) and torch.isfinite(wam).all()
        expected = (item['expert_executable']*item['scale']+item['offset'])[:, :6]
        assert torch.allclose(y[i], expected, atol=1e-6, rtol=1e-6)
        z[i].copy_(wam)
        if i in val_positions:
            real_val[val_positions[i]].copy_(torch.cat((item['current'][None], item['real_future']), 0))
        if i in audit_ids:
            real_audit.append((i, torch.cat((item['current'][None], item['real_future']), 0)))
        if (i+1) % 512 == 0 or i+1 == len(rows):
            emit('DATA_LOAD_PROGRESS', loaded=i+1, total=len(rows), seconds=time.monotonic()-started)
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)
    assert source['epoch'] == 29
    teacher = InverseDynamics().cuda().eval().requires_grad_(False)
    teacher.load_state_dict(source['model'])
    with torch.no_grad():
        for i, real in real_audit:
            _, la = teacher(real[None].cuda())
            assert torch.allclose(la[0].cpu(), target_la[i], atol=2e-5, rtol=2e-5)
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    emit('TEACHER_CACHE_VERIFIED', epoch=29, checked=len(real_audit), teacher_frozen=True)
    del teacher, real_audit
    model = InverseDynamics().cuda()
    model.load_state_dict(source['model'])
    del source
    config = dict(epochs=30, early_stopping=False, train=len(train), val=len(val), stride=32,
                  batch=8, steps_per_epoch=math.ceil(len(train)/8), seed=239, dtype='float32',
                  idm_lr=1e-4, head_lr=3e-4, dropout=.1, weight_decay=.01, grad_clip=1.,
                  teacher_checkpoint=str(SOURCE), teacher_epoch=29, teacher_frozen=True,
                  teacher_cache=str(TEACHER_CACHE), student_initialization=str(SOURCE),
                  encoder_frozen=True, predictor_not_used=True, transformer_not_trained=True,
                  input='current + two WAM future clips, full 256x1408 spatial tokens; cached rot180 main view',
                  raw_action_input=False, target='real-future frozen teacher LA1/LA2 + demonstrated 8x6 actions',
                  loss='MSE((student_LA-teacher_LA)/fixed_train_channel_std) + action_MSE',
                  la_weight=1., action_weight=1., best_selection='validation scaled LA MSE',
                  idm_parameters=sum(p.numel() for p in model.idm.parameters()),
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  manifest=str(MANIFEST), token_cache=str(CACHE),
                  real_future='teacher targets and diagnostics only; never student training input')
    (OUT/'config.json').write_text(json.dumps(config, indent=2))
    save(dict(scale=scale.cpu(), teacher_train_mean=target_la[train].mean(0)), OUT/'teacher_statistics.pt')

    def make_optimizer():
        return torch.optim.AdamW([dict(params=model.idm.parameters(), lr=1e-4),
                                  dict(params=model.head.parameters(), lr=3e-4)], weight_decay=.01)

    def losses(pred, la, ids):
        latent = ((la-target_la[ids].cuda())/scale).square().mean()
        action = F.mse_loss(pred, y[ids].cuda())
        return latent+action, latent, action

    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = make_optimizer()
    smoke = []
    for step in range(3):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        ids = train[:8]
        pred, la = model(z[ids].cuda())
        total, latent, action = losses(pred, la, ids)
        assert torch.isfinite(total)
        total.backward()
        grad = torch.stack([p.grad.square().sum() for p in model.idm.parameters() if p.grad is not None]).sum().sqrt()
        assert torch.isfinite(grad) and float(grad) > 0
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        smoke.append(dict(step=step+1, total=float(total), la=float(latent), action=float(action),
                          idm_grad_norm=float(grad), grad_norm=float(norm)))
    assert not torch.equal(model.idm.output.weight.detach().cpu(), initial['idm.output.weight'])
    (OUT/'smoke.json').write_text(json.dumps(smoke, indent=2))
    emit('SMOKE_PASSED', steps=smoke, peak_gib=torch.cuda.max_memory_allocated()/2**30)
    model.load_state_dict(initial)
    del initial, optimizer
    optimizer = make_optimizer()
    torch.manual_seed(239)
    probe = np.asarray(train)[np.linspace(0, len(train)-1, 256, dtype=int)].tolist()

    @torch.no_grad()
    def evaluate(ids, source='wam'):
        model.eval()
        raw_errors, scaled_errors, actions, cosines, latent = [], [], [], [], []
        shuffled = np.random.default_rng(239).permutation(val).tolist()
        for start in range(0, len(ids), 8):
            ix = ids[start:start+8]
            zz = real_val[start:start+8] if source == 'real' else z[ix]
            if source == 'repeated':
                zz = zz[:, :1].expand(-1, 3, -1, -1)
            if source == 'shuffled_future':
                zz = zz.clone()
                zz[:, 1:] = z[shuffled[start:start+len(ix)], 1:]
            pred, la = model(zz.cuda())
            target = target_la[ix].cuda()
            raw_errors.append((la-target).square().mean((1, 2)).cpu())
            scaled_errors.append(((la-target)/scale).square().mean((1, 2)).cpu())
            actions.append((pred-y[ix].cuda()).square().mean((1, 2)).cpu())
            cosines.append(F.cosine_similarity(la, target, dim=-1).cpu())
            latent.append(la.cpu())
        la = torch.cat(latent)
        return dict(la_mse=float(torch.cat(raw_errors).mean()),
                    scaled_la_mse=float(torch.cat(scaled_errors).mean()),
                    action_mse=float(torch.cat(actions).mean()),
                    teacher_cosine=float(torch.cat(cosines).mean()),
                    teacher_cosine_per_time=torch.cat(cosines).mean(0).tolist(),
                    la_channel_std=float(la.flatten(1).std(0).mean()),
                    teacher_channel_std=float(target_la[ids].flatten(1).std(0).mean()),
                    la1_la2_cosine=float(F.cosine_similarity(la[:, 0], la[:, 1], dim=1).mean()))

    initial_eval = dict(wam_val=evaluate(val), real_val=evaluate(val, 'real'),
                        train_probe=evaluate(probe), repeated_current=evaluate(val, 'repeated'),
                        shuffled_future=evaluate(val, 'shuffled_future'))
    (OUT/'initial_evaluation.json').write_text(json.dumps(initial_eval, indent=2))
    emit('INITIAL_EVALUATION', **initial_eval)
    best, best_epoch, global_step = float('inf'), 0, 0
    started = time.monotonic()

    def checkpoint(name, epoch):
        save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                  epoch=epoch, step=global_step, best_val=best, best_epoch=best_epoch,
                  la_scale=scale.cpu(), torch_rng=torch.get_rng_state(),
                  cuda_rng=torch.cuda.get_rng_state_all()), OUT/name)

    for epoch in range(1, 31):
        ids = np.random.default_rng(238+epoch).permutation(train).tolist()
        model.train()
        weighted = np.zeros(3)
        count = clipped = batches = 0
        epoch_started = time.monotonic()
        for start in range(0, len(ids), 8):
            ix = ids[start:start+8]
            optimizer.zero_grad(set_to_none=True)
            pred, la = model(z[ix].cuda())
            total, latent, action = losses(pred, la, ix)
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            clipped += int(float(norm) > 1)
            optimizer.step()
            weighted += np.asarray([float(total), float(latent), float(action)])*len(ix)
            count += len(ix)
            batches += 1
            global_step += 1
            if batches % 100 == 0:
                emit('TRAIN_PROGRESS', epoch=epoch, batch=batches, step=global_step,
                     total=weighted[0]/count, la=weighted[1]/count, action=weighted[2]/count,
                     seconds_per_batch=(time.monotonic()-epoch_started)/batches)
        row = dict(epoch=epoch, step=global_step, optimization_total=weighted[0]/count,
                   optimization_la=weighted[1]/count, optimization_action=weighted[2]/count,
                   wam_val=evaluate(val), real_val=evaluate(val, 'real'), train_probe=evaluate(probe),
                   clip_rate=clipped/batches, peak_gib=torch.cuda.max_memory_allocated()/2**30)
        if epoch % 5 == 0 or epoch == 1:
            row.update(repeated_current=evaluate(val, 'repeated'),
                       shuffled_future=evaluate(val, 'shuffled_future'))
        if row['wam_val']['scaled_la_mse'] < best:
            best, best_epoch = row['wam_val']['scaled_la_mse'], epoch
            checkpoint('best.pt', epoch)
        checkpoint('last.pt', epoch)
        if epoch % 5 == 0:
            checkpoint(f'epoch{epoch:03d}.pt', epoch)
        row.update(epoch_seconds=time.monotonic()-epoch_started, seconds=time.monotonic()-started)
        with (OUT/'metrics.jsonl').open('a') as file:
            file.write(json.dumps(row)+'\n')
        emit('EPOCH_COMPLETED', **row)
    summary = dict(config=config, initial=initial_eval, best_val_scaled_la_mse=best,
                   best_epoch=best_epoch, final=row, seconds=time.monotonic()-started)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2))
    emit('ALL_COMPLETED', best_epoch=best_epoch, best_val_scaled_la_mse=best,
         seconds=summary['seconds'])


if __name__ == '__main__':
    main()
