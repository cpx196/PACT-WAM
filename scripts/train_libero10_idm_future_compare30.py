"""Stage 1: compare Predictor/WAM futures and distill frozen real-future IDM LA.

This is not temporal inverse dynamics at the student input: the two token groups
are alternative futures at the SAME time. The existing two learned embeddings
become source embeddings (Predictor, WAM). Stage 2 is deliberately not launched.
"""
import fcntl
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scripts.train_libero10_idm_residual_pilot import IDM
from scripts.train_libero10_idm_two_stage450 import InverseDynamics

ROOT = Path('/data/pxchen/PACT-WAM')
OUT = ROOT/'artifacts/libero10_idm_predictor_wam_compare30_20261007'
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
    assert not (OUT/'summary.json').exists() and not (OUT/'metrics.jsonl').exists()
    (OUT/'pid.txt').write_text(str(os.getpid())+'\n')
    rows = json.loads(MANIFEST.read_text())
    train = [i for i, r in enumerate(rows) if r['split'] == 'train']
    val = [i for i, r in enumerate(rows) if r['split'] == 'val']
    assert (len(rows), len(train), len(val)) == (4307, 3871, 436)
    assert {(rows[i]['task_file'], rows[i]['demo']) for i in train}.isdisjoint(
        {(rows[i]['task_file'], rows[i]['demo']) for i in val})
    data = torch.load(TEACHER_CACHE, map_location='cpu', weights_only=False)
    assert data['rows'] == rows and data['future_source'] == 'real'
    assert data['idm_checkpoint'] == str(SOURCE) and int(data['idm_epoch']) == 29
    target = data['la'].float().clone()
    assert target.shape == (4307, 2, 128) and torch.isfinite(target).all()
    del data
    scale = target[train].flatten(0, 1).std(0)
    scale = scale.clamp_min(float(scale.mean())*.05).clamp_min(1e-4).cuda()
    teacher_mean = target[train].mean(0)
    z = torch.empty(len(rows), 2, 2, 256, 1408)
    real_val = torch.empty(len(val), 2, 2, 256, 1408)
    val_pos = {i: j for j, i in enumerate(val)}
    audits = {}
    audit_ids = {train[0], train[len(train)//2], train[-1], val[0], val[-1]}
    started = time.monotonic()
    for i, row in enumerate(rows):
        item = torch.load(CACHE/f'{i:05d}.pt', map_location='cpu', weights_only=False)
        assert item['index'] == i and item['row'] == row
        p, w, r = item['predicted_future'].float(), item['wam_future'].float(), item['real_future'].float()
        assert p.shape == w.shape == r.shape == (2, 256, 1408)
        assert torch.isfinite(p).all() and torch.isfinite(w).all() and torch.isfinite(r).all()
        # [time, source, space, channel]; compare alternatives at each time.
        z[i].copy_(torch.stack((p, w), 1))
        if i in val_pos:
            real_val[val_pos[i]].copy_(torch.stack((p, r), 1))
        if i in audit_ids:
            audits[i] = torch.cat((item['current'][None], r), 0)
        if (i+1) % 512 == 0 or i+1 == len(rows):
            emit('DATA_LOAD_PROGRESS', loaded=i+1, total=len(rows), seconds=time.monotonic()-started)
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)
    assert source['epoch'] == 29
    teacher = InverseDynamics().cuda().eval().requires_grad_(False)
    teacher.load_state_dict(source['model'])
    with torch.no_grad():
        for i, real in audits.items():
            _, la = teacher(real[None].cuda())
            assert torch.allclose(la[0].cpu(), target[i], atol=2e-5, rtol=2e-5)
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    emit('TEACHER_CACHE_VERIFIED', teacher_epoch=29, samples=len(audits), frozen=True)
    del teacher, audits
    model = IDM().cuda()
    model.load_state_dict({k.removeprefix('idm.'): v for k, v in source['model'].items() if k.startswith('idm.')})
    del source
    config = dict(stage=1, epochs=30, early_stopping=False, batch=8, train=len(train), val=len(val),
                  stride=32, steps_per_epoch=math.ceil(len(train)/8), seed=239, dtype='float32',
                  lr=1e-4, weight_decay=.01, dropout=.1, grad_clip=1.,
                  student_input='two [Predictor future latent, WAM Encoder future latent] pairs at matching times',
                  student_output='two 128D features aligned to teacher LA1 and LA2',
                  teacher_input='current/real future1, real future1/real future2 chronological pairs',
                  target='cached frozen real-future IDM epoch29 LA; no direct action or future-latent reconstruction loss',
                  loss='mean square LA error scaled by fixed train-only teacher channel std',
                  teacher_checkpoint=str(SOURCE), teacher_epoch=29, teacher_cache=str(TEACHER_CACHE),
                  student_initialization=str(SOURCE), parameters=sum(p.numel() for p in model.parameters()),
                  encoder_frozen=True, predictor_frozen=True, teacher_frozen=True,
                  transformer_trained=False, action_decoder_used=False, raw_action_direct_input=False,
                  predictor_action_source='cached original WAM action; Predictor history rolled out for two AC steps',
                  view='main camera, consistent cached rot180', pooling=False,
                  source_embedding_order=['predictor', 'wam'],
                  best_selection='validation scaled LA MSE', manifest=str(MANIFEST), cache=str(CACHE))
    (OUT/'config.json').write_text(json.dumps(config, indent=2))
    save(dict(scale=scale.cpu(), teacher_train_mean=teacher_mean), OUT/'teacher_statistics.pt')

    def forward(features):
        assert features.shape[1:] == (2, 2, 256, 1408)
        return model(features.cuda().flatten(0, 1)).reshape(-1, 2, 128)

    def objective(la, ids):
        return ((la-target[ids].cuda())/scale).square().mean()

    def make_optimizer():
        return torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)

    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = make_optimizer()
    smoke = []
    for step in range(3):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        ids = train[:8]
        loss = objective(forward(z[ids]), ids)
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        assert float(norm) > 0
        optimizer.step()
        smoke.append(dict(step=step+1, loss=float(loss), grad_norm=float(norm)))
    assert not torch.equal(model.output.weight.detach().cpu(), initial['output.weight'])
    (OUT/'smoke.json').write_text(json.dumps(smoke, indent=2))
    emit('SMOKE_PASSED', steps=smoke, peak_gib=torch.cuda.max_memory_allocated()/2**30)
    model.load_state_dict(initial)
    del initial, optimizer
    optimizer = make_optimizer()
    torch.manual_seed(239)
    probe = np.asarray(train)[np.linspace(0, len(train)-1, 256, dtype=int)].tolist()

    @torch.no_grad()
    def evaluate(ids, mode='normal'):
        model.eval()
        latent, raw_error, scaled_error, cos = [], [], [], []
        permutation = np.random.default_rng(239).permutation(val).tolist()
        for start in range(0, len(ids), 8):
            ix = ids[start:start+8]
            zz = real_val[start:start+8] if mode == 'real_alternative' else z[ix]
            if mode != 'normal' and mode != 'real_alternative':
                zz = zz.clone()
                if mode == 'same_wam':
                    zz[:, :, 0] = zz[:, :, 1]
                elif mode == 'same_predictor':
                    zz[:, :, 1] = zz[:, :, 0]
                elif mode == 'shuffle_wam':
                    zz[:, :, 1] = z[permutation[start:start+len(ix)], :, 1]
                else:
                    raise ValueError(mode)
            la = forward(zz)
            truth = target[ix].cuda()
            raw_error.append((la-truth).square().mean((1, 2)).cpu())
            scaled_error.append(((la-truth)/scale).square().mean((1, 2)).cpu())
            cos.append(F.cosine_similarity(la, truth, dim=-1).cpu())
            latent.append(la.cpu())
        la = torch.cat(latent)
        return dict(la_mse=float(torch.cat(raw_error).mean()),
                    scaled_la_mse=float(torch.cat(scaled_error).mean()),
                    teacher_cosine=float(torch.cat(cos).mean()),
                    teacher_cosine_per_time=torch.cat(cos).mean(0).tolist(),
                    la_channel_std=float(la.flatten(1).std(0).mean()),
                    teacher_channel_std=float(target[ids].flatten(1).std(0).mean()),
                    la1_la2_cosine=float(F.cosine_similarity(la[:, 0], la[:, 1], dim=1).mean()))

    initial_eval = dict(val=evaluate(val), train_probe=evaluate(probe),
                        real_alternative=evaluate(val, 'real_alternative'),
                        same_wam=evaluate(val, 'same_wam'), same_predictor=evaluate(val, 'same_predictor'),
                        shuffle_wam=evaluate(val, 'shuffle_wam'),
                        constant_teacher_mean_scaled_mse=float(((teacher_mean-target[val])/scale.cpu()).square().mean()))
    (OUT/'initial_evaluation.json').write_text(json.dumps(initial_eval, indent=2))
    emit('INITIAL_EVALUATION', **initial_eval)
    best, best_epoch, global_step = float('inf'), 0, 0
    started = time.monotonic()

    def checkpoint(name, epoch):
        save(dict(idm=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                  epoch=epoch, step=global_step, best_val=best, best_epoch=best_epoch,
                  la_scale=scale.cpu(), torch_rng=torch.get_rng_state(),
                  cuda_rng=torch.cuda.get_rng_state_all()), OUT/name)

    for epoch in range(1, 31):
        ids = np.random.default_rng(238+epoch).permutation(train).tolist()
        model.train()
        total, count, clipped, batches = 0., 0, 0, 0
        epoch_started = time.monotonic()
        for start in range(0, len(ids), 8):
            ix = ids[start:start+8]
            optimizer.zero_grad(set_to_none=True)
            loss = objective(forward(z[ix]), ix)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            clipped += int(float(norm) > 1)
            optimizer.step()
            total += float(loss)*len(ix)
            count += len(ix)
            batches += 1
            global_step += 1
            if batches % 100 == 0:
                emit('TRAIN_PROGRESS', epoch=epoch, batch=batches, step=global_step, loss=total/count,
                     seconds_per_batch=(time.monotonic()-epoch_started)/batches)
        row = dict(epoch=epoch, step=global_step, optimization_loss=total/count,
                   val=evaluate(val), train_probe=evaluate(probe),
                   clip_rate=clipped/batches, peak_gib=torch.cuda.max_memory_allocated()/2**30)
        if epoch == 1 or epoch % 5 == 0:
            row.update(real_alternative=evaluate(val, 'real_alternative'),
                       same_wam=evaluate(val, 'same_wam'), same_predictor=evaluate(val, 'same_predictor'),
                       shuffle_wam=evaluate(val, 'shuffle_wam'))
        if row['val']['scaled_la_mse'] < best:
            best, best_epoch = row['val']['scaled_la_mse'], epoch
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
    emit('ALL_COMPLETED', best_epoch=best_epoch, best_val_scaled_la_mse=best, seconds=summary['seconds'])


if __name__ == '__main__':
    main()
