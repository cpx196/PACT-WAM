"""LaWAM-inspired future correction: comparison IDM + AdaLN decoder, 10 epochs.

Both trainable networks start fresh. Real future is a reconstruction label,
never an IDM input. This is an adaptation, not a reproduction of LaWAM.
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
from scripts.train_libero10_idm_residual_pilot import IDM, spatial_position

ROOT = Path('/data/pxchen/PACT-WAM')
OUT = ROOT/'artifacts/libero10_future_correction_reconstruct10_20261007'
CACHE = ROOT/'artifacts/libero10_idm_cache_stride32_20261004/tokens'
MANIFEST = ROOT/'artifacts/libero10_idm_dense_windows_20261004/windows_stride32.json'


def emit(event, **kw):
    print(json.dumps(dict(time=time.strftime('%F %T'), event=event, **kw)), flush=True)


def save(value, path):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


class AdaLNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(512, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(512, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(512, 8, dropout=.1, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(512, 1024), nn.GELU(), nn.Dropout(.1),
                                nn.Linear(1024, 512), nn.Dropout(.1))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(512, 6*512))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, c):
        s1, a1, g1, s2, a2, g2 = self.modulation(c).chunk(6, -1)
        q = self.norm1(x)*(1+a1[:, None])+s1[:, None]
        x = x+g1[:, None]*self.attn(q, q, q, need_weights=False)[0]
        q = self.norm2(x)*(1+a2[:, None])+s2[:, None]
        return x+g2[:, None]*self.ff(q)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(1408, 512)
        self.code_project = nn.Linear(128, 512)
        self.register_buffer('spatial', spatial_position())
        self.blocks = nn.ModuleList([AdaLNBlock() for _ in range(6)])
        self.output = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 1408))

    def forward(self, predictor, code):
        x = self.project(predictor)+self.spatial[None]
        c = self.code_project(code)
        for block in self.blocks:
            x = block(x, c)
        # Residual latent correction, then restore the cached patch-LN convention.
        return F.layer_norm(predictor+self.output(x), (1408,))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.idm = IDM()
        self.posterior = nn.Linear(128, 256)
        self.decoder = Decoder()

    def encode(self, pairs):
        mu, logvar = self.posterior(self.idm(pairs)).chunk(2, -1)
        return mu, logvar.clamp(-10, 5)

    def forward(self, pairs):
        mu, logvar = self.encode(pairs)
        code = mu+torch.randn_like(mu)*torch.exp(.5*logvar) if self.training else mu
        reconstruction = self.decoder(pairs[:, 0], code)
        kl = .5*(mu.square()+logvar.exp()-1-logvar).mean()
        return reconstruction, mu, kl


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--epochs', type=int, default=10, help='Total epoch number including resumed epochs')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    OUT = args.out
    torch.set_num_threads(4)
    torch.manual_seed(239)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'launcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT/'metrics.jsonl').exists() and not (OUT/'summary.json').exists()
    (OUT/'pid.txt').write_text(str(os.getpid())+'\n')
    rows = json.loads(MANIFEST.read_text())
    train = [i for i, row in enumerate(rows) if row['split'] == 'train']
    val = [i for i, row in enumerate(rows) if row['split'] == 'val']
    assert (len(rows), len(train), len(val)) == (4307, 3871, 436)
    assert {(rows[i]['task_file'], rows[i]['demo']) for i in train}.isdisjoint(
        {(rows[i]['task_file'], rows[i]['demo']) for i in val})
    # CPU cache: [window, future_time, source(Predictor/WAM/real label), patch, channel].
    z = torch.empty(len(rows), 2, 3, 256, 1408)
    started = time.monotonic()
    for i, row in enumerate(rows):
        item = torch.load(CACHE/f'{i:05d}.pt', map_location='cpu', weights_only=False)
        assert item['row'] == row and item['index'] == i
        for j, key in enumerate(['predicted_future', 'wam_future', 'real_future']):
            feature = item[key].float()
            assert feature.shape == (2, 256, 1408) and torch.isfinite(feature).all()
            z[i, :, j].copy_(feature)
        if (i+1) % 512 == 0 or i+1 == len(rows):
            emit('DATA_LOAD_PROGRESS', loaded=i+1, total=len(rows), seconds=time.monotonic()-started)
    model = Model().cuda()
    config = dict(epochs=args.epochs, early_stopping=False, train=len(train), val=len(val), batch=8,
                  stride=32, steps_per_epoch=math.ceil(len(train)/8), seed=239, dtype='float32',
                  lr=1e-4, weight_decay=.01, grad_clip=1., dropout=.1, beta_kl=1e-5,
                  idm_layers=6, idm_hidden=512, idm_heads=8, idm_ffn=2048,
                  decoder_layers=6, decoder_hidden=512, decoder_heads=8, decoder_ffn=1024,
                  code_dim=128, posterior='diagonal Gaussian; train sample, eval mean',
                  idm_parameters=sum(p.numel() for p in model.idm.parameters()),
                  decoder_parameters=sum(p.numel() for p in model.decoder.parameters()),
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  initialization='fresh random IDM, posterior and Decoder; only Decoder AdaLN modulation zero',
                  pretrained_trainable_weights=None, teacher_la_used=False, eef_auxiliary_loss=False,
                  training_input='matching-time Predictor/WAM future pairs, full 256x1408 patches',
                  decoder_input='Predictor future + comparison code', target='real future latent, labels only',
                  loss='full patch future MSE + 1e-5 mean per-dimension KL',
                  residual_decoder=True, output_patch_layer_norm=True,
                  encoder_frozen=True, predictor_frozen=True, wam_frozen=True, arb_trained=False,
                  view='cached main camera consistent rot180', source_order=['predictor', 'wam'],
                  caveat='LaWAM-inspired adapted objective; no EEF auxiliary head or raw-image augmentation',
                  manifest=str(MANIFEST), token_cache=str(CACHE))
    (OUT/'config.json').write_text(json.dumps(config, indent=2))
    emit('MODEL_READY', **config)
    assert all(torch.count_nonzero(b.modulation[-1].weight) == 0 for b in model.decoder.blocks)
    assert torch.count_nonzero(model.idm.project.weight) > 0
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def make_optimizer():
        return torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)

    def batch(ids):
        features = z[ids].cuda().flatten(0, 1)
        return features[:, :2], features[:, 2]

    def grad_norm(module):
        grads = [p.grad.square().sum() for p in module.parameters() if p.grad is not None]
        return float(torch.stack(grads).sum().sqrt()) if grads else 0.

    optimizer = make_optimizer()
    smoke = []
    for step in range(5):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pairs, truth = batch(train[:8])
        prediction, code, kl = model(pairs)
        recon = F.mse_loss(prediction, truth)
        loss = recon+1e-5*kl
        assert torch.isfinite(loss)
        loss.backward()
        idm_grad, decoder_grad = grad_norm(model.idm), grad_norm(model.decoder)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        smoke.append(dict(step=step+1, recon_mse=float(recon), kl=float(kl),
                          idm_grad_norm=idm_grad, decoder_grad_norm=decoder_grad,
                          global_grad_norm=float(norm)))
    assert smoke[-1]['idm_grad_norm'] > 0 and smoke[-1]['decoder_grad_norm'] > 0
    assert not torch.equal(model.idm.project.weight.detach().cpu(), initial['idm.project.weight'])
    # Prove reconstruction alone reaches IDM, independently of KL regularization.
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    prediction, code, _ = model(pairs)
    F.mse_loss(prediction, truth).backward()
    recon_idm_grad = grad_norm(model.idm)
    assert recon_idm_grad > 0 and math.isfinite(recon_idm_grad)
    (OUT/'smoke.json').write_text(json.dumps(dict(steps=smoke, reconstruction_only_idm_grad=recon_idm_grad), indent=2))
    emit('SMOKE_PASSED', steps=smoke, reconstruction_only_idm_grad=recon_idm_grad,
         peak_gib=torch.cuda.max_memory_allocated()/2**30)
    model.load_state_dict(initial)
    del initial, optimizer, pairs, truth, prediction, code, loss, recon, kl
    optimizer = make_optimizer()
    torch.manual_seed(239)
    resumed = None
    if args.resume:
        resumed = torch.load(args.resume, map_location='cpu', weights_only=False)
        for key in ['train', 'val', 'batch', 'stride', 'seed', 'dtype', 'lr', 'weight_decay',
                    'grad_clip', 'dropout', 'beta_kl', 'source_order', 'manifest', 'token_cache']:
            assert resumed['config'][key] == config[key], f'Resume configuration mismatch: {key}'
        assert args.epochs > resumed['epoch']
        model.load_state_dict(resumed['model'], strict=True)
        optimizer.load_state_dict(resumed['optimizer'])
        assert optimizer.state and all(g['lr'] == config['lr'] for g in optimizer.param_groups)
        torch.set_rng_state(resumed['torch_rng'])
        torch.cuda.set_rng_state_all(resumed['cuda_rng'])
        config.update(resume_checkpoint=str(args.resume), resumed_epoch=resumed['epoch'],
                      additional_epochs=args.epochs-resumed['epoch'])
        (OUT/'config.json').write_text(json.dumps(config, indent=2))
        emit('RESUME_RESTORED', epoch=resumed['epoch'], step=resumed['step'],
             optimizer_states=len(optimizer.state), lr=optimizer.param_groups[0]['lr'])
    probe = np.asarray(train)[np.linspace(0, len(train)-1, 256, dtype=int)].tolist()

    @torch.no_grad()
    def evaluate(ids, mode='normal'):
        model.eval()
        errors, p_errors, w_errors, cosines, codes, kls = [], [], [], [], [], []
        permutation = np.random.default_rng(239).permutation(ids).tolist()
        for start in range(0, len(ids), 8):
            ix = ids[start:start+8]
            pairs, truth = batch(ix)
            mu, logvar = model.encode(pairs)
            if mode == 'zero_code':
                mu = torch.zeros_like(mu)
            elif mode == 'shuffle_code':
                other, _ = batch(permutation[start:start+len(ix)])
                mu = model.encode(other)[0]
            elif mode == 'same_predictor':
                identical = pairs.clone()
                identical[:, 1] = identical[:, 0]
                mu = model.encode(identical)[0]
            prediction = model.decoder(pairs[:, 0], mu)
            errors.append((prediction-truth).square().mean((1, 2)).reshape(-1, 2).cpu())
            p_errors.append((pairs[:, 0]-truth).square().mean((1, 2)).reshape(-1, 2).cpu())
            w_errors.append((pairs[:, 1]-truth).square().mean((1, 2)).reshape(-1, 2).cpu())
            cosines.append(F.cosine_similarity(prediction, truth, dim=-1).mean(-1).cpu())
            codes.append(mu.reshape(-1, 2, 128).cpu())
            kls.append(float(.5*(mu.square()+logvar.exp()-1-logvar).mean()))
        code = torch.cat(codes)
        return dict(reconstruction_mse=float(torch.cat(errors).mean()),
                    reconstruction_mse_per_time=torch.cat(errors).mean(0).tolist(),
                    predictor_baseline_mse=float(torch.cat(p_errors).mean()),
                    wam_baseline_mse=float(torch.cat(w_errors).mean()),
                    feature_cosine=float(torch.cat(cosines).mean()),
                    code_channel_std=float(code.flatten(1).std(0).mean()),
                    code_rms=float(code.square().mean().sqrt()),
                    code1_code2_cosine=float(F.cosine_similarity(code[:, 0], code[:, 1], dim=-1).mean()),
                    mean_batch_kl=float(np.mean(kls)))

    initial_eval = dict(val=evaluate(val), train_probe=evaluate(probe),
                        zero_code=evaluate(val, 'zero_code'), shuffled_code=evaluate(val, 'shuffle_code'))
    (OUT/'initial_evaluation.json').write_text(json.dumps(initial_eval, indent=2))
    emit('INITIAL_EVALUATION', **initial_eval)
    best, best_epoch, global_step = float('inf'), 0, 0
    start_epoch = 0
    if resumed is not None:
        best, best_epoch, global_step = resumed['best_val'], resumed['best_epoch'], resumed['step']
        start_epoch = resumed['epoch']
        assert abs(initial_eval['val']['reconstruction_mse']-best) < 1e-6, 'Expected final epoch10 best checkpoint'
        save(resumed, OUT/'best.pt')
        del resumed
    started = time.monotonic()

    def checkpoint(name, epoch):
        save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                  epoch=epoch, step=global_step, best_val=best, best_epoch=best_epoch,
                  torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), OUT/name)

    for epoch in range(start_epoch+1, args.epochs+1):
        model.train()
        ids = np.random.default_rng(238+epoch).permutation(train).tolist()
        count, weighted_recon, weighted_kl = 0, 0., 0.
        norms, idm_norms = [], []
        epoch_started = time.monotonic()
        for start in range(0, len(ids), 8):
            ix = ids[start:start+8]
            optimizer.zero_grad(set_to_none=True)
            pairs, truth = batch(ix)
            prediction, code, kl = model(pairs)
            recon = F.mse_loss(prediction, truth)
            loss = recon+1e-5*kl
            loss.backward()
            idm_norms.append(grad_norm(model.idm))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            norms.append(float(norm))
            optimizer.step()
            weighted_recon += float(recon)*len(ix)
            weighted_kl += float(kl)*len(ix)
            count += len(ix)
            global_step += 1
            if len(norms) % 100 == 0:
                emit('TRAIN_PROGRESS', epoch=epoch, batch=len(norms), step=global_step,
                     reconstruction_mse=weighted_recon/count, kl=weighted_kl/count,
                     seconds_per_batch=(time.monotonic()-epoch_started)/len(norms))
        row = dict(epoch=epoch, step=global_step, optimization_recon_mse=weighted_recon/count,
                   optimization_kl=weighted_kl/count, val=evaluate(val), train_probe=evaluate(probe),
                   zero_code=evaluate(val, 'zero_code'), shuffled_code=evaluate(val, 'shuffle_code'),
                   same_predictor=evaluate(val, 'same_predictor'),
                   clip_rate=float(np.mean(np.asarray(norms)>1)),
                   grad_norm_p50=float(np.median(norms)), grad_norm_p95=float(np.quantile(norms,.95)),
                   idm_grad_norm_p50=float(np.median(idm_norms)),
                   clip_scale_p50=float(np.median(np.minimum(1, 1/np.maximum(norms,1e-12)))),
                   peak_gib=torch.cuda.max_memory_allocated()/2**30)
        if row['val']['reconstruction_mse'] < best:
            best, best_epoch = row['val']['reconstruction_mse'], epoch
            checkpoint('best.pt', epoch)
        checkpoint('last.pt', epoch)
        checkpoint(f'epoch{epoch:03d}.pt', epoch)
        row.update(epoch_seconds=time.monotonic()-epoch_started, seconds=time.monotonic()-started)
        with (OUT/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        emit('EPOCH_COMPLETED', **row)
    summary = dict(config=config, initial=initial_eval, best_val_mse=best, best_epoch=best_epoch,
                   final=row, seconds=time.monotonic()-started)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2))
    emit('ALL_COMPLETED', best_epoch=best_epoch, best_val_mse=best, seconds=summary['seconds'])


if __name__ == '__main__':
    main()
