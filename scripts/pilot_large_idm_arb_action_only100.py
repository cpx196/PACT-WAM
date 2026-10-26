"""Capacity pilot: action-only joint optimization, no Decoder/KL/ranking loss.

Fixed 64 train windows are intentionally reused to test fitting, NOT full-data
generalization. Validation windows belong to entirely held-out demonstrations.
"""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path('/data/pxchen/PACT-WAM')
CACHE = ROOT/'artifacts/libero10_idm_cache_stride32_20261004/tokens'
PCACHE = ROOT/'artifacts/libero10_calibrated_idm_arb_joint10_20261007/cache'
MANIFEST = ROOT/'artifacts/libero10_idm_dense_windows_20261004/windows_stride32.json'


def emit(event, **kw):
    print(json.dumps(dict(time=time.strftime('%F %T'), event=event, **kw)), flush=True)


class IDM(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(1408)
        self.project = nn.Linear(1408, 800)
        self.source = nn.Parameter(torch.randn(1, 2, 1, 800)*.02)
        self.position = nn.Parameter(torch.randn(1, 1, 256, 800)*.02)
        self.query = nn.Parameter(torch.randn(1, 1, 800)*.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(800, 10, 3200, dropout=0.,
                activation='gelu', batch_first=True, norm_first=True)
            for _ in range(13)])
        self.output = nn.Sequential(nn.LayerNorm(800), nn.Linear(800, 512))

    def forward(self, pairs):
        # Each matching future time is an independent comparison; same IDM.
        b = len(pairs)
        x = self.project(self.norm(pairs))+self.source+self.position
        x = torch.cat((x.flatten(1, 2), self.query.expand(b, -1, -1)), 1)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.training else block(x)
        return self.output(x[:, -1])


class ARBBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(640) for _ in range(3)])
        self.self_attn = nn.MultiheadAttention(640, 10, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(640, 10, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(640, 2560), nn.GELU(), nn.Linear(2560, 640))
        self.gate = nn.Parameter(torch.tensor(.1))

    def forward(self, x, memory):
        q = self.norms[0](x)
        x = x+self.self_attn(q, q, q, need_weights=False)[0]
        x = x+self.gate.tanh()*self.cross_attn(self.norms[1](x), memory, memory, need_weights=False)[0]
        return x+self.ff(self.norms[2](x))


class ARB(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.action = nn.Linear(7, 640)
        self.la = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 640))
        self.apos = nn.Parameter(torch.randn(1, 8, 640)*.02)
        self.lpos = nn.Parameter(torch.randn(1, 2, 640)*.02)
        self.blocks = nn.ModuleList([ARBBlock() for _ in range(7)])
        self.output = nn.Sequential(nn.LayerNorm(640), nn.Linear(640, 6))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, action, la):
        x = self.action((action-self.mean)/self.std)+self.apos
        memory = self.la(la)+self.lpos
        for block in self.blocks:
            x = block(x, memory)
        return action[:, :, :6]+self.output(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cpu-preflight', action='store_true')
    p.add_argument('--full-epoch', action='store_true', help='All 3871 train windows exactly once, 436 val')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(239)
    idm = IDM()
    arb = ARB(torch.zeros(8, 7), torch.ones(8, 7))
    counts = dict(idm=sum(p.numel() for p in idm.parameters()),
                  arb=sum(p.numel() for p in arb.parameters()))
    counts['total'] = sum(counts.values())
    assert 140_000_000 < counts['total'] < 160_000_000
    if args.cpu_preflight:
        idm.eval(); arb.eval()
        with torch.no_grad():
            code = idm(torch.randn(1,2,256,1408))
            assert code.shape == (1,512) and torch.isfinite(code).all()
            action = torch.randn(1,8,7)
            result = arb(action,code[:,None].expand(-1,2,-1))
            assert result.shape == (1,8,6) and torch.equal(result,action[:,:,:6])
        emit('CPU_PREFLIGHT_PASSED', **counts)
        return
    del idm, arb
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out/'training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (args.out/'metrics.jsonl').exists()
    (args.out/'train.pid').write_text(str(os.getpid()))
    rows = json.loads(MANIFEST.read_text())
    train = [i for i, r in enumerate(rows) if r['split']=='train']
    val = [i for i, r in enumerate(rows) if r['split']=='val']
    assert (len(train), len(val)) == (3871, 436)
    assert {(rows[i]['task_file'], rows[i]['demo']) for i in train}.isdisjoint(
        {(rows[i]['task_file'], rows[i]['demo']) for i in val})
    def select(ids):
        # Round-robin tasks, random windows within each task; deterministic.
        rng = np.random.default_rng(239)
        groups = {}
        for i in ids:
            groups.setdefault(rows[i]['task_file'], []).append(i)
        groups = [rng.permutation(g).tolist() for _, g in sorted(groups.items())]
        picked = []
        while len(picked)<64:
            for group in groups:
                if group and len(picked)<64:
                    picked.append(group.pop())
        return picked
    tr, va = (train, val) if args.full_epoch else (select(train), select(val))
    ids = tr+va
    meta = json.loads((PCACHE/'config.json').read_text())
    assert meta['checkpoint_step']==2000 and meta['calibration']['fitted_split']=='train_only'
    assert meta['manifest']==str(MANIFEST) and meta['main_view_rotation']==180
    z = torch.empty(len(ids),2,2,256,1408)
    actions, targets = [], []
    for local_i, i in enumerate(ids):
        item = torch.load(CACHE/f'{i:05d}.pt', map_location='cpu', weights_only=False, mmap=True)
        updated = torch.load(PCACHE/'tokens'/f'{i:05d}.pt', map_location='cpu', weights_only=False, mmap=True)
        assert item['row']==updated['row']==rows[i] and item['index']==updated['index']==i
        assert updated['checkpoint']==meta['checkpoint'] and updated['calibration']==meta['calibration']
        feature = torch.stack((updated['predicted_future'], item['wam_future']), 1).float()
        assert feature.shape==(2,2,256,1408) and torch.isfinite(feature).all()
        z[local_i].copy_(feature)
        a, y = item['action'].float(), item['expert_normalized_first6'].float()
        assert a.shape==(8, 7) and y.shape==(8, 6)
        assert torch.allclose(y, (item['expert_executable']*item['scale']+item['offset'])[:, :6], atol=1e-6, rtol=1e-6)
        actions.append(a); targets.append(y)
        if (local_i+1)%512==0:
            emit('CACHE_LOAD_PROGRESS',loaded=local_i+1,total=len(ids))
    a, y = torch.stack(actions).cuda(), torch.stack(targets).cuda()
    assert z.shape==(len(ids), 2, 2, 256, 1408)
    mean, std = a[:len(tr)].mean(0), a[:len(tr)].std(0).clamp_min(.01)
    steps = (len(tr)+7)//8 if args.full_epoch else 100
    config = dict(parameters=counts, idm_layers=13, idm_hidden=800, idm_heads=10,
        arb_layers=7, arb_hidden=640, arb_heads=10, la_shape=[2,512],
        steps_per_arm=steps, epochs=1 if args.full_epoch else None, batch=8, lr=1e-4, weight_decay=.01, clip_norm=1.,
        precision='FP32; TF32 disabled', dropout=0., loss='action MSE ONLY',
        decoder=False, kl=False, ranking=False, initialization='fresh random; zero residual output',
        train_ids=tr, val_ids=va, train_windows=len(tr), val_windows=len(va),
        predictor_metadata=meta, gripper='unchanged; first6 supervised',
        seed=239, no_early_stopping=True, fixed_train_fitting_pilot=not args.full_epoch,
        caveat='Offline action error, not robot rollout evaluation',
        source_order=['predictor','wam'], real_future_used=False)
    (args.out/'config.json').write_text(json.dumps(config, indent=2))
    (args.out/'window_manifest.json').write_text(json.dumps([dict(index=i, **rows[i]) for i in ids], indent=2))
    emit('DATA_READY', **config)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    summaries = {}
    for variant in ('normal_la', 'zero_la'):
        torch.manual_seed(239)
        idm = IDM().cuda() if variant=='normal_la' else None
        # Reproduce exact ARB initialization in both arms despite extra IDM draws.
        torch.manual_seed(240)
        arb = ARB(mean, std).cuda()
        params = list(arb.parameters())+(list(idm.parameters()) if idm else [])
        opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01)
        dest = args.out/variant; dest.mkdir()
        first_idm = idm.project.weight.detach().clone() if idm else None
        first_arb = arb.output[-1].weight.detach().clone()
        def predict(ix, mode='normal'):
            code = a.new_zeros(len(ix), 2, 512)
            if idm is not None and mode!='zero':
                pair = z[ix].cuda().flatten(0,1)
                code = idm(pair).reshape(len(ix),2,512)
                if mode=='shuffle': code = code.roll(1,0)
            return arb(a[ix], code)
        @torch.no_grad()
        def evaluate(ix, mode='normal'):
            arb.eval()
            if idm is not None: idm.eval()
            errs=[]
            for start in range(0,len(ix),8):
                batch=ix[start:start+8]
                errs.append((predict(batch,mode)-y[batch]).square().mean((1,2)).cpu())
            return float(torch.cat(errs).mean())
        train_ix=list(range(len(tr))); val_ix=list(range(len(tr),len(ids)))
        probe_ix=np.asarray(train_ix)[np.linspace(0,len(tr)-1,min(256,len(tr)),dtype=int)].tolist()
        initial=dict(train_mse=evaluate(train_ix), val_mse=evaluate(val_ix))
        emit('INITIAL', variant=variant, **initial)
        metrics=[]; started=time.monotonic()
        order=np.random.default_rng(239).permutation(len(tr)).tolist()
        seen=[]
        for step in range(1,steps+1):
            if not args.full_epoch and (step-1)%8==0:
                order=np.random.default_rng(239+(step-1)//8).permutation(64).tolist()
            ix=order[(step-1)*8:step*8] if args.full_epoch else order[((step-1)%8)*8:((step-1)%8+1)*8]
            seen.extend(ix)
            arb.train()
            if idm is not None: idm.train()
            opt.zero_grad(set_to_none=True)
            loss=F.mse_loss(predict(ix), y[ix]); assert torch.isfinite(loss)
            loss.backward()
            def gn(module):
                gs=[p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None]
                return float(torch.stack(gs).sum().sqrt()) if gs else 0.
            gi, ga=gn(idm) if idm else 0., gn(arb)
            g=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
            opt.step()
            if step==3:
                assert ga>0 and torch.isfinite(loss)
                if idm is not None:
                    assert gi>0 and not torch.equal(idm.project.weight,first_idm)
                emit('SMOKE_PASSED',variant=variant,step=step,idm_grad_norm=gi,arb_grad_norm=ga,
                     note='zero output intentionally blocks IDM gradient at first step; subsequent steps must update IDM')
            row=dict(variant=variant,step=step,batch_mse=float(loss),idm_grad_norm=gi,
                arb_grad_norm=ga,global_grad_norm=g,clipped=g>1,
                peak_gib=torch.cuda.max_memory_allocated()/2**30)
            evaluation_step=(step==3 or step%100==0 or step==steps) if args.full_epoch else step in (1,3,10,25,50,75,100)
            if evaluation_step:
                row.update(train_probe_mse=evaluate(probe_ix),val_mse=evaluate(val_ix),
                           elapsed=time.monotonic()-started)
                emit('EVALUATION',**row)
            metrics.append(row)
            with (args.out/'metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            if (step%100==0 or step==steps) if args.full_epoch else step in (50,100):
                tmp=dest/f'step{step:03d}.tmp'
                torch.save(dict(idm=idm.state_dict() if idm else None, arb=arb.state_dict(),
                    optimizer=opt.state_dict(), step=step, config=config,variant=variant,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),tmp)
                tmp.replace(dest/f'step{step:03d}.pt')
        if args.full_epoch:
            assert sorted(seen)==train_ix and len(seen)==3871, 'Must cover each training window exactly once'
        final=dict(train_mse=evaluate(train_ix), val_mse=evaluate(val_ix),
                   zero_la_val_mse=evaluate(val_ix,'zero'),shuffle_la_val_mse=evaluate(val_ix,'shuffle'))
        summary=dict(initial=initial,final=final,seconds=time.monotonic()-started,
            idm_gradient_nonzero_steps=sum(m['idm_grad_norm']>0 for m in metrics),
            clip_rate=float(np.mean([m['clipped'] for m in metrics])),
            idm_parameter_delta_rms=float((idm.project.weight-first_idm).square().mean().sqrt()) if idm else None,
            arb_parameter_delta_rms=float((arb.output[-1].weight-first_arb).square().mean().sqrt()),
            peak_gib=torch.cuda.max_memory_allocated()/2**30)
        assert summary['arb_parameter_delta_rms']>0
        if idm: assert summary['idm_parameter_delta_rms']>0 and summary['idm_gradient_nonzero_steps']>0
        summaries[variant]=summary
        (dest/'summary.json').write_text(json.dumps(summary,indent=2))
        emit('ARM_COMPLETED',variant=variant,**summary)
        del opt,params,arb,idm,first_idm,first_arb
        torch.cuda.empty_cache()
    (args.out/'summary.json').write_text(json.dumps(dict(config=config,results=summaries),indent=2))
    emit('PILOT_COMPLETED')


if __name__=='__main__': main()
