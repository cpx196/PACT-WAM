"""Matched 100-step pilot: full Predictor/WAM tokens -> action-query ARB.

No IDM, pooled LA, reconstruction, ranking, online guidance, or execution feedback.
Frozen caches only. Demonstration-disjoint full validation; each train window used
at most once. Offline action MSE is NOT robot success rate.
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

ROOT = Path('/data/pxchen/PACT-WAM')
CACHE = ROOT/'artifacts/libero10_idm_cache_stride32_20261004/tokens'
PCACHE = ROOT/'artifacts/libero10_calibrated_idm_arb_joint10_20261007/cache'
MANIFEST = ROOT/'artifacts/libero10_idm_dense_windows_20261004/windows_stride32.json'


def emit(event, **kwargs):
    print(json.dumps(dict(time=time.strftime('%F %T'), event=event, **kwargs)), flush=True)


def atomic_json(value, path):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(768) for _ in range(3)])
        self.self_attention = nn.MultiheadAttention(768, 12, dropout=0., batch_first=True)
        self.cross_attention = nn.MultiheadAttention(768, 12, dropout=0., batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(768, 3072), nn.GELU(), nn.Linear(3072, 768))

    def forward(self, action, memory):
        q = self.norms[0](action)
        action = action+self.self_attention(q, q, q, need_weights=False)[0]
        action = action+self.cross_attention(self.norms[1](action), memory, memory,
                                           need_weights=False)[0]
        return action+self.ffn(self.norms[2](action))


class DirectFutureARB(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.visual_norm = nn.LayerNorm(1408)
        self.visual_project = nn.Linear(1408, 768)
        self.memory_norm = nn.LayerNorm(768)
        # memory layout: batch, future-time, source(P/W), spatial-position, channel
        self.time_embedding = nn.Parameter(torch.randn(1, 2, 1, 1, 768)*.02)
        self.source_embedding = nn.Parameter(torch.randn(1, 1, 2, 1, 768)*.02)
        self.spatial_embedding = nn.Parameter(torch.randn(1, 1, 1, 256, 768)*.02)
        self.action_project = nn.Linear(7, 768)
        self.action_position = nn.Parameter(torch.randn(1, 8, 768)*.02)
        self.blocks = nn.ModuleList([Block() for _ in range(6)])
        self.output = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 6))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, action, visual, include_predictor=True):
        if include_predictor:
            sources = self.source_embedding
        else:
            visual = visual[:, :, 1:2]
            sources = self.source_embedding[:, :, 1:2]
        memory = self.visual_project(self.visual_norm(visual))
        memory = self.memory_norm(memory+self.time_embedding+sources+self.spatial_embedding)
        memory = memory.flatten(1, 3)  # 1024 tokens(full) or 512(WAM only); NO pooling
        x = self.action_project((action-self.mean)/self.std)+self.action_position
        for block in self.blocks:
            x = block(x, memory)
        return action[:, :, :6]+self.output(x)


def gradient_norm(parameters):
    values = [p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cpu-preflight', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(240)
    model = DirectFutureARB(torch.zeros(8, 7), torch.ones(8, 7))
    parameters = sum(p.numel() for p in model.parameters())
    assert 55_000_000 < parameters < 65_000_000
    if args.cpu_preflight:
        model.eval()
        action = torch.randn(1, 8, 7)
        visual = torch.randn(1, 2, 2, 256, 1408)
        with torch.no_grad():
            for include in (True, False):
                output = model(action, visual, include)
                assert output.shape == (1, 8, 6) and torch.equal(output, action[:, :, :6])
        emit('CPU_PREFLIGHT_PASSED', parameters=parameters)
        return
    del model
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out/'training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (args.out/'metrics.jsonl').exists() and not (args.out/'summary.json').exists()
    (args.out/'train.pid').write_text(str(os.getpid()))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = json.loads(MANIFEST.read_text())
    train = [i for i, row in enumerate(rows) if row['split']=='train']
    val = [i for i, row in enumerate(rows) if row['split']=='val']
    assert (len(train), len(val)) == (3871, 436)
    assert {(rows[i]['task_file'], rows[i]['demo']) for i in train}.isdisjoint(
        {(rows[i]['task_file'], rows[i]['demo']) for i in val})
    # Balanced interleaving across tasks; 800 UNIQUE windows for 100 batches.
    rng = np.random.default_rng(239)
    groups = {}
    for i in train:
        groups.setdefault(rows[i]['task_file'], []).append(i)
    groups = [rng.permutation(group).tolist() for _, group in sorted(groups.items())]
    sampled = []
    while len(sampled)<800:
        for group in groups:
            if group and len(sampled)<800:
                sampled.append(group.pop())
    assert len(sampled)==len(set(sampled))==800
    probe = np.asarray(train)[np.linspace(0, len(train)-1, 256, dtype=int)].tolist()
    ids = sorted(set(sampled+probe+val))
    positions = {i: local for local, i in enumerate(ids)}
    z = torch.empty(len(ids), 2, 2, 256, 1408, dtype=torch.float32)
    all_actions, all_targets = [], []
    meta = json.loads((PCACHE/'config.json').read_text())
    assert meta['checkpoint_step']==2000 and meta['calibration']['fitted_split']=='train_only'
    assert meta['manifest']==str(MANIFEST) and meta['main_view_rotation']==180
    started_load = time.monotonic()
    for i, row in enumerate(rows):
        item = torch.load(CACHE/f'{i:05d}.pt', map_location='cpu', weights_only=False, mmap=True)
        assert item['index']==i and item['row']==row
        a = item['action'].float()
        target = item['expert_normalized_first6'].float()
        assert a.shape==(8,7) and target.shape==(8,6)
        assert torch.isfinite(a).all() and torch.isfinite(target).all()
        assert torch.allclose(target,(item['expert_executable']*item['scale']+item['offset'])[:,:6],
                              atol=1e-6,rtol=1e-6)
        all_actions.append(a); all_targets.append(target)
        if i in positions:
            updated = torch.load(PCACHE/'tokens'/f'{i:05d}.pt',map_location='cpu',
                                 weights_only=False,mmap=True)
            assert updated['index']==i and updated['row']==row
            assert updated['checkpoint']==meta['checkpoint'] and updated['calibration']==meta['calibration']
            features = torch.stack((updated['predicted_future'],item['wam_future']),1).float()
            assert features.shape==(2,2,256,1408) and torch.isfinite(features).all()
            z[positions[i]].copy_(features)
        if (i+1)%512==0:
            emit('CACHE_LOAD_PROGRESS', checked=i+1,total=len(rows),seconds=time.monotonic()-started_load)
    all_a = torch.stack(all_actions)
    mean = all_a[train].mean(0); std = all_a[train].std(0).clamp_min(.01)
    a = all_a[ids].cuda(); y = torch.stack(all_targets)[ids].cuda()
    batches = [[positions[i] for i in sampled[k:k+8]] for k in range(0,800,8)]
    probe_ix = [positions[i] for i in probe]
    val_ix = [positions[i] for i in val]
    fit_ix = [positions[i] for i in sampled]
    # Same-task permutation, keeping the two future times paired, for a final
    # diagnostic of whether sample-specific Predictor conditioning is used.
    shuffle = {}
    val_groups = {}
    for i in val:
        val_groups.setdefault(rows[i]['task_file'], []).append(positions[i])
    for group in val_groups.values():
        assert len(group)>1
        for left, right in zip(group,group[1:]+group[:1]):
            shuffle[left] = right
    config = dict(parameters=parameters,architecture='direct full-future action-query Transformer ARB',
        layers=6,hidden=768,heads=12,ffn=3072,batch=8,steps_per_arm=100,
        lr=1e-4,weight_decay=.01,clip_norm=1.,dropout=0.,precision='FP32; TF32 disabled',
        loss='normalized demonstrated action MSE, first8x6 ONLY',gripper='unchanged',
        frozen='FastWAM, JEPA Encoder, calibrated LIBERO10 Predictor step2000; cached outputs only',
        initialization='fresh identical seed240, zero residual head',data_seed=239,
        source_order=['predictor','wam'],memory_shape=[2,2,256,1408],full_memory_tokens=1024,
        no_predictor_memory_tokens=512,spatial_or_time_pooling=False,idm=False,la=False,
        reconstruction_loss=False,ranking_loss=False,feedback=False,rms_cap=False,
        train_pool_windows=len(train),sampled_train_windows=800,val_windows=len(val),
        train_ids=sampled,train_probe_ids=probe,val_ids=val,
        sampled_task_counts={task:sum(rows[i]['task_file']==task for i in sampled) for task in sorted({rows[i]['task_file'] for i in sampled})},
        input_normalization='all 3871 train windows only',real_future_input=False,
        predictor_metadata=meta,early_stopping=False,
        evaluation_steps=[0,3,10,25,50,75,100],save_steps=[50,100],
        caveat='offline validation action MSE; not rollout success or proof of convergence')
    atomic_json(config,args.out/'config.json')
    atomic_json([dict(index=i,**rows[i]) for i in ids],args.out/'window_manifest.json')
    emit('DATA_READY',parameters=parameters,loaded_windows=len(ids),train_windows=800,
         val_windows=436,load_seconds=time.monotonic()-started_load)
    results = {}
    for variant in ('full_future','no_predictor'):
        include = variant=='full_future'
        torch.manual_seed(240)
        model = DirectFutureARB(mean,std).cuda()
        opt = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
        dest = args.out/variant; dest.mkdir()
        first_project = model.visual_project.weight.detach().clone()
        first_cross = model.blocks[0].cross_attention.in_proj_weight.detach().clone()
        first_output = model.output[-1].weight.detach().clone()
        torch.cuda.reset_peak_memory_stats()

        def predict(ix, mode='normal'):
            features = z[ix].cuda()
            if mode=='shuffle_predictor':
                features[:,:,0] = z[[shuffle[i] for i in ix],:,0].cuda()
            return model(a[ix],features,include_predictor=include)

        @torch.no_grad()
        def evaluate(ix, mode='normal'):
            model.eval(); values=[]
            for start in range(0,len(ix),8):
                batch=ix[start:start+8]
                values.append((predict(batch,mode)-y[batch]).square().mean((1,2)).cpu())
            return float(torch.cat(values).mean())

        raw_probe = float((a[probe_ix,:,:6]-y[probe_ix]).square().mean())
        raw_val = float((a[val_ix,:,:6]-y[val_ix]).square().mean())
        initial = dict(train_probe_mse=evaluate(probe_ix),val_mse=evaluate(val_ix),
                       raw_train_probe_mse=raw_probe,raw_val_mse=raw_val)
        assert abs(initial['train_probe_mse']-raw_probe)<1e-7
        assert abs(initial['val_mse']-raw_val)<1e-7
        atomic_json(initial,dest/'initial.json')
        emit('INITIAL',variant=variant,**initial)

        def save(step):
            temporary=dest/f'step{step:03d}.tmp'
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,
                variant=variant,config=config,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),
                next_batch=step),temporary)
            temporary.replace(dest/f'step{step:03d}.pt')

        metrics=[]; started=time.monotonic()
        for step,ix in enumerate(batches,1):
            model.train(); opt.zero_grad(set_to_none=True)
            loss=F.mse_loss(predict(ix),y[ix])
            assert torch.isfinite(loss)
            loss.backward()
            visual_g=gradient_norm(model.visual_project.parameters())
            cross_g=gradient_norm(model.blocks[0].cross_attention.parameters())
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
            opt.step()
            if step==3:
                assert visual_g>0 and cross_g>0
                assert not torch.equal(model.visual_project.weight,first_project)
                assert not torch.equal(model.blocks[0].cross_attention.in_proj_weight,first_cross)
                smoke=dict(passed=True,step=3,visual_grad_norm=visual_g,cross_grad_norm=cross_g,
                    visual_parameter_updated=True,cross_parameter_updated=True,
                    finite=True,peak_gib=torch.cuda.max_memory_allocated()/2**30,
                    note='zero residual head blocks upstream gradient on step1 only')
                atomic_json(smoke,dest/'smoke.json')
                emit('SMOKE_PASSED',variant=variant,**smoke)
            record=dict(variant=variant,step=step,batch_mse=float(loss),
                visual_grad_norm=visual_g,cross_grad_norm=cross_g,global_grad_norm=grad,
                clipped=grad>1,peak_gib=torch.cuda.max_memory_allocated()/2**30)
            if step in (3,10,25,50,75,100):
                record.update(train_probe_mse=evaluate(probe_ix),val_mse=evaluate(val_ix),
                              elapsed=time.monotonic()-started)
                emit('EVALUATION',**record)
            metrics.append(record)
            with (args.out/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            if step in (50,100):
                save(step)
                # Verify saved optimizer/step/provenance; full model forward
                # roundtrip is checked at final checkpoint below.
                ck=torch.load(dest/f'step{step:03d}.pt',map_location='cpu',weights_only=False,mmap=True)
                assert ck['step']==step and ck['variant']==variant
                assert len(ck['optimizer']['state'])>0
                del ck
        final = dict(train_probe_mse=evaluate(probe_ix),fit800_mse=evaluate(fit_ix),
                     val_mse=evaluate(val_ix),
                     shuffled_predictor_val_mse=evaluate(val_ix,'shuffle_predictor') if include else None)
        model.eval()
        with torch.no_grad():
            reference=predict(val_ix[:2]).cpu()
        ck=torch.load(dest/'step100.pt',map_location='cpu',weights_only=False,mmap=True)
        clone=DirectFutureARB(mean,std).cuda(); clone.load_state_dict(ck['model']); clone.eval()
        with torch.no_grad():
            restored=clone(a[val_ix[:2]],z[val_ix[:2]].cuda(),include).cpu()
        assert torch.equal(reference,restored)
        del ck,clone
        result=dict(initial=initial,final=final,seconds=time.monotonic()-started,
            visual_nonzero_grad_steps=sum(r['visual_grad_norm']>0 for r in metrics),
            clip_rate=float(np.mean([r['clipped'] for r in metrics])),
            visual_delta_rms=float((model.visual_project.weight-first_project).square().mean().sqrt()),
            cross_delta_rms=float((model.blocks[0].cross_attention.in_proj_weight-first_cross).square().mean().sqrt()),
            output_delta_rms=float((model.output[-1].weight-first_output).square().mean().sqrt()),
            peak_gib=max(r['peak_gib'] for r in metrics),checkpoint_roundtrip=True)
        assert result['visual_delta_rms']>0 and result['cross_delta_rms']>0
        atomic_json(result,dest/'summary.json'); results[variant]=result
        emit('ARM_COMPLETED',variant=variant,**result)
        del model,opt,first_project,first_cross,first_output
        torch.cuda.empty_cache()
    assert abs(results['full_future']['initial']['val_mse']-results['no_predictor']['initial']['val_mse'])<1e-7
    atomic_json(dict(config=config,results=results),args.out/'summary.json')
    emit('PILOT_COMPLETED')


if __name__=='__main__':
    main()
