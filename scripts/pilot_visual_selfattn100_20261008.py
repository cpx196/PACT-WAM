"""Matched short continuation: fixed visual memory versus interacting P/W tokens."""
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
from pilot_direct_future_arb100 import DirectFutureARB, ROOT, CACHE, PCACHE, MANIFEST, atomic_json, emit, gradient_norm

SOURCE = ROOT/'artifacts/libero10_direct_future_arb_continue3_lr1e5_20261008/full_future/epoch003.pt'


class VisualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(768)
        self.attention = nn.MultiheadAttention(768, 12, dropout=0., batch_first=True)
        self.norm2 = nn.LayerNorm(768)
        self.ffn = nn.Sequential(nn.Linear(768, 3072), nn.GELU(), nn.Linear(3072, 768))
        # Identity insertion: initial predictions exactly equal the source model.
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, x):
        q = self.norm1(x)
        x = x + self.attention(q, q, q, need_weights=False)[0]
        return x + self.ffn(self.norm2(x))


class InteractingFutureARB(DirectFutureARB):
    def __init__(self, mean, std):
        super().__init__(mean, std)
        self.visual_blocks = nn.ModuleList([VisualBlock() for _ in range(2)])

    def forward(self, action, visual, include_predictor=True):
        assert include_predictor
        memory = self.visual_project(self.visual_norm(visual))
        memory = self.memory_norm(memory + self.time_embedding + self.source_embedding + self.spatial_embedding)
        memory = memory.flatten(1, 3)
        for block in self.visual_blocks:
            memory = block(memory)
        x = self.action_project((action-self.mean)/self.std)+self.action_position
        for block in self.blocks:
            x = block(x, memory)
        return action[:, :, :6]+self.output(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--variant', choices=['original', 'visual_selfattn'], required=True)
    ap.add_argument('--max-seconds', type=float, default=540.)
    args = ap.parse_args()
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(240)
    np.random.seed(239)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out/'training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (args.out/'metrics.jsonl').exists() and not (args.out/'summary.json').exists()
    (args.out/'train.pid').write_text(str(os.getpid()))
    checkpoint = torch.load(SOURCE, map_location='cpu', weights_only=False, mmap=True)
    state = checkpoint['model']
    cls = DirectFutureARB if args.variant=='original' else InteractingFutureARB
    model = cls(state['mean'], state['std'])
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and all(k.startswith('visual_blocks.') for k in missing)
    if args.variant=='original':
        assert not missing
    model = model.cuda()
    parameters = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=.01)
    rows = json.loads(MANIFEST.read_text())
    train = [i for i,r in enumerate(rows) if r['split']=='train']
    val = [i for i,r in enumerate(rows) if r['split']=='val']
    assert (len(train),len(val))==(3871,436)
    assert {(rows[i]['task_file'],rows[i]['demo']) for i in train}.isdisjoint(
        {(rows[i]['task_file'],rows[i]['demo']) for i in val})
    rng = np.random.default_rng(239)
    groups = {}
    for i in train:
        groups.setdefault(rows[i]['task_file'],[]).append(i)
    groups = [rng.permutation(g).tolist() for _,g in sorted(groups.items())]
    sampled = []
    while len(sampled)<800:
        for group in groups:
            if group and len(sampled)<800:
                sampled.append(group.pop())
    probe = np.array(train)[np.linspace(0,len(train)-1,256,dtype=int)].tolist()
    ids = sorted(set(sampled+probe+val))
    positions = {i:k for k,i in enumerate(ids)}
    features = torch.empty(len(ids),2,2,256,1408)
    actions, targets = [], []
    meta = json.loads((PCACHE/'config.json').read_text())
    assert meta['checkpoint_step']==2000 and meta['main_view_rotation']==180
    assert meta['calibration']['fitted_split']=='train_only'
    for k,i in enumerate(ids):
        item = torch.load(CACHE/f'{i:05d}.pt',map_location='cpu',weights_only=False,mmap=True)
        pred = torch.load(PCACHE/'tokens'/f'{i:05d}.pt',map_location='cpu',weights_only=False,mmap=True)
        assert item['index']==pred['index']==i and item['row']==pred['row']==rows[i]
        assert pred['checkpoint']==meta['checkpoint'] and pred['calibration']==meta['calibration']
        z = torch.stack((pred['predicted_future'],item['wam_future']),1).float()
        a = item['action'].float()
        y = item['expert_normalized_first6'].float()
        assert z.shape==(2,2,256,1408) and a.shape==(8,7) and y.shape==(8,6)
        assert torch.isfinite(z).all() and torch.isfinite(a).all() and torch.isfinite(y).all()
        assert torch.allclose(y,(item['expert_executable']*item['scale']+item['offset'])[:,:6],atol=1e-6,rtol=1e-6)
        features[k].copy_(z)
        actions.append(a); targets.append(y)
    a = torch.stack(actions).cuda(); y = torch.stack(targets).cuda()
    probe_ix = [positions[i] for i in probe]; val_ix = [positions[i] for i in val]
    config = dict(variant=args.variant,parameters=parameters,source_checkpoint=str(SOURCE),
        source_epoch=checkpoint['epoch'],source_step=checkpoint['step'],
        visual_layers=0 if args.variant=='original' else 2,action_layers=6,
        hidden=768,heads=12,ffn=3072,memory_tokens=1024,pooling=False,
        attention_mask='full bidirectional on available predicted futures',
        initialization='shared epoch003 model; new residual visual blocks identity-initialized',
        optimizer='AdamW reset identically in both arms',lr=1e-5,weight_decay=.01,clip_norm=1.,
        precision='FP32, TF32 disabled',batch=8,steps=100,
        train_ids=sampled,probe_ids=probe,val_ids=val,train_pool=3871,val_windows=436,
        supervision='demonstrated normalized first8x6 action MSE only',gripper='unchanged',
        real_future_input=False,predictor_metadata=meta,
        caveat='short offline continuation; capacity also differs; not robot success')
    atomic_json(config,args.out/'config.json')
    emit('DATA_READY',variant=args.variant,parameters=parameters,seconds=time.monotonic()-started)

    def predict(ix):
        return model(a[ix],features[ix].cuda())

    @torch.no_grad()
    def evaluate(ix):
        model.eval()
        values=[]
        for k in range(0,len(ix),8):
            batch=ix[k:k+8]
            values.append((predict(batch)-y[batch]).square().mean((1,2)).cpu())
        return float(torch.cat(values).mean())

    model.eval()
    with torch.no_grad():
        ix=val_ix[:2]
        baseline=DirectFutureARB(state['mean'],state['std']).cuda()
        baseline.load_state_dict(state); baseline.eval()
        difference=float((predict(ix)-baseline(a[ix],features[ix].cuda())).abs().max())
        assert difference==0., difference
        del baseline
    del checkpoint,state
    torch.cuda.empty_cache()
    initial=dict(train_probe_mse=evaluate(probe_ix),val_mse=evaluate(val_ix),
        raw_val_mse=float((a[val_ix,:,:6]-y[val_ix]).square().mean()),initial_identity_max_difference=difference)
    atomic_json(initial,args.out/'initial.json')
    emit('INITIAL',variant=args.variant,**initial)
    first_visual = model.visual_project.weight.detach().clone()
    first_new = model.visual_blocks[0].attention.out_proj.weight.detach().clone() if hasattr(model,'visual_blocks') else None
    torch.cuda.reset_peak_memory_stats()
    records=[]
    for step in range(1,101):
        if time.monotonic()-started>args.max_seconds:
            emit('TIME_LIMIT',completed_steps=step-1)
            break
        ix=[positions[i] for i in sampled[(step-1)*8:step*8]]
        model.train(); opt.zero_grad(set_to_none=True)
        loss=F.mse_loss(predict(ix),y[ix])
        assert torch.isfinite(loss)
        loss.backward()
        visual_grad=gradient_norm(model.visual_project.parameters())
        new_grad=gradient_norm(model.visual_blocks.parameters()) if hasattr(model,'visual_blocks') else None
        qkv_grad=gradient_norm([model.visual_blocks[0].attention.in_proj_weight]) if hasattr(model,'visual_blocks') else None
        norm=float(nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
        opt.step()
        record=dict(step=step,batch_mse=float(loss),global_grad_norm=norm,clipped=norm>1.,
            visual_grad_norm=visual_grad,new_visual_grad_norm=new_grad,new_qkv_grad_norm=qkv_grad,
            elapsed_seconds=time.monotonic()-started)
        if step==3:
            assert visual_grad>0 and not torch.equal(model.visual_project.weight,first_visual)
            if first_new is not None:
                assert new_grad>0 and qkv_grad>0
                assert not torch.equal(model.visual_blocks[0].attention.out_proj.weight,first_new)
            atomic_json(dict(passed=True,**record),args.out/'smoke.json')
            emit('SMOKE_PASSED',variant=args.variant,**record)
        if step in (25,50,75,100):
            record.update(train_probe_mse=evaluate(probe_ix),val_mse=evaluate(val_ix))
            emit('EVALUATION',variant=args.variant,**record)
        records.append(record)
        with (args.out/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
    final=dict(train_probe_mse=evaluate(probe_ix),val_mse=evaluate(val_ix))
    steps=records[-1]['step']
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=steps,config=config,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),args.out/f'step{steps:03d}.pt')
    saved=torch.load(args.out/f'step{steps:03d}.pt',map_location='cpu',weights_only=False,mmap=True)
    assert saved['step']==steps and len(saved['optimizer']['state'])>0
    result=dict(initial=initial,final=final,completed_steps=steps,parameters=parameters,
        seconds=time.monotonic()-started,clip_rate=float(np.mean([r['clipped'] for r in records])),
        peak_gib=torch.cuda.max_memory_allocated()/2**30,smoke_passed=(args.out/'smoke.json').exists(),
        optimizer_saved=True,config=config)
    atomic_json(result,args.out/'summary.json')
    emit('COMPLETED',variant=args.variant,steps=steps,**final)


if __name__=='__main__':
    main()
