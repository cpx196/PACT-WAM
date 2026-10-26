"""One full matched epoch after pilot; hard bounded to the short-test budget."""
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
from pilot_visual_selfattn100_20261008 import DirectFutureARB, InteractingFutureARB, ROOT, CACHE, PCACHE, MANIFEST, atomic_json, emit

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--variant',choices=['original','visual_selfattn'],required=True)
    args=ap.parse_args()
    started=time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(241)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    parent=ROOT/'artifacts/libero10_visual_selfattn_pilot100_20261008'/args.variant
    out=parent/'full_epoch'
    out.mkdir(exist_ok=False)
    lock=(out/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (out/'train.pid').write_text(str(os.getpid()))
    ck=torch.load(parent/'step100.pt',map_location='cpu',weights_only=False,mmap=True)
    state=ck['model']
    cls=DirectFutureARB if args.variant=='original' else InteractingFutureARB
    model=cls(state['mean'],state['std']).cuda();model.load_state_dict(state)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01);opt.load_state_dict(ck['optimizer'])
    config=dict(ck['config'],continuation='one full epoch; restored model and optimizer from own pilot',
        continuation_seed=241,continuation_lr=1e-5,continuation_source=str(parent/'step100.pt'))
    rows=json.loads(MANIFEST.read_text())
    train=[i for i,r in enumerate(rows) if r['split']=='train'];val=[i for i,r in enumerate(rows) if r['split']=='val']
    assert (len(train),len(val))==(3871,436)
    order=np.random.default_rng(241).permutation(train).tolist()
    config['continuation_train_ids']=order
    atomic_json(config,out/'config.json')
    z=torch.empty(len(rows),2,2,256,1408)
    actions=[];targets=[]
    for i,row in enumerate(rows):
        item=torch.load(CACHE/f'{i:05d}.pt',map_location='cpu',mmap=True,weights_only=False)
        pred=torch.load(PCACHE/'tokens'/f'{i:05d}.pt',map_location='cpu',mmap=True,weights_only=False)
        assert item['index']==pred['index']==i and item['row']==pred['row']==row
        z[i].copy_(torch.stack((pred['predicted_future'],item['wam_future']),1).float())
        actions.append(item['action'].float());targets.append(item['expert_normalized_first6'].float())
    a=torch.stack(actions).cuda();y=torch.stack(targets).cuda()
    assert torch.isfinite(z).all() and torch.isfinite(a).all() and torch.isfinite(y).all()
    probe=config['probe_ids']
    @torch.no_grad()
    def evaluate(ids):
        model.eval();values=[]
        for k in range(0,len(ids),8):
            ix=ids[k:k+8]
            values.append((model(a[ix],z[ix].cuda())-y[ix]).square().mean((1,2)).cpu())
        return float(torch.cat(values).mean())
    emit('DATA_READY',variant=args.variant,seconds=time.monotonic()-started)
    initial=dict(train_probe_mse=evaluate(probe),val_mse=evaluate(val))
    expected=json.loads((parent/'summary.json').read_text())['final']
    assert all(abs(initial[k]-expected[k])<1e-7 for k in initial)
    emit('INITIAL',variant=args.variant,**initial)
    metrics=[]
    for step,k in enumerate(range(0,len(order),8),1):
        if time.monotonic()-started>270:
            emit('TIME_LIMIT',steps=step-1);break
        ix=order[k:k+8]
        model.train();opt.zero_grad(set_to_none=True)
        loss=F.mse_loss(model(a[ix],z[ix].cuda()),y[ix]);assert torch.isfinite(loss)
        loss.backward()
        norm=float(nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step()
        record=dict(step=step,batch_mse=float(loss),global_grad_norm=norm,clipped=norm>1,
            seconds=time.monotonic()-started)
        if step%100==0 or k+8>=len(order):
            record.update(train_probe_mse=evaluate(probe),val_mse=evaluate(val))
            emit('EVALUATION',variant=args.variant,**record)
        metrics.append(record)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    final=dict(train_probe_mse=evaluate(probe),val_mse=evaluate(val),train_full_mse=evaluate(train))
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=100+metrics[-1]['step'],
        epoch=1 if metrics[-1]['step']==484 else None,config=config,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),out/'last.pt')
    result=dict(initial=initial,final=final,steps=metrics[-1]['step'],full_epoch_completed=metrics[-1]['step']==484,
        clip_rate=float(np.mean([r['clipped'] for r in metrics])),seconds=time.monotonic()-started)
    atomic_json(result,out/'summary.json');emit('COMPLETED',variant=args.variant,**result)

if __name__=='__main__':main()
