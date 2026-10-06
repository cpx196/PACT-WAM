"""100-step one-sided WAM-to-expert path-ranking pilot."""
import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path('/data/pxchen/PACT-WAM')
BASE=ROOT/'artifacts/libero10_predictor_action_oneepoch_20260930'
RANK_CACHE=ROOT/'artifacts/libero10_predictor_energy_ranking_pilot100_20260930'
OUT=ROOT/'artifacts/libero10_predictor_path_ranking_pilot100_20260930'
SOURCE=ROOT/'checkpoints/vjepa2-ac-libero90-euler-fixed-rot180-step2000-inference.pt'
sys.path.insert(0,str(ROOT))

def emit(event,**kw):print(json.dumps(dict(time=time.strftime('%F %T'),event=event,**kw)),flush=True)

def main(smoke=False):
    from experiments.libero.vjepa2_ac_ranker import VJEPA2ACRanker,LiberoACAdapter
    from scripts.train_predictor_energy_ranking_pilot import selected_ids,configure_predictor
    rows,order,valids=selected_ids()
    assert len(list((RANK_CACHE/'real_latent').glob('*.pt')))==150
    assert len(list((RANK_CACHE/'wam_action').glob('*.pt')))==150
    ranker=VJEPA2ACRanker(checkpoint_path=str(SOURCE),vjepa2_repo='/data/pxchen/vjepa2',
        device='cpu',dtype=torch.float32,max_ac_steps=2)
    del ranker.encoder;ranker.device=torch.device('cuda')
    predictor=ranker.predictor.cuda().eval()
    heads,blocks=configure_predictor(predictor);params=heads+blocks
    optimizer=torch.optim.AdamW([dict(params=heads,lr=1e-5),dict(params=blocks,lr=3e-6)],weight_decay=0.04)
    adapter=LiberoACAdapter(low_level_steps_per_ac_step=4)
    runroot=OUT/('smoke' if smoke else 'train100');runroot.mkdir(parents=True,exist_ok=False)
    metric_path=runroot/'metrics.jsonl'
    trust_rms=0.02;margin=5e-5;temperature=2e-4

    def load(index):
        r=torch.load(BASE/'wam'/f'{index:04d}.pt',map_location='cpu',weights_only=False)
        real=torch.load(RANK_CACHE/'real_latent'/f'{index:04d}.pt',map_location='cpu',weights_only=True).cuda()
        online=torch.load(BASE/'latent'/f'{index:04d}.pt',map_location='cpu',weights_only=True).cuda()
        wam=torch.load(RANK_CACHE/'wam_action'/f'{index:04d}.pt',map_location='cpu',weights_only=True).cuda()
        return r,real,online,wam

    def one(index,update):
        r,real,online,wam=load(index)
        expert=r['expert'].cuda()[:8]
        residual=expert[:,:6]-wam[:,:6]
        raw_rms=residual.float().square().mean().sqrt()
        delta=residual*torch.clamp(torch.tensor(trust_rms,device='cuda')/(raw_rms+1e-8),max=1.0)
        a0=wam.clone();a1=wam.clone();a2=wam.clone()
        a1[:,:6]+=0.5*delta;a2[:,:6]+=delta
        state=torch.as_tensor(adapter.state_from_observation(r['obs']),device='cuda')
        scale,offset=r['scale'].cuda(),r['offset'].cuda()
        def energy(action,reps):
            ac=adapter.convert_torch(((action[None]-offset)/scale),state)
            return ranker.differentiable_energy(current_rep=reps[0],target_reps=reps[1:],
                ac_actions=ac.actions,ac_states=ac.states).mean()
        optimizer.zero_grad(set_to_none=True);started=time.monotonic()
        e0=energy(a0,real);e1=energy(a1,real);e2=energy(a2,real)
        loss01=F.softplus((e1-e0+margin)/temperature)
        loss12=F.softplus((e2-e1+margin)/temperature)
        loss=loss01+loss12
        if update:loss.backward()

        def correction(reps):
            probe=wam.detach().clone().requires_grad_(True)
            e=energy(probe,reps)
            g=torch.autograd.grad(e,probe,create_graph=False)[0]
            pred=-4.0*g[:,:6]
            prms=pred.float().square().mean().sqrt()
            pred=pred*torch.clamp(torch.tensor(trust_rms,device='cuda')/(prms+1e-8),max=1.0)
            cos=F.cosine_similarity(pred.flatten(),residual.flatten(),dim=0,eps=1e-8)
            return pred.detach(),float(cos.detach()),float(pred.float().square().mean().sqrt())
        real_pred,real_cos,real_rms=correction(real)
        online_pred,online_cos,online_rms=correction(online)
        clean_mse=F.mse_loss(wam[:,:6],expert[:,:6])
        corrected_mse=F.mse_loss(wam[:,:6]+online_pred,expert[:,:6])
        row=dict(window=index,loss=float(loss.detach()),e0=float(e0.detach()),e1=float(e1.detach()),e2=float(e2.detach()),
            margin01=float((e0-e1).detach()),margin12=float((e1-e2).detach()),
            segment01_ranked=bool((e1<e0).detach()),segment12_ranked=bool((e2<e1).detach()),
            path_ranked=bool(((e2<e1)&(e1<e0)).detach()),desired_raw_rms=float(raw_rms),
            real_cosine=real_cos,real_correction_rms=real_rms,
            online_cosine=online_cos,online_correction_rms=online_rms,
            clean_action_mse=float(clean_mse),corrected_action_mse=float(corrected_mse),seconds=time.monotonic()-started)
        if update:
            grads=[p.grad for p in params if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            gn=float(torch.nn.utils.clip_grad_norm_(params,1.0));row.update(predictor_grad_norm=gn,global_clip_active=bool(gn>1.0))
            optimizer.step()
        del r,real,online,wam,expert,residual,delta,a0,a1,a2,e0,e1,e2,loss,real_pred,online_pred
        gc.collect();torch.cuda.empty_cache();return row

    def evaluate(label):
        values=[]
        for j,index in enumerate(valids,1):
            row=one(index,False);row.update(split='val',eval_at=label,val_index=j);values.append(row)
            with metric_path.open('a') as f:f.write(json.dumps(row)+'\n')
            if j%10==0:emit('VAL_PROGRESS',eval_at=label,completed=j,
                path_accuracy=float(np.mean([x['path_ranked'] for x in values])),
                real_cosine=float(np.mean([x['real_cosine'] for x in values])),
                online_cosine=float(np.mean([x['online_cosine'] for x in values])))
        return values

    baseline=None if smoke else evaluate('step_0000');training=[]
    for step,index in enumerate(order[:2] if smoke else order,1):
        row=one(index,True);row.update(split='train',step=step);training.append(row)
        with metric_path.open('a') as f:f.write(json.dumps(row)+'\n')
        if step<=3 or step%10==0:emit('TRAIN_PROGRESS',step=step,**{k:v for k,v in row.items() if k not in ('split','step')})
    if smoke:
        summary=dict(passed=True,steps=2,source_checkpoint=str(SOURCE),metrics=training,peak_gib=torch.cuda.max_memory_allocated()/2**30)
    else:
        final=evaluate('step_0100')
        keys=('e0','margin01','margin12','segment01_ranked','segment12_ranked','path_ranked',
              'real_cosine','real_correction_rms','online_cosine','online_correction_rms','clean_action_mse','corrected_action_mse')
        def agg(xs):return {k:float(np.mean([x[k] for x in xs])) for k in keys}
        b,f=agg(baseline),agg(final)
        summary=dict(passed=True,steps=100,source_checkpoint=str(SOURCE),
            objective='one-sided WAM-to-expert trust-region path ranking on real future latent',
            trust_rms=trust_rms,margin=margin,temperature=temperature,
            baseline=b,final=f,change={k:f[k]-b[k] for k in keys},
            train_first10_loss=float(np.mean([x['loss'] for x in training[:10]])),
            train_last10_loss=float(np.mean([x['loss'] for x in training[-10:]])),
            clip_rate=float(np.mean([x['global_clip_active'] for x in training])),peak_gib=torch.cuda.max_memory_allocated()/2**30)
        payload=dict(predictor=predictor.state_dict(),optimizer=optimizer.state_dict(),step=100,
            source_checkpoint=str(SOURCE),objective=summary['objective'])
        tmp=runroot/'step_0100.tmp';torch.save(payload,tmp);os.replace(tmp,runroot/'step_0100.pt')
    (runroot/'summary.json').write_text(json.dumps(summary,indent=2));emit('SMOKE_COMPLETED' if smoke else 'PILOT_COMPLETED',**summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','train'],required=True);a=p.parse_args()
    try:main(a.phase=='smoke')
    except Exception:traceback.print_exc();sys.exit(1)
