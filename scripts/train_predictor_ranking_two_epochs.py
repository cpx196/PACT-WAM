"""Predictor-only real-future path ranking: 450 fixed windows x 2 epochs."""
import argparse, gc, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path('/data/pxchen/PACT-WAM');sys.path.insert(0,str(ROOT))
BASE=ROOT/'artifacts/libero10_predictor_action_oneepoch_20260930'
CACHE=ROOT/'artifacts/libero10_predictor_energy_ranking_pilot100_20260930'
OUT=ROOT/'artifacts/libero10_predictor_real_path_ranking_2epoch900_20260930'
SOURCE=ROOT/'checkpoints/vjepa2-ac-libero90-euler-fixed-rot180-step2000-inference.pt'

def emit(event,**kw):print(json.dumps(dict(time=time.strftime('%F %T'),event=event,**kw)),flush=True)

def selection():
    rows=json.loads((BASE/'windows.json').read_text())
    train=[i for i,r in enumerate(rows) if r['split']=='train'];val=[i for i,r in enumerate(rows) if r['split']=='val']
    assert len(train)==450 and len(val)==50
    return rows,train,val

def cache(phase):
    from scripts import train_predictor_energy_ranking_pilot as support
    support.selected_ids=selection
    if phase=='cache-real':support.cache_real_latents()
    else:support.cache_wam_actions()

def train(smoke=False,resume=None):
    from experiments.libero.vjepa2_ac_ranker import VJEPA2ACRanker,LiberoACAdapter
    from scripts.train_predictor_energy_ranking_pilot import configure_predictor
    rows,trainids,valids=selection();out=OUT/('smoke' if smoke else 'train900')
    out.mkdir(parents=True,exist_ok=bool(resume))
    order=sum([np.random.default_rng(239+epoch).permutation(trainids).tolist() for epoch in range(2)],[])
    config=dict(source_checkpoint=str(SOURCE),objective='real-future one-sided WAM-to-expert path ranking',
        encoder_frozen=True,trainable='action/state encoder, projection/norm, last four predictor blocks',
        predictor_dtype='float32',batch_size=1,epochs=2,steps=900,windows=500,train_windows=450,val_windows=50,
        trust_rms=.02,margin=5e-5,temperature=2e-4,heads_lr=1e-5,blocks_lr=3e-6,weight_decay=.04,grad_clip=1.,
        save_every=100,latent_loss_weight=0.,ac_steps=2,low_level_steps_per_ac_step=4,
        image_orientation='rot180 main-camera current and future',gripper='WAM 1-2*g; expert LIBERO executable unchanged')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    (out/'window_manifest.json').write_text(json.dumps(rows,indent=2))
    ranker=VJEPA2ACRanker(checkpoint_path=str(SOURCE),vjepa2_repo='/data/pxchen/vjepa2',device='cpu',dtype=torch.float32,max_ac_steps=2)
    del ranker.encoder;ranker.device=torch.device('cuda');p=ranker.predictor.cuda().eval()
    heads,blocks=configure_predictor(p);params=heads+blocks
    optimizer=torch.optim.AdamW([dict(params=heads,lr=1e-5),dict(params=blocks,lr=3e-6)],weight_decay=.04)
    adapter=LiberoACAdapter(low_level_steps_per_ac_step=4);startstep=0
    initial_weight=p.action_encoder.weight.detach().cpu().clone()
    if resume:
        ck=torch.load(resume,map_location='cpu',weights_only=False)
        assert ck['config']==config and ck['order']==order
        p.load_state_dict(ck['predictor']);optimizer.load_state_dict(ck['optimizer']);startstep=ck['step']
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        initial_weight=ck['initial_action_encoder_weight'];del ck
    def load(i):
        r=torch.load(BASE/'wam'/f'{i:04d}.pt',map_location='cpu',weights_only=False)
        return dict(real=torch.load(CACHE/'real_latent'/f'{i:04d}.pt',map_location='cpu',weights_only=True).cuda(),
            wam=torch.load(CACHE/'wam_action'/f'{i:04d}.pt',map_location='cpu',weights_only=True).cuda(),
            expert=r['expert'][:8].cuda(),scale=r['scale'].cuda(),offset=r['offset'].cuda(),
            state=torch.tensor(adapter.state_from_observation(r['obs']),device='cuda'))
    def energy(d,a,wam_format=True):
        raw=(a-d['offset'])/d['scale']
        if wam_format:raw=torch.cat((raw[:,:6],1-2*raw[:,6:7]),-1)
        ac=adapter.convert_torch(raw[None],d['state']);z=d['real']
        return ranker.differentiable_energy(current_rep=z[0],target_reps=z[1:],ac_actions=ac.actions,ac_states=ac.states).mean()
    def loss_for(d):
        residual=d['expert'][:,:6]-d['wam'][:,:6]
        delta=residual*torch.clamp(.02/(residual.square().mean().sqrt()+1e-8),max=1)
        a=[d['wam'].clone() for _ in range(3)];a[1][:,:6]+=.5*delta;a[2][:,:6]+=delta
        e=[energy(d,x) for x in a]
        loss=F.softplus((e[1]-e[0]+5e-5)/2e-4)+F.softplus((e[2]-e[1]+5e-5)/2e-4)
        return loss,dict(margin01=float((e[0]-e[1]).detach()),margin12=float((e[1]-e[2]).detach()),ranked=float(bool((e[0]>e[1])&(e[1]>e[2]))))
    fixedtrain=trainids[:16]
    def evaluate(step):
        report={}
        with torch.no_grad():
            for split,ids in [('fixed_train16',fixedtrain),('val50',valids)]:
                values=[]
                for i in ids:
                    d=load(i);loss,m=loss_for(d)
                    values.append(dict(window=i,ranking_loss=float(loss),expert_real_latent_l1=float(energy(d,d['expert'],False)),**m))
                report[split]={k:float(np.mean([v[k] for v in values])) for k in values[0] if k!='window'}
                (out/f'eval_{step:04d}_{split}.json').write_text(json.dumps(values,indent=2))
        with (out/'evaluations.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,metrics=report))+'\n')
        emit('EVALUATION',step=step,metrics=report);return report
    def save(step):
        dest=out/f'step_{step:04d}.pt';tmp=dest.with_suffix('.tmp')
        torch.save(dict(predictor=p.state_dict(),optimizer=optimizer.state_dict(),step=step,config=config,order=order,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),initial_action_encoder_weight=initial_weight),tmp)
        os.replace(tmp,dest);emit('CHECKPOINT_SAVED',step=step,path=str(dest))
    emit('TRAIN_READY',gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),resume_step=startstep,config=config)
    baseline=None if smoke or startstep else evaluate(0)
    metrics=[];started=time.monotonic();steps=2 if smoke else 900
    for j in range(startstep,steps):
        optimizer.zero_grad(set_to_none=True);d=load(order[j]);loss,m=loss_for(d);loss.backward()
        grads=[x.grad for x in params if x.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        gn=float(torch.nn.utils.clip_grad_norm_(params,1.));optimizer.step()
        row=dict(step=j+1,epoch=j//450+1,window=order[j],ranking_loss=float(loss.detach()),grad_norm=gn,clipped=gn>1,
            elapsed_seconds=time.monotonic()-started,**m);metrics.append(row)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if j<3 or (j+1)%10==0:emit('TRAIN_PROGRESS',**row)
        if not smoke and (j+1)%100==0:
            save(j+1);evaluate(j+1)
        del d,loss,grads
    update=float((p.action_encoder.weight.detach().cpu()-initial_weight).norm()/initial_weight.norm());assert update>0
    if smoke:
        summary=dict(passed=True,steps=2,relative_parameter_update=update,metrics=metrics)
    else:
        summary=dict(passed=True,steps=900,baseline=baseline,final=evaluate(900) if startstep==900 else None,
            relative_parameter_update=update,seconds=time.monotonic()-started)
        # The final checkpoint and per-window evaluation were already saved at step 900.
        summary['final']=json.loads((out/'evaluations.jsonl').read_text().splitlines()[-1])['metrics']
    (out/'summary.json').write_text(json.dumps(summary,indent=2));emit('SMOKE_COMPLETED' if smoke else 'TRAIN_COMPLETED',**summary)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['cache-real','cache-wam','smoke','train'],required=True)
    ap.add_argument('--resume');args=ap.parse_args()
    if args.phase.startswith('cache'):cache(args.phase)
    else:train(args.phase=='smoke',args.resume)
