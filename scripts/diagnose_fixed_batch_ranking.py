"""Bounded fixed-four-window overfit test; no production/checkpoint mutation."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path('/data/pxchen/PACT-WAM');sys.path.insert(0,str(ROOT))
from experiments.libero.vjepa2_ac_ranker import VJEPA2ACRanker,LiberoACAdapter
from scripts.train_predictor_energy_ranking_pilot import selected_ids,configure_predictor
BASE=ROOT/'artifacts/libero10_predictor_action_oneepoch_20260930'
CACHE=ROOT/'artifacts/libero10_predictor_energy_ranking_pilot100_20260930'
SOURCE=ROOT/'checkpoints/vjepa2-ac-libero90-euler-fixed-rot180-step2000-inference.pt'
OUT=ROOT/'artifacts/predictor_fixed_batch_debug_20260930'

def main(smoke):
    out=OUT/('smoke' if smoke else 'run100');out.mkdir(parents=True,exist_ok=False)
    _,order,valid=selected_ids();trainids=order[:4];valid=valid[:8]
    ranker=VJEPA2ACRanker(checkpoint_path=str(SOURCE),vjepa2_repo='/data/pxchen/vjepa2',device='cpu',dtype=torch.float32,max_ac_steps=2)
    del ranker.encoder;ranker.device=torch.device('cuda');p=ranker.predictor.cuda().eval()
    heads,blocks=configure_predictor(p);params=heads+blocks
    initial={k:v.detach().cpu().clone() for k,v in p.state_dict().items()}
    adapter=LiberoACAdapter(low_level_steps_per_ac_step=4);data={}
    for i in trainids+valid:
        r=torch.load(BASE/'wam'/f'{i:04d}.pt',map_location='cpu',weights_only=False)
        data[i]=dict(real=torch.load(CACHE/'real_latent'/f'{i:04d}.pt',map_location='cpu',weights_only=True).cuda(),
            wamrep=torch.load(BASE/'latent'/f'{i:04d}.pt',map_location='cpu',weights_only=True).cuda(),
            wam=torch.load(CACHE/'wam_action'/f'{i:04d}.pt',map_location='cpu',weights_only=True).cuda(),
            expert=r['expert'][:8].cuda(),scale=r['scale'].cuda(),offset=r['offset'].cuda(),
            state=torch.tensor(adapter.state_from_observation(r['obs']),device='cuda'))
    def energy(d,a,reps,wam_format=True):
        raw=(a-d['offset'])/d['scale']
        if wam_format:raw=torch.cat((raw[:,:6],1-2*raw[:,6:7]),dim=-1)
        ac=adapter.convert_torch(raw[None],d['state'])
        return ranker.differentiable_energy(current_rep=reps[0],target_reps=reps[1:],ac_actions=ac.actions,ac_states=ac.states).mean()
    def losses(d,target):
        residual=d['expert'][:,:6]-d['wam'][:,:6]
        delta=residual*torch.clamp(.02/(residual.square().mean().sqrt()+1e-8),max=1)
        actions=[d['wam'].clone() for _ in range(3)]
        actions[1][:,:6]+=.5*delta;actions[2][:,:6]+=delta
        reps=d['real'] if target=='real' else d['wamrep']
        e=[energy(d,a,reps) for a in actions]
        loss=F.softplus((e[1]-e[0]+5e-5)/2e-4)+F.softplus((e[2]-e[1]+5e-5)/2e-4)
        return loss,e
    def evaluate(ids,target):
        result=[]
        for i in ids:
            d=data[i]
            with torch.no_grad():
                loss,e=losses(d,target);latent=energy(d,d['expert'],d['real'],False)
            row=dict(window=i,loss=float(loss),margin01=float(e[0]-e[1]),margin12=float(e[1]-e[2]),ranked=float(e[0]>e[1]>e[2]),expert_latent=float(latent))
            for tag,reps in [('real',d['real']),('wam',d['wamrep'])]:
                a=d['wam'].clone().requires_grad_(True);g=torch.autograd.grad(energy(d,a,reps),a)[0]
                correction=-4*g[:,:6];correction*=torch.clamp(.02/(correction.square().mean().sqrt()+1e-8),max=1)
                residual=d['expert'][:,:6]-a[:,:6]
                row[tag+'_cos']=float(F.cosine_similarity(correction.flatten(),residual.flatten(),dim=0).detach())
                row[tag+'_mse']=float(F.mse_loss(a[:,:6]+correction,d['expert'][:,:6]).detach())
            row['clean_mse']=float(F.mse_loss(d['wam'][:,:6],d['expert'][:,:6]));result.append(row)
        return {k:float(np.mean([r[k] for r in result])) for k in result[0] if k!='window'}
    config=dict(source=str(SOURCE),trainids=trainids,validids=valid,steps=2 if smoke else 100,
        objectives=['real','wam'],gripper='production 1-2*g for WAM only',heads_lr=1e-5,blocks_lr=3e-6,
        clip=1.0,margin=5e-5,temperature=2e-4,weight_decay=0.04)
    (out/'config.json').write_text(json.dumps(config,indent=2));summary={}
    for target in config['objectives']:
        arm_started=time.monotonic()
        p.load_state_dict(initial);optimizer=torch.optim.AdamW([dict(params=heads,lr=1e-5),dict(params=blocks,lr=3e-6)],weight_decay=.04)
        before={s:evaluate(ids,target) for s,ids in [('train',trainids),('val',valid)]}
        print(json.dumps(dict(event='BASELINE',target=target,metrics=before)),flush=True)
        # Disentangle ranking versus ordinary latent gradients on the identical window.
        rankloss,_=losses(data[trainids[0]],target)
        latent=energy(data[trainids[0]],data[trainids[0]]['expert'],data[trainids[0]]['real'],False)
        gr=torch.autograd.grad(rankloss,params,allow_unused=True)
        gl=torch.autograd.grad(latent,params,allow_unused=True)
        nr=sum(float(g.square().sum()) for g in gr if g is not None)**.5
        nl=sum(float(g.square().sum()) for g in gl if g is not None)**.5
        dot=sum(float(a.mul(b).sum()) for a,b in zip(gr,gl) if a is not None and b is not None)
        gradient=dict(rank_norm=nr,latent_norm=nl,cosine=dot/(nr*nl+1e-20));del gr,gl
        logs=[]
        for step in range(1,config['steps']+1):
            optimizer.zero_grad(set_to_none=True);i=trainids[(step-1)%len(trainids)]
            loss,_=losses(data[i],target);loss.backward()
            assert any(x.grad is not None for x in params)
            assert all(torch.isfinite(x.grad).all() for x in params if x.grad is not None)
            gn=float(torch.nn.utils.clip_grad_norm_(params,1.0));optimizer.step()
            row=dict(target=target,step=step,window=i,loss=float(loss.detach()),grad_norm=gn,elapsed_seconds=time.monotonic()-arm_started)
            logs.append(row)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if step<=2 or step%20==0:print(json.dumps(dict(event='TRAIN',**row)),flush=True)
        after={s:evaluate(ids,target) for s,ids in [('train',trainids),('val',valid)]}
        delta=float((p.action_encoder.weight.detach().cpu()-initial['action_encoder.weight']).norm()/initial['action_encoder.weight'].norm())
        assert delta>0
        summary[target]=dict(before=before,after=after,component_gradients=gradient,clip_rate=float(np.mean([x['grad_norm']>1 for x in logs])),action_encoder_relative_update=delta)
        torch.save(dict(predictor=p.state_dict(),optimizer=optimizer.state_dict(),config=config,target=target),out/f'{target}_step_{config["steps"]:04d}.pt')
        print(json.dumps(dict(event='ARM_DONE',target=target,summary=summary[target])),flush=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('COMPLETED',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');main(ap.parse_args().smoke)
