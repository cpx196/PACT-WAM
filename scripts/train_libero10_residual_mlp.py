"""Frozen JEPA features, direct demonstrated-action supervision for a residual MLP."""
import argparse, gc, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path('/data/pxchen/PACT-WAM');sys.path.insert(0,str(ROOT))
BASE=ROOT/'artifacts/libero10_predictor_action_oneepoch_20260930'
CACHE=ROOT/'artifacts/libero10_predictor_energy_ranking_pilot100_20260930'
DATA=ROOT/'artifacts/libero10_guidance_backward_smoke_20260929/data/libero_10'
SOURCE=ROOT/'checkpoints/vjepa2-ac-libero90-euler-fixed-rot180-step2000-inference.pt'
OUT=ROOT/'artifacts/libero10_mlp_residual_pilot_20261004'

def emit(event,**kw):print(json.dumps(dict(time=time.strftime('%F %T'),event=event,**kw)),flush=True)
def atomic_save(payload,path):
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path)

def prepare():
    import h5py
    from experiments.libero.vjepa2_ac_ranker import VJEPA2ACRanker,LiberoACAdapter
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'features.pt').exists(),'Feature output exists; use the existing cache explicitly.'
    rows=json.loads((BASE/'windows.json').read_text());assert len(rows)==500
    ranker=VJEPA2ACRanker(checkpoint_path=str(SOURCE),vjepa2_repo='/data/pxchen/vjepa2',device='cpu',dtype=torch.float32,max_ac_steps=2)
    del ranker.encoder;gc.collect();ranker.device=torch.device('cuda')
    ranker.predictor.cuda().eval().requires_grad_(False)
    adapter=LiberoACAdapter(low_level_steps_per_ac_step=4)
    features=[];max_label_error=0.;max_energy_error=0.;h5s={};started=time.monotonic()
    try:
        for start in range(0,500,8):
            pack=[]
            for i in range(start,min(start+8,500)):
                r=torch.load(BASE/'wam'/f'{i:04d}.pt',map_location='cpu',weights_only=False)
                assert r['row']==rows[i]
                if rows[i]['task_file'] not in h5s:h5s[rows[i]['task_file']]=h5py.File(DATA/rows[i]['task_file'],'r')
                group=h5s[rows[i]['task_file']]['data'][rows[i]['demo']];t=rows[i]['t']
                raw_expert=torch.tensor(np.asarray(group['actions'][t:t+8]),dtype=torch.float32)
                expert=raw_expert*r['scale']+r['offset']
                error=float((expert[:,:6]-r['expert'][:,:6]).abs().max());max_label_error=max(max_label_error,error);assert error<2e-6
                # Gripper labels are not used: HDF5 expert and WAM use different conventions.
                wam=torch.load(CACHE/'wam_action'/f'{i:04d}.pt',map_location='cpu',weights_only=True)
                z=torch.load(BASE/'latent'/f'{i:04d}.pt',map_location='cpu',weights_only=True)
                state=torch.tensor(adapter.state_from_observation(r['obs']))
                raw=(wam-r['offset'])/r['scale']
                executable=torch.cat((raw[:,:6],1-2*raw[:,6:7]),-1)
                ac=adapter.convert_torch(executable[None],state)
                assert wam.shape==(8,7) and state.shape==(7,) and z.shape==(3,256,1408)
                pack.append(dict(index=i,wam=wam,expert=expert[:,:6],state=state,z=z,ac=ac))
            z=torch.stack([x['z'] for x in pack]).cuda()
            actions=torch.cat([x['ac'].actions for x in pack]).cuda()
            states=torch.cat([x['ac'].states for x in pack]).cuda()
            with torch.no_grad():
                history=z[:,0];predictions=[]
                for k in range(2):
                    pred=ranker.predictor(history,actions[:,:k+1],states[:,:k+1])
                    pred=F.layer_norm(pred[:,-ranker.tokens_per_frame:],(pred.shape[-1],))
                    predictions.append(pred);history=torch.cat((history,pred),dim=1)
                predicted=torch.stack(predictions,dim=1)
                difference=z[:,1:]-predicted
                pooled=difference.mean(dim=(1,2)).cpu()
                energy=difference.abs().mean(dim=(1,2,3)).cpu()
                if start==0:
                    for j in range(2):
                        ref=ranker.differentiable_energy(current_rep=z[j,0],target_reps=z[j,1:],ac_actions=actions[j:j+1],ac_states=states[j:j+1])[0]
                        err=abs(float(ref)-float(energy[j]));max_energy_error=max(max_energy_error,err);assert err<2e-5
            for j,x in enumerate(pack):
                features.append(dict(index=x['index'],action=x['wam'],state=x['state'],difference=pooled[j],expert=x['expert'],energy=energy[j]))
            if len(features)%40==0 or len(features)==500:emit('FEATURE_CACHE_PROGRESS',completed=len(features),total=500,seconds=time.monotonic()-started)
    finally:
        for f in h5s.values():f.close()
    assert not any(p.requires_grad for p in ranker.predictor.parameters())
    payload={k:torch.stack([x[k] for x in features]) for k in ('action','state','difference','expert','energy')}
    payload['rows']=rows;payload['train_ids']=[i for i,r in enumerate(rows) if r['split']=='train'];payload['val_ids']=[i for i,r in enumerate(rows) if r['split']=='val']
    assert len(payload['train_ids'])==450 and len(payload['val_ids'])==50
    assert not set((r['task_file'],r['demo']) for r in rows if r['split']=='train')&set((r['task_file'],r['demo']) for r in rows if r['split']=='val')
    for k in ('action','state','difference','expert'):assert torch.isfinite(payload[k]).all()
    digest=hashlib.sha256()
    with SOURCE.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
    payload['metadata']=dict(source=str(SOURCE),source_sha256=digest.hexdigest(),
        current_and_wam_latent_cache=str(BASE/'latent'),wam_action_cache=str(CACHE/'wam_action'),
        feature_definition='concat normalized WAM action(56), observed xyz/EulerXYZ/closedness(7), mean over both future clips and spatial tokens of WAM latent minus autoregressive predicted latent(1408)',
        input_uses_expert=False,input_uses_real_future=False,gripper='1-2*g only for WAM to JEPA adapter; output gripper untouched',
        eef='observation quaternion to Euler XYZ via shared adapter',max_label_error=max_label_error,max_energy_error=max_energy_error,
        orientation='existing main-camera rot180 latent cache',peak_gib=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-started)
    atomic_save(payload,OUT/'features.pt');(OUT/'preflight.json').write_text(json.dumps(payload['metadata'],indent=2));emit('FEATURE_CACHE_COMPLETED',**payload['metadata'])

class ResidualMLP(nn.Module):
    def __init__(self,mean,std,use_jepa=True):
        super().__init__();self.register_buffer('mean',mean);self.register_buffer('std',std);self.use_jepa=use_jepa
        self.feature_projection=nn.Linear(1408,128)
        self.net=nn.Sequential(nn.Linear(56+7+128,256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,48))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,action,state,difference):
        x=torch.cat((action.flatten(1),state,difference),-1);x=(x-self.mean)/self.std
        latent=x[:,63:] if self.use_jepa else torch.zeros_like(x[:,63:])
        correction=self.net(torch.cat((x[:,:63],self.feature_projection(latent)),-1)).reshape(-1,8,6)
        return action[:,:,:6]+correction,correction

def run(smoke=False,resume=None):
    torch.manual_seed(239);np.random.seed(239)
    data=torch.load(OUT/'features.pt',map_location='cpu',weights_only=False)
    trainids=data['train_ids'];val=data['val_ids']
    flat=torch.cat((data['action'].flatten(1),data['state'],data['difference']),-1)
    mean=flat[trainids].mean(0);std=flat[trainids].std(0).clamp_min(1e-4)
    tensors={k:data[k].cuda() for k in ('action','state','difference','expert')}
    root=OUT/('smoke' if smoke else 'train');root.mkdir(parents=True,exist_ok=bool(resume))
    config=dict(source=data['metadata'],seed=239,epochs=50,batch_size=32,lr=3e-4,weight_decay=1e-4,
        hidden=[256,256],jepa_projection=128,loss='normalized action MSE on 8x6 demonstrated executable commands',
        rms_cap=None,residual_penalty=0.,output_zero_init=True,gripper_corrected=False,save_every_steps=100,
        frozen='FastWAM, JEPA encoder and original step2000 predictor; only cached forward features used',
        train_windows=450,val_windows=50,normalization='training-only input mean/std; output stays in WAM normalized first-six coordinates')
    (root/'config.json').write_text(json.dumps(config,indent=2));(root/'window_manifest.json').write_text(json.dumps(data['rows'],indent=2))
    results={}
    variants=['jepa'] if smoke else ['jepa','no_jepa']
    if resume:variants=[torch.load(resume,map_location='cpu',weights_only=False)['variant']]
    for variant in variants:
        torch.manual_seed(239)
        model=ResidualMLP(mean,std,variant=='jepa').cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        total=sum(p.numel() for p in model.parameters());start_epoch=0;start_offset=0;step=0;best=float('inf')
        dest=root/variant;dest.mkdir(exist_ok=True)
        def batch(ids):return [tensors[k][ids] for k in ('action','state','difference','expert')]
        @torch.no_grad()
        def evaluate(ids):
            model.eval();a,s,d,y=batch(ids);pred,c=model(a,s,d)
            per=(pred-y).square().mean((1,2));baseline=(a[:,:,:6]-y).square().mean((1,2))
            rms=c.square().mean((1,2)).sqrt()
            return dict(mse=float(per.mean()),baseline_mse=float(baseline.mean()),relative_mse_reduction=1-float(per.mean()/baseline.mean()),
                correction_rms_mean=float(rms.mean()),correction_rms_p95=float(torch.quantile(rms,.95)),
                better_windows=int((per<baseline).sum()),worse_windows=int((per>baseline).sum()),windows=len(ids))
        def save(name,epoch):
            atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=config,variant=variant,step=step,epoch=epoch,
                input_mean=mean,input_std=std,train_ids=trainids,val_ids=val,best_val_mse=best,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),
                next_epoch=step//15,next_batch_start=(step%15)*32),dest/name)
        if resume:
            ck=torch.load(resume,map_location='cpu',weights_only=False);assert ck['variant']==variant and ck['config']==config
            model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);step=ck['step']
            start_epoch=ck['next_epoch'];start_offset=ck['next_batch_start']
            torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);np.random.set_state(ck['numpy_rng']);best=ck['best_val_mse']
        baseline=dict(train=evaluate(trainids),val=evaluate(val));assert abs(baseline['train']['mse']-baseline['train']['baseline_mse'])<1e-8 if not resume else True
        emit('BASELINE',variant=variant,parameters=total,metrics=baseline)
        if smoke:
            ids=trainids[:16];before=evaluate(ids)['mse']
            for j in range(30):
                model.train();optimizer.zero_grad(set_to_none=True);a,s,d,y=batch(ids);pred,c=model(a,s,d);loss=F.mse_loss(pred,y);loss.backward()
                assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                optimizer.step();step+=1
            after=evaluate(ids)['mse'];assert after<before*.98,(before,after)
            save('smoke_checkpoint.pt',0)
            ck=torch.load(dest/'smoke_checkpoint.pt',map_location='cpu',weights_only=False)
            clone=ResidualMLP(mean,std,True).cuda();clone.load_state_dict(ck['model'])
            with torch.no_grad():assert torch.allclose(clone(*batch(ids)[:3])[0],model(*batch(ids)[:3])[0])
            results[variant]=dict(passed=True,initial_mse=before,final_mse=after,parameters=total,checkpoint_roundtrip=True)
            emit('SMOKE_COMPLETED',**results[variant]);continue
        started=time.monotonic();history=[]
        for epoch in range(start_epoch,50):
            order=np.random.default_rng(239+epoch).permutation(trainids).tolist();model.train()
            for start in range(start_offset if epoch==start_epoch else 0,len(order),32):
                ids=order[start:start+32];optimizer.zero_grad(set_to_none=True);a,s,d,y=batch(ids)
                pred,c=model(a,s,d);loss=F.mse_loss(pred,y);loss.backward()
                assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                optimizer.step();step+=1
                if step%100==0:save(f'step_{step:04d}.pt',epoch)
            metrics=dict(epoch=epoch+1,step=step,train=evaluate(trainids),val=evaluate(val),seconds=time.monotonic()-started)
            history.append(metrics)
            with (dest/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
            if metrics['val']['mse']<best:
                best=metrics['val']['mse'];save('best.pt',epoch+1)
            save('last.pt',epoch+1)
            if epoch<3 or (epoch+1)%5==0:emit('EPOCH_COMPLETED',variant=variant,**metrics)
        save(f'step_{step:04d}.pt',50)
        results[variant]=dict(parameters=total,epochs=50,steps=step,baseline=baseline,final=history[-1],best_val_mse=best,
            best_epoch=min(history,key=lambda m:m['val']['mse'])['epoch'],seconds=time.monotonic()-started)
        with torch.no_grad():
            a,s,d,y=batch(val);pred,c=model(a,s,d)
            atomic_save(dict(indices=val,original=a.cpu(),predicted_first6=pred.cpu(),correction=c.cpu(),expert_first6=y.cpu()),dest/'validation_predictions.pt')
        emit('VARIANT_COMPLETED',variant=variant,summary=results[variant])
    (root/'summary.json').write_text(json.dumps(results,indent=2));emit('ALL_COMPLETED',summary=results)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['prepare','smoke','train'],required=True);ap.add_argument('--resume')
    args=ap.parse_args()
    if args.phase=='prepare':prepare()
    else:run(args.phase=='smoke',args.resume)
