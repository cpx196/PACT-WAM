"""Read-only checkpoint audit using the production gripper conversion."""
import json,sys,time,gc
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
ROOT=Path('/data/pxchen/PACT-WAM');sys.path.insert(0,str(ROOT))
from experiments.libero.vjepa2_ac_ranker import VJEPA2ACRanker,LiberoACAdapter
BASE=ROOT/'artifacts/libero10_predictor_action_oneepoch_20260930'
CACHE=ROOT/'artifacts/libero10_predictor_energy_ranking_pilot100_20260930'
OUT=ROOT/'artifacts/predictor_latent_contract_audit_20260930';OUT.mkdir(exist_ok=True)
SOURCE=ROOT/'checkpoints/vjepa2-ac-libero90-euler-fixed-rot180-step2000-inference.pt'
ranker=VJEPA2ACRanker(checkpoint_path=str(SOURCE),vjepa2_repo='/data/pxchen/vjepa2',device='cpu',dtype=torch.float32,max_ac_steps=2)
del ranker.encoder;ranker.device=torch.device('cuda');ranker.predictor.cuda().eval().requires_grad_(False)
adapter=LiberoACAdapter();records=[]
rows=json.loads((BASE/'windows.json').read_text());ids=[i for i,r in enumerate(rows) if r['split']=='val']
assert len(ids)==50
sources=[('base',SOURCE),('symmetric',CACHE/'train100/step_0100.pt'),('path',ROOT/'artifacts/libero10_predictor_path_ranking_pilot100_20260930/train100/step_0100.pt')]
for label,path in sources:
 ck=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
 sd=ranker._clean_state_dict(ck['predictor']);ranker.predictor.load_state_dict(sd,strict=True);del ck,sd;gc.collect()
 for j,i in enumerate(ids):
  r=torch.load(BASE/'wam'/f'{i:04d}.pt',map_location='cpu',weights_only=False)
  real=torch.load(CACHE/'real_latent'/f'{i:04d}.pt',weights_only=True).cuda()
  wamrep=torch.load(BASE/'latent'/f'{i:04d}.pt',weights_only=True).cuda()
  wam=torch.load(CACHE/'wam_action'/f'{i:04d}.pt',weights_only=True).cuda()
  expert=r['expert'].cuda();scale=r['scale'].cuda();offset=r['offset'].cuda()
  state=torch.tensor(adapter.state_from_observation(r['obs']),device='cuda')
  def energy(a,reps,is_wam):
   raw=(a-offset)/scale
   if is_wam:raw=torch.cat((raw[:,:6],1-2*raw[:,6:7]),-1)
   ac=adapter.convert_torch(raw[None],state)
   return ranker.differentiable_energy(current_rep=reps[0],target_reps=reps[1:],ac_actions=ac.actions,ac_states=ac.states).mean()
  with torch.no_grad():
   expert_error=float(energy(expert,real,False));base_real=float(energy(wam,real,True))
  residual=expert[:,:6]-wam[:,:6];row=dict(checkpoint=label,window=i,expert_real_latent_l1=expert_error,wam_real_latent_l1=base_real)
  for target,reps in [('real',real),('wam',wamrep)]:
   a=wam.detach().clone().requires_grad_(True);e=energy(a,reps,True)
   grad=torch.autograd.grad(e,a)[0];delta=-4*grad[:,:6]
   delta*=torch.clamp(.02/(delta.square().mean().sqrt()+1e-8),max=1)
   row[target+'_cosine']=float(F.cosine_similarity(delta.flatten(),residual.flatten(),dim=0))
   row[target+'_corrected_mse']=float(F.mse_loss(wam[:,:6]+delta,expert[:,:6]))
   row[target+'_gradient_rms']=float(grad[:,:6].square().mean().sqrt())
   if label=='base' and j<3:
    direction=residual/(residual.square().mean().sqrt()+1e-8)
    v=torch.zeros_like(a);v[:,:6]=direction
    with torch.no_grad():fd=float((energy(a+.001*v,reps,True)-energy(a-.001*v,reps,True))/.002)
    row[target+'_finite_difference']=fd;row[target+'_autograd_directional']=float((grad*v).sum())
  row['clean_mse']=float(F.mse_loss(wam[:,:6],expert[:,:6]));records.append(row)
  if (j+1)%10==0:print(json.dumps(dict(checkpoint=label,completed=j+1)),flush=True)
 summary={k:float(np.mean([x[k] for x in records if x['checkpoint']==label])) for k in row if k not in ('checkpoint','window')}
 print(json.dumps(dict(checkpoint=label,summary=summary)),flush=True)
(OUT/'records.json').write_text(json.dumps(records,indent=2))
summary={label:{k:float(np.mean([x[k] for x in records if x['checkpoint']==label])) for k in ('expert_real_latent_l1','wam_real_latent_l1','real_cosine','wam_cosine','real_corrected_mse','wam_corrected_mse','clean_mse')} for label,_ in sources}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
