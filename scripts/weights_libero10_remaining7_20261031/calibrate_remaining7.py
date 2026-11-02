"""No training/rollouts: task-specific 200-state x32-demo-action variance."""
from wand.api import library as _wand_library
import hashlib,json,os,time
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from fastwam_ar2_eval import ARPredictor
from vectorized_ac_attention import enable_vectorized
from weighted_ar2_ranker import variance_weights
ROOT=Path('/data/pxchen/PACT-WAM')
OUT=Path(os.environ['ACTION_WEIGHTS_OUT'])
DATA=ROOT/'artifacts/predictor21_joint20_stride20_20261011'
PRED=ROOT/'artifacts/fastwam_predictor21_ar2_full10epoch_bs16_20261014/step19690.pt'
OLD_REGISTRY=ROOT/'evaluate_results/feedback_weights_6task_seed7_sameT5_20261021/offline/weights_registry.json'
old=json.loads(OLD_REGISTRY.read_text())
existing={x['task'] for x in old['tasks']}
all_rows=json.loads((DATA/'index.json').read_text())
TASKS={task:[] for task in sorted({r['task'] for r in all_rows if r['suite']=='libero10' and r['split']=='train'}-existing)}
assert len(TASKS)==7


def save(name,obj):
 p=OUT/name;q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2));q.replace(p)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 assert old['predictor_sha256']==sha(PRED)
 for item in old['tasks']:assert sha(item['path'])==item['sha256']
 OUT.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(4);torch.manual_seed(7)
 rows=json.loads((DATA/'index.json').read_text());groups={}
 for task,cases in TASKS.items():
  rs=[r for r in rows if r['suite']=='libero10' and r['task']==task and r['split']=='train']
  assert len(rs)==45 and all((DATA/r['file']).exists() for r in rs)
  groups[task]=rs
 stats_path=ROOT/'artifacts/fastwam_predictor21_ar2_stride20_20261014/train_only_stats.json'
 stats=json.loads(stats_path.read_text());pred_sha=sha(PRED)
 save('config.json',dict(case_set='libero10_remaining7_base_task_weights',method='FeedbackWM Eq14/15, full per-element variance',
  states_per_task=200,actions_per_state=32,beta=1.,epsilon=1e-8,seed=7,
  weight_shape=[2,576,1024],predictor=str(PRED),predictor_sha256=pred_sha,normalization_sha256=sha(stats_path),
  input_contract=json.loads((DATA/'contract.json').read_text()),
  precision='Native deployment FP32 Predictor, no autocast, batch1',
  horizons='For each fixed current state, vary full demonstrated 8-step chunks; compute P1 and AR P2 for each candidate. Each horizon gets its own per-element variance.',
  loss='Keep existing L1; borrow paper weighting only. No online feedback, no expert-memory target replacement, no parameter training.',
  split='Only 45 train demos/task; no validation/test rollout data',task_map=TASKS))
 model=ARPredictor();payload=torch.load(PRED,map_location='cpu',weights_only=False,mmap=True)
 assert payload['step']==19690
 model.load_state_dict(payload['model'],strict=True);del payload
 model.cuda().eval().requires_grad_(False);enable_vectorized(model)
 norm={k:(torch.tensor(stats[k]['mean'],device='cuda'),torch.tensor(stats[k]['std'],device='cuda').clamp_min(1e-6)) for k in ('action','state')}
 cache=OrderedDict()
 def shard(file):
  if file not in cache:
   cache[file]=torch.load(DATA/file,map_location='cpu',weights_only=False,mmap=True)
   if len(cache)>8:cache.popitem(last=False)
  cache.move_to_end(file);return cache[file]
 registry=[];started=time.time()
 for task,rs in groups.items():
  folder=OUT/task;folder.mkdir(exist_ok=True)
  rng=np.random.default_rng(7)
  # Round-robin across demos, with evenly distributed per-demo temporal samples.
  state_refs=[]
  positions={r['file']:np.linspace(0,r['windows']-1,min(5,r['windows']),dtype=int).tolist() for r in rs}
  for j in range(5):
   for r in rs:
    if j<len(positions[r['file']]):state_refs.append((r['file'],positions[r['file']][j]))
  state_refs=state_refs[:200];assert len(state_refs)==200 and len(set(state_refs))==200
  pool=[(r['file'],j) for r in rs for j in range(r['windows'])]
  candidate_refs=[[pool[int(i)] for i in rng.choice(len(pool),32,replace=False)] for _ in state_refs]
  manifest=dict(task=task,case_ids=TASKS[task],state_refs=state_refs,candidate_refs=candidate_refs)
  (folder/'sampling_manifest.json').write_text(json.dumps(manifest,indent=2))
  accumulator=torch.zeros((2,576,1024),dtype=torch.float64)
  tick=time.time()
  with torch.inference_mode():
   for n,((file,j),candidates) in enumerate(zip(state_refs,candidate_refs)):
    d=shard(file);z=d['current_latent'][j:j+1].float().cuda();s=d['state'][j:j+1].float().cuda()
    sm,sst=norm['state'];s=(s-sm)/sst
    predictions=[[],[]]
    for f,k in candidates:
     a=shard(f)['raw_actions'][k:k+1,:8].float().cuda();am,ast=norm['action'];a=(a-am)/ast
     p1=model(z,a[:,:4],s,0);p2=model(p1,a[:,4:8],s,1)
     assert p1.shape==p2.shape==(1,576,1024)
     predictions[0].append(p1);predictions[1].append(p2)
    variance=torch.stack([torch.cat(p).var(0,unbiased=False) for p in predictions])
    assert torch.isfinite(variance).all()
    accumulator+=variance.double().cpu()
    del predictions,variance,p1,p2
    if n==0 or (n+1)%10==0:
     elapsed=time.time()-tick
     save('progress.json',dict(state='CALIBRATING',task=task,completed_tasks=len(registry),states_done=n+1,states_total=200,elapsed_task_s=elapsed,estimated_remaining_task_s=elapsed/(n+1)*(199-n),updated=time.strftime('%F %T')))
     print(task,n+1,'/200',round(elapsed,1),'seconds',flush=True)
  variance=(accumulator/200).float();weights=variance_weights(variance)
  assert tuple(weights.shape)==(2,576,1024) and torch.isfinite(weights).all() and (weights>=0).all()
  assert torch.allclose(weights.mean((1,2)),torch.ones(2),atol=1e-5)
  path=folder/'weights.pt'
  temporary=path.with_suffix('.pt.tmp')
  torch.save(dict(weights=weights,variance=variance,beta=1.,epsilon=1e-8,task=task,predictor_sha256=pred_sha),temporary)
  temporary.replace(path)
  item=dict(task=task,case_ids=TASKS[task],path=str(path),sha256=sha(path),elapsed_s=time.time()-tick,
    weights_min=float(weights.min()),weights_max=float(weights.max()),weights_mean_per_horizon=weights.mean((1,2)).tolist())
  registry.append(item);save('weights_registry.json',dict(predictor_sha256=pred_sha,tasks=registry,complete=len(registry)==len(TASKS)))
  save('progress.json',dict(state='TASK_COMPLETED',completed_tasks=len(registry),total_tasks=len(TASKS),updated=time.strftime('%F %T')))
 save('progress.json',dict(state='COMPLETED',completed_tasks=len(TASKS),total_tasks=len(TASKS),elapsed_s=time.time()-started,updated=time.strftime('%F %T')))

if __name__=='__main__':main()
