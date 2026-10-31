"""Prepare frozen models on CPU, then acquire a card and immediately copy real weights."""
import os,time,json,subprocess,threading
from pathlib import Path
import torch
OUT=Path(os.environ['GUIDANCE_SIX_OUT'])
def save(name,x):
 p=OUT/name;q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def snapshot():
 lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20).splitlines()
 return [dict(zip(['gpu','free_mib','utilization'],map(lambda x:int(x.strip()),l.split(',')))) for l in lines]
def acquire(kw):
 model=kw['model'];ranker=kw['ac_ranker'];assert all(p.device.type=='cpu' for p in model.parameters());assert all(p.device.type=='cpu' for p in ranker.encoder.parameters());assert all(p.device.type=='cpu' for p in ranker.predictor.parameters())
 save('cpu_models_ready.json',dict(ready=True,time=time.strftime('%F %T'),wam_parameters=sum(p.numel() for p in model.parameters()),jepa_parameters=sum(p.numel() for p in ranker.encoder.parameters()),predictor_parameters=sum(p.numel() for p in ranker.predictor.parameters()),note='Actual pretrained frozen weights loaded on CPU before GPU selection; no dummy VRAM reservation.'))
 since={};cooldown={}
 while True:
  try:rows=snapshot()
  except (subprocess.SubprocessError,ValueError) as e:
   save('gpu_acquisition.json',dict(state='GPU_QUERY_RETRY',error=str(e),updated=time.strftime('%F %T')));time.sleep(1);continue
  now=time.monotonic();available=[str(r['gpu']) for r in rows if r['free_mib']>=18500 and now>=cooldown.get(str(r['gpu']),0)];since={g:since.get(g,now) for g in available};ready=[g for g in available if now-since[g]>=5]
  save('gpu_acquisition.json',dict(state='CPU_READY_WAITING_GPU',updated=time.strftime('%F %T'),gpus=rows,required_free_mib=18500,stable_seconds=5,poll_seconds=1))
  if not ready:time.sleep(1);continue
  gpu='7' if '7' in ready else max(ready,key=lambda g:next(r['free_mib'] for r in rows if str(r['gpu'])==g));target=torch.device('cuda:'+gpu);started=time.perf_counter();torch.cuda.set_device(int(gpu));os.environ['MUJOCO_EGL_DEVICE_ID']=gpu
  save('gpu_acquisition.json',dict(state='MOVING_PRELOADED_WEIGHTS',gpu=int(gpu),updated=time.strftime('%F %T')))
  try:
   model.to(device=target);ranker.encoder.to(device=target);ranker.predictor.to(device=target)
   stats={k:tuple(v.to(target) for v in values) for k,values in ranker.stats.items()};copies={}
   for w in ranker.task_weights.values():
    if id(w) not in copies:copies[id(w)]=w.to(target)
   task_weights={tid:copies[id(w)] for tid,w in ranker.task_weights.items()};torch.cuda.synchronize(target)
  except torch.OutOfMemoryError as e:
   model.to('cpu');ranker.encoder.to('cpu');ranker.predictor.to('cpu');torch.cuda.empty_cache();cooldown[gpu]=time.monotonic()+60;since={};save('gpu_migration_oom.json',dict(gpu=int(gpu),error=str(e),time=time.strftime('%F %T')));continue
  model.device=target;ranker.device=target;ranker.stats=stats;ranker.task_weights=task_weights;kw['model_device']=str(target)
  save('gpu_acquisition.json',dict(state='GPU_READY',gpu=int(gpu),weight_transfer_s=time.perf_counter()-started,updated=time.strftime('%F %T'),allocated_mib=torch.cuda.memory_allocated(target)/2**20));return kw
