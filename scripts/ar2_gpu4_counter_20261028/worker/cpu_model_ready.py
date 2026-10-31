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
 # All real model weights are loaded on CPU before vLLM is stopped.
 gate=OUT/'release_gpu4'
 deadline=time.monotonic()+1800
 while not gate.exists():
  if time.monotonic()>deadline:raise TimeoutError('CPU preload gate timeout; vLLM must remain running')
  time.sleep(.2)
 gpu='4';target=torch.device('cuda:0');started=time.perf_counter();torch.cuda.set_device(0);os.environ['MUJOCO_EGL_DEVICE_ID']='4'
 save('gpu_acquisition.json',dict(state='MOVING_PRELOADED_WEIGHTS',gpu=4,updated=time.strftime('%F %T')))
 model.to(device=target);ranker.encoder.to(device=target);ranker.predictor.to(device=target)
 stats={k:tuple(v.to(target) for v in values) for k,values in ranker.stats.items()};copies={}
 for w in ranker.task_weights.values():
  if id(w) not in copies:copies[id(w)]=w.to(target)
 torch.cuda.synchronize(target)
 model.device=target;ranker.device=target;ranker.stats=stats;ranker.task_weights={tid:copies[id(w)] for tid,w in ranker.task_weights.items()};kw['model_device']=str(target)
 save('gpu_acquisition.json',dict(state='GPU_READY',gpu=4,weight_transfer_s=time.perf_counter()-started,allocated_mib=torch.cuda.memory_allocated(target)/2**20));return kw
