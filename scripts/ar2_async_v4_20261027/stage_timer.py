"""Asynchronous CUDA-event spans and host call spans; no per-stage synchronize."""
import time,threading
import torch
class StageTimer:
 def __init__(self,device):
  self.device=device;self.host_origin=time.perf_counter();self.origin=torch.cuda.Event(enable_timing=True);self.origin.record();self.rows=[];self.lock=threading.Lock();self.sequence=0
 def begin(self,name,gpu=True):
  with self.lock:identifier=self.sequence;self.sequence+=1
  e=torch.cuda.Event(enable_timing=True) if gpu else None
  if e is not None:e.record()
  return dict(name=name,id=identifier,host_start_ms=(time.perf_counter()-self.host_origin)*1000,start=e)
 def end(self,row):
  end=torch.cuda.Event(enable_timing=True) if row['start'] is not None else None
  if end is not None:end.record()
  row.update(end=end,host_end_ms=(time.perf_counter()-self.host_origin)*1000)
  with self.lock:self.rows.append(row)
 def call(self,name,fn,gpu=True):
  row=self.begin(name,gpu)
  try:return fn()
  finally:self.end(row)
 def collect(self):
  # Caller must synchronize only once, at the complete iteration boundary.
  result=[];gpu={};host={}
  for row in sorted(self.rows,key=lambda r:r['id']):
   r={k:v for k,v in row.items() if k not in ['start','end']};r['host_call_ms']=r['host_end_ms']-r['host_start_ms']
   host[r['name']]=host.get(r['name'],0.)+r['host_call_ms']
   if row['start'] is not None:
    r['gpu_start_ms']=self.origin.elapsed_time(row['start']);r['gpu_end_ms']=self.origin.elapsed_time(row['end']);r['gpu_span_ms']=row['start'].elapsed_time(row['end']);gpu[r['name']]=gpu.get(r['name'],0.)+r['gpu_span_ms']
   result.append(r)
  return dict(intervals=result,gpu_span_ms=gpu,host_call_ms=host)
