"""Create a fixed-input benchmark from the audited v6 source without changing models."""
import pathlib,shutil,json,hashlib
ROOT=pathlib.Path('/data/pxchen/PACT-WAM');HERE=pathlib.Path(__file__).resolve().parent
OLD=ROOT/'evaluate_results/ar2_async_timing_v6_20261027/source_snapshot';SRC=HERE/'worker';SRC.mkdir(exist_ok=True)
for p in OLD.iterdir():
 if p.is_file():shutil.copy2(p,SRC/p.name)
s=(SRC/'bench_async_timing.py').read_text()
s=s.replace("MODES=['no_guidance','full_gpu_verify0_async','full_gpu_verify0_prefill_first']","MODES=['no_guidance','full_gpu_verify0_serial','full_gpu_verify0_async','full_gpu_verify0_prefill_first']")
a=s.index(" status('PROFILING')");b=s.index(" pool.shutdown",a)
s=s[:a]+''' status('HARDWARE_COUNTER_CAPTURE')
 profile_rows=[]
 torch.cuda.synchronize()
 torch.cuda.cudart().cudaProfilerStart()
 try:
  for repeat in range(5):
   for mode in MODES:
    torch.cuda.nvtx.range_push('iteration/'+mode)
    try:row,_=iteration(mode)
    finally:torch.cuda.nvtx.range_pop()
    profile_rows.append(row)
 finally:
  torch.cuda.synchronize()
  torch.cuda.cudart().cudaProfilerStop()
 save('counter_iterations.json',profile_rows)
'''+s[b:]
(SRC/'bench_async_timing.py').write_text(s)
# NVTX ranges label existing stages on both streams. CUDA event timing remains unchanged.
s=(SRC/'stage_timer.py').read_text().replace("e=torch.cuda.Event(enable_timing=True) if gpu else None","torch.cuda.nvtx.range_push(name)\n  e=torch.cuda.Event(enable_timing=True) if gpu else None")
s=s.replace("with self.lock:self.rows.append(row)","with self.lock:self.rows.append(row)\n  torch.cuda.nvtx.range_pop()")
(SRC/'stage_timer.py').write_text(s)
s=(SRC/'cpu_model_ready.py').read_text();a=s.index(' since={};cooldown={}');s=s[:a]+''' # All real model weights are loaded on CPU before vLLM is stopped.
 gate=OUT/'release_gpu4'
 deadline=time.monotonic()+1800
 while not gate.exists():
  if time.monotonic()>deadline:raise TimeoutError('CPU preload gate timeout; vLLM must remain running')
  time.sleep(.2)
 gpu='4';target=torch.device('cuda:0');started=time.perf_counter();torch.cuda.set_device(0);os.environ['MUJOCO_EGL_DEVICE_ID']='0'
 save('gpu_acquisition.json',dict(state='MOVING_PRELOADED_WEIGHTS',gpu=4,updated=time.strftime('%F %T')))
 model.to(device=target);ranker.encoder.to(device=target);ranker.predictor.to(device=target)
 stats={k:tuple(v.to(target) for v in values) for k,values in ranker.stats.items()};copies={}
 for w in ranker.task_weights.values():
  if id(w) not in copies:copies[id(w)]=w.to(target)
 torch.cuda.synchronize(target)
 model.device=target;ranker.device=target;ranker.stats=stats;ranker.task_weights={tid:copies[id(w)] for tid,w in ranker.task_weights.items()};kw['model_device']=str(target)
 save('gpu_acquisition.json',dict(state='GPU_READY',gpu=4,weight_transfer_s=time.perf_counter()-started,allocated_mib=torch.cuda.memory_allocated(target)/2**20));return kw
''';(SRC/'cpu_model_ready.py').write_text(s)
for p in SRC.glob('*.py'):compile(p.read_text(),str(p),'exec')
(HERE/'worker_source_sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SRC.iterdir() if p.is_file()},indent=2))
print('Prepared worker source; Python syntax checks passed')
