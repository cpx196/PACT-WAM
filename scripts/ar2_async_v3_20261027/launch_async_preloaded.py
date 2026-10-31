"""Keep actual frozen models ready on CPU while waiting for any GPU."""
import os,json,time,subprocess,shutil,fcntl,threading,traceback,hashlib
from pathlib import Path
from launch_async_timing import ROOT,BASE,PREVIOUS,OUT,PY,save,status,monitor_resources
CHILD=None

def main():
 global CHILD
 OUT.mkdir(exist_ok=True);lock=(OUT/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'launcher_completed.json').exists();assert json.loads((OUT/'cpu_preflight.json').read_text())['passed']
 (OUT/'launcher.pid').write_text(str(os.getpid()));src=OUT/'source_snapshot';src.mkdir(exist_ok=True)
 for name in ['bench_async_timing.py','stage_timer.py','async_guidance_bridge.py','cpu_model_ready.py','launch_async_timing.py','launch_async_preloaded.py']:shutil.copy2(Path(__file__).parent/name,src/name)
 m=json.loads((OUT/'manifest.json').read_text());m['startup_mode']='Models prepared on CPU first; choose GPU after weights ready, then copy weights immediately';m['source_sha256'].update({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir() if p.is_file()});save('manifest.json',m)
 cache=PREVIOUS/'compiler_cache';attempt=max([int(p.name.split('_')[1]) for p in (OUT/'failed_attempts').glob('attempt_*') if p.name.split('_')[1].isdigit()],default=0)
 while True:
  attempt+=1
  env=os.environ.copy()
  for k in ['CUDA_VISIBLE_DEVICES','ASYNC_TIMING_FORCE_GPU']:env.pop(k,None)
  env.update(ASYNC_PRELOAD_CPU='1',TZ='Asia/Shanghai',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',LIBERO_CONFIG_PATH='/data/pxchen/FastWAM/.libero_plus_config',DIFFSYNTH_MODEL_BASE_PATH='/data/pxchen/FastWAM/checkpoints',DIFFSYNTH_SKIP_DOWNLOAD='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',GUIDANCE_SIX_OUT=str(OUT),ACTION_WEIGHTS_REGISTRY=str(BASE/'weights_registry.json'),TORCHINDUCTOR_CACHE_DIR=str(cache),TRITON_CACHE_DIR=str(cache/'triton'),TORCHINDUCTOR_COMPILE_THREADS='2',TORCH_LOGS='recompiles,graph_breaks',PYTHONPATH=':'.join(map(str,[src,ROOT,ROOT/'src',Path('/data/pxchen/LIBERO-plus'),ROOT/'artifacts/jepawam_deploy_smoke_20261009/python_deps'])))
  env['LD_LIBRARY_PATH']='/data/pxchen/envs/jepa-wam/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
  for k in list(env):
   if k.startswith('LIBERO_WORKER_'):env.pop(k)
  old=json.loads((BASE/'weighted/command.json').read_text());args=[('EVALUATION.output_dir='+str(OUT)) if a.startswith('EVALUATION.output_dir=') else ('EVALUATION.task_id=559') if a.startswith('EVALUATION.task_id=') else a for a in old[2:]];cmd=[PY,str(src/'bench_async_timing.py'),*args];save('command.json',cmd)
  stop=threading.Event();watcher=None;gpu=None
  with (OUT/'run.log').open('w') as log:
   CHILD=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   started=time.strftime('%F %T');status('PRELOADING_CPU',pid=CHILD.pid,attempt=attempt);save('active_process.json',dict(pid=CHILD.pid,launcher_pid=os.getpid(),gpu=None,attempt=attempt,started=started,startup_mode='cpu_preload'))
   while CHILD.poll() is None:
    q=OUT/'gpu_acquisition.json';acq=json.loads(q.read_text()) if q.exists() else {};state=acq.get('state','PRELOADING_CPU')
    if 'gpu' in acq and watcher is None:
     gpu=str(acq['gpu']);watcher=threading.Thread(target=monitor_resources,args=(stop,gpu,attempt),daemon=True);watcher.start()
    progress=json.loads((OUT/'progress.json').read_text()) if (OUT/'progress.json').exists() else {}
    status(state if state!='GPU_READY' else 'RUNNING',pid=CHILD.pid,attempt=attempt,gpu=int(gpu) if gpu is not None else None,gpu_acquisition=acq,worker_progress=progress)
    save('active_process.json',dict(pid=CHILD.pid,launcher_pid=os.getpid(),gpu=int(gpu) if gpu is not None else None,attempt=attempt,started=started,startup_mode='cpu_preload'));time.sleep(2)
   code=CHILD.returncode;stop.set()
   if watcher is not None:watcher.join(timeout=15)
  if code==0:
   assert (OUT/'latency_summary.json').exists() and json.loads((OUT/'progress.json').read_text())['state']=='COMPLETED'
   save('launcher_completed.json',dict(returncode=0,completed=time.strftime('%F %T'),gpu=int(gpu),attempt=attempt));status('COMPLETED',gpu=int(gpu),attempt=attempt);return
  error=(OUT/'failure.json').read_text() if (OUT/'failure.json').exists() else (OUT/'run.log').read_text(errors='replace')[-25000:]
  if 'out of memory' not in error.lower():status('FAILED_NEEDS_REPAIR',attempt=attempt,returncode=code);raise RuntimeError('Non-OOM failure; inspect failure.json')
  a=OUT/'failed_attempts'/f'attempt_{attempt:03d}';a.mkdir(parents=True)
  for q in list(OUT.iterdir()):
   if q.is_file() and q.suffix in ['.json','.jsonl','.log','.npy'] and q.name not in ['manifest.json','launcher_progress.json','gpu_resource_samples.jsonl','cpu_preflight.json','cpu_aot_bridge_check.json']:shutil.move(str(q),a/q.name)
  CHILD=None
if __name__=='__main__':
 try:main()
 except BaseException:
  if CHILD is not None and CHILD.poll() is None:CHILD.terminate()
  if OUT.exists():save('launcher_failure.json',dict(traceback=traceback.format_exc()))
  raise
