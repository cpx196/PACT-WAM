"""Watch every GPU; reuse caches and retry another card after CUDA OOM."""
import os,json,time,hashlib,subprocess,shutil,fcntl,traceback,threading
from pathlib import Path
ROOT=Path('/data/pxchen/PACT-WAM');PY='/data/pxchen/envs/jepa-wam/bin/python'
BASE=ROOT/'evaluate_results/white_mug_robot42_seed7_sameT5_20261025'
PREVIOUS=ROOT/'evaluate_results/ar2_compiled_latency_20261026'
OUT=Path(os.environ.get('ASYNC_TIMING_OUT',str(ROOT/'evaluate_results/ar2_async_timing_v6_20261027')))
REQUIRED=19000;STABLE=5;POLL=1;CHILD=None

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,data):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(data,indent=2,ensure_ascii=False));q.replace(p)
def status(state,**kw):save('launcher_progress.json',dict(state=state,updated=time.strftime('%F %T'),**kw))
def snapshots():
 lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
 return [dict(zip(['gpu','free_mib','utilization'],[int(v.strip()) for v in line.split(',')])) for line in lines]
def choose(cooldown):
 since={}
 while True:
  rows=snapshots();now=time.monotonic();eligible=[str(x['gpu']) for x in rows if x['free_mib']>=REQUIRED and now>=cooldown.get(str(x['gpu']),0)]
  since={g:since.get(g,now) for g in eligible};ready=[g for g in eligible if now-since[g]>=STABLE]
  status('WAITING_GPU',gpu_scope='all',required_free_mib=REQUIRED,stable_seconds=STABLE,poll_seconds=POLL,gpus=rows,shared_gpu_allowed=True)
  if ready:return max(ready,key=lambda g:next(x['free_mib'] for x in rows if str(x['gpu'])==g))
  time.sleep(POLL)
def monitor_resources(stop,gpu,attempt):
 with (OUT/'gpu_resource_samples.jsonl').open('a') as f:
  while not stop.is_set():
   try:
    gpu_state=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True,timeout=10).strip()
    f.write(json.dumps(dict(time=time.strftime('%F %T'),attempt=attempt,gpu=int(gpu),state=gpu_state,processes=processes))+'\n');f.flush()
   except Exception as e:f.write(json.dumps(dict(error=str(e),time=time.strftime('%F %T')))+'\n');f.flush()
   stop.wait(5)
def main():
 global CHILD
 OUT.mkdir(exist_ok=True);lock=(OUT/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'launcher_completed.json').exists(),'Already completed'
 assert json.loads((OUT/'cpu_preflight.json').read_text())['passed'], 'CPU preflight not passed'
 (OUT/'launcher.pid').write_text(str(os.getpid()));src=OUT/'source_snapshot';src.mkdir(exist_ok=True)
 for p in (PREVIOUS/'source_snapshot').iterdir():
  if p.is_file() and p.suffix=='.py':shutil.copy2(p,src/p.name)
 for name in ['bench_async_timing.py','async_guidance_bridge.py','stage_timer.py','launch_async_timing.py']:shutil.copy2(ROOT/'scripts'/name,src/name)
 cache=PREVIOUS/'compiler_cache';assert cache.is_dir()
 assert sha(ROOT/'experiments/libero/eval_libero_single.py')==json.loads((BASE/'manifest.json').read_text())['native_sha256']
 save('manifest.json',dict(created=time.strftime('%F %T'),task_id=559,seed=7,fixed_T5_sha256=json.loads((BASE/'text_cache_manifest.json').read_text())['sha256'],previous_results=str(PREVIOUS),compiler_cache=str(cache),cache_policy='Reuse existing cache without deletion; new split graphs may compile. Separate warmup from steady state.',queue=dict(all_gpus=True,required_free_mib=REQUIRED,utilization_requirement=None,stable_seconds=STABLE,poll_seconds=POLL),requested_schedule='current JEPA on auxiliary stream during video; prefix decode + future JEPA during actions; predictor forward/backward on auxiliary stream with action-ready and gradient-done events',variants=['no_guidance','reference_joint_late_main','split_late_main','split_early_main','split_early_aux_guidance'],samples_per_variant=20,no_environment_rollouts=True,unchanged_success_rate_studies=True,source_sha256={p.name:sha(p) for p in src.iterdir() if p.is_file()}))
 cooldown={};attempt=0
 while True:
  gpu=choose(cooldown);attempt+=1
  save('selected_gpu_context.json',dict(gpu=int(gpu),attempt=attempt,time=time.strftime('%F %T'),snapshot=subprocess.check_output(['nvidia-smi'],text=True,timeout=10)))
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,MUJOCO_EGL_DEVICE_ID=gpu,TZ='Asia/Shanghai',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',LIBERO_CONFIG_PATH='/data/pxchen/FastWAM/.libero_plus_config',DIFFSYNTH_MODEL_BASE_PATH='/data/pxchen/FastWAM/checkpoints',DIFFSYNTH_SKIP_DOWNLOAD='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',GUIDANCE_SIX_OUT=str(OUT),ACTION_WEIGHTS_REGISTRY=str(BASE/'weights_registry.json'),TORCHINDUCTOR_CACHE_DIR=str(cache),TRITON_CACHE_DIR=str(cache/'triton'),TORCHINDUCTOR_COMPILE_THREADS='2',TORCH_LOGS='recompiles,graph_breaks',PYTHONPATH=':'.join(map(str,[src,ROOT,ROOT/'src',Path('/data/pxchen/LIBERO-plus'),ROOT/'artifacts/jepawam_deploy_smoke_20261009/python_deps'])))
  env['LD_LIBRARY_PATH']='/data/pxchen/envs/jepa-wam/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
  for key in list(env):
   if key.startswith('LIBERO_WORKER_'):env.pop(key)
  old=json.loads((BASE/'weighted/command.json').read_text());args=[('EVALUATION.output_dir='+str(OUT)) if a.startswith('EVALUATION.output_dir=') else ('EVALUATION.task_id=559') if a.startswith('EVALUATION.task_id=') else a for a in old[2:]]
  cmd=[PY,str(src/'bench_async_timing.py'),*args];save('command.json',cmd)
  stop=threading.Event();watcher=threading.Thread(target=monitor_resources,args=(stop,gpu,attempt),daemon=True)
  with (OUT/'run.log').open('w') as log:
   CHILD=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   save('active_process.json',dict(pid=CHILD.pid,launcher_pid=os.getpid(),gpu=int(gpu),attempt=attempt,started=time.strftime('%F %T')));status('RUNNING',gpu=int(gpu),pid=CHILD.pid,attempt=attempt);watcher.start();code=CHILD.wait();stop.set();watcher.join(timeout=15)
  if code==0:
   assert (OUT/'latency_summary.json').exists() and json.loads((OUT/'progress.json').read_text())['state']=='COMPLETED'
   save('launcher_completed.json',dict(returncode=0,completed=time.strftime('%F %T'),attempt=attempt,gpu=int(gpu)));status('COMPLETED',gpu=int(gpu),attempt=attempt);return
  error=(OUT/'failure.json').read_text() if (OUT/'failure.json').exists() else (OUT/'run.log').read_text(errors='replace')[-25000:]
  if 'out of memory' not in error.lower():
   status('FAILED_NEEDS_REPAIR',gpu=int(gpu),attempt=attempt,returncode=code);raise RuntimeError('Non-OOM failure; see failure.json and run.log. No blind software retries.')
  archive=OUT/'failed_attempts'/f'attempt_{attempt:03d}';archive.mkdir(parents=True)
  for p in list(OUT.iterdir()):
   if p.is_file() and p.suffix in ['.json','.jsonl','.log','.npy'] and p.name not in ['manifest.json','launcher_progress.json','gpu_resource_samples.jsonl','cpu_preflight.json']:shutil.move(str(p),archive/p.name)
  cooldown[gpu]=time.monotonic()+60;status('RETRY_AFTER_OOM',previous_gpu=int(gpu),attempt=attempt);CHILD=None
if __name__=='__main__':
 try:main()
 except BaseException:
  if CHILD is not None and CHILD.poll() is None:CHILD.terminate()
  if OUT.exists():save('launcher_failure.json',dict(traceback=traceback.format_exc()))
  raise
