"""Wait for an idle GPU, then compile and profile current AR2 in isolation."""
import fcntl,hashlib,json,os,shutil,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/data/pxchen/PACT-WAM');BASE=ROOT/'evaluate_results/white_mug_robot42_seed7_sameT5_20261025'
OUT=ROOT/'evaluate_results/ar2_compiled_latency_20261026';PY='/data/pxchen/envs/jepa-wam/bin/python';CHILD=None
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(name,x):
 p=OUT/name;q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(p)
def main():
 global CHILD
 OUT.mkdir(exist_ok=True);lock=(OUT/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'active_process.json').exists(), 'GPU worker already started'
 (OUT/'launcher.pid').write_text(str(os.getpid()));src=OUT/'source_snapshot';cache=OUT/'compiler_cache'
 assert src.is_dir() and cache.is_dir() and (OUT/'manifest.json').exists()
 since={}
 while True:
  busy=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).split())
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines();available=[];snapshot=[]
  for line in rows:
   i,u,f,v=[s.strip() for s in line.split(',')];ok=u not in busy and int(f)>=23000 and int(v)<5;snapshot.append(dict(gpu=int(i),free_mib=int(f),utilization=int(v),no_compute_process=u not in busy))
   if ok:available.append(i)
  now=time.monotonic();since={g:since.get(g,now) for g in available};ready=[g for g in available if now-since[g]>=10]
  save('launcher_progress.json',dict(state='WAITING_IDLE_GPU',updated=time.strftime('%F %T'),required_free_mib=23000,stable_seconds=10,gpus=snapshot))
  if ready:gpu=ready[0];break
  time.sleep(10)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,MUJOCO_EGL_DEVICE_ID=gpu,TZ='Asia/Shanghai',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',LIBERO_CONFIG_PATH='/data/pxchen/FastWAM/.libero_plus_config',DIFFSYNTH_MODEL_BASE_PATH='/data/pxchen/FastWAM/checkpoints',DIFFSYNTH_SKIP_DOWNLOAD='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',GUIDANCE_SIX_OUT=str(OUT),ACTION_WEIGHTS_REGISTRY=str(BASE/'weights_registry.json'),TORCHINDUCTOR_CACHE_DIR=str(cache),TRITON_CACHE_DIR=str(cache/'triton'),TORCHINDUCTOR_COMPILE_THREADS='2',TORCH_LOGS='recompiles,graph_breaks',PYTHONPATH=':'.join(map(str,[src,ROOT,ROOT/'src',Path('/data/pxchen/LIBERO-plus'),ROOT/'artifacts/jepawam_deploy_smoke_20261009/python_deps'])))
 env['LD_LIBRARY_PATH']='/data/pxchen/envs/jepa-wam/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
 for k in list(env):
  if k.startswith('LIBERO_WORKER_'):env.pop(k)
 old=json.loads((BASE/'weighted/command.json').read_text());replace={'EVALUATION.output_dir':str(OUT),'EVALUATION.task_id':'559'}
 args=[a.split('=',1)[0]+'='+replace[a.split('=',1)[0]] if a.split('=',1)[0] in replace else a for a in old[2:]]
 cmd=[PY,str(src/'bench_compiled.py'),*args];save('command.json',cmd)
 with (OUT/'run.log').open('w') as log:
  CHILD=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  save('active_process.json',dict(pid=CHILD.pid,launcher_pid=os.getpid(),gpu=int(gpu),started=time.strftime('%F %T')))
  save('launcher_progress.json',dict(state='COMPILING_AND_BENCHMARKING',gpu=int(gpu),pid=CHILD.pid,updated=time.strftime('%F %T')))
  code=CHILD.wait()
 assert code==0,code
 assert (OUT/'latency_summary.json').exists() and (OUT/'bottleneck_report.md').exists()
 save('launcher_completed.json',dict(returncode=0,completed=time.strftime('%F %T')))
 save('launcher_progress.json',dict(state='COMPLETED',updated=time.strftime('%F %T')))
if __name__=='__main__':
 try:main()
 except BaseException:
  if CHILD is not None and CHILD.poll() is None:CHILD.terminate()
  if OUT.exists():save('launcher_failure.json',dict(traceback=traceback.format_exc()))
  raise
