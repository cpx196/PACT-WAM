import fcntl,hashlib,json,os,shutil,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/data/pxchen/PACT-WAM')
OUT=ROOT/'evaluate_results/t4_repro_audit_20261021'
CHILD=None
OWNER=False

def save(name,x):
 p=OUT/name;q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(p)
def ready():
 g=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
 _,uuid,free=next([v.strip() for v in l.split(',')] for l in g.splitlines() if l.split(',')[0].strip()=='6')
 assert int(free)>=18432,free
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
 for l in apps.splitlines():
  u,pid,name=[s.strip() for s in l.split(',',2)]
  if u==uuid:assert name=='[Not Found]' and not Path('/proc',pid).exists(),l
 return dict(gpu=6,free_mib=int(free),apps=apps)
def main():
 global CHILD,OWNER
 OUT.mkdir(parents=True,exist_ok=False);OWNER=True
 lock=(OUT/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (OUT/'launcher.pid').write_text(str(os.getpid()))
 save('gpu_snapshot.json',ready())
 old=json.loads((ROOT/'evaluate_results/fastwam_t4_replan_pilot_20261020/command.json').read_text())
 args=[]
 for a in old[2:]:
  k=a.split('=',1)[0]
  if k=='seed':a='seed=7'
  elif k=='EVALUATION.output_dir':a=k+'='+str(OUT)
  elif k=='EVALUATION.task_id':a=k+'=151'
  args.append(a)
 cmd=['/data/pxchen/envs/jepa-wam/bin/python','scripts/diagnose_t4_repro.py',*args]
 save('command.json',cmd)
 source=OUT/'source_snapshot';source.mkdir()
 hashes={}
 for name in ('scripts/diagnose_t4_repro.py','scripts/run_t4_repro_audit.py','experiments/libero/eval_libero_single.py'):
  p=ROOT/name;hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copyfile(p,source/p.name)
 save('manifest.json',dict(purpose='Reproducibility root-cause isolation, not an effectiveness experiment',created=time.strftime('%F %T'),seed=7,gpu=6,one_model_instance=True,no_environment_steps_between_action_trials=True,vary_only='cached T5 context from prompt batch group',source_sha256=hashes))
 env=os.environ.copy();env.update(TZ='Asia/Shanghai',CUDA_VISIBLE_DEVICES='6',MUJOCO_EGL_DEVICE_ID='6',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',LIBERO_CONFIG_PATH='/data/pxchen/FastWAM/.libero_plus_config',DIFFSYNTH_MODEL_BASE_PATH='/data/pxchen/FastWAM/checkpoints',DIFFSYNTH_SKIP_DOWNLOAD='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH=':'.join(map(str,[ROOT,ROOT/'src',Path('/data/pxchen/LIBERO-plus'),ROOT/'artifacts/jepawam_deploy_smoke_20261009/python_deps'])))
 env.pop('LIBERO_WORKER_MODE',None)
 ready();save('progress.json',dict(state='LOADING',updated=time.strftime('%F %T')))
 with (OUT/'run.log').open('w') as f:
  CHILD=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
  save('active_process.json',dict(pid=CHILD.pid,launcher_pid=os.getpid(),gpu=6))
  rc=CHILD.wait()
 assert rc==0,rc
if __name__=='__main__':
 try:main()
 except BaseException:
  if OWNER:
   if CHILD is not None and CHILD.poll() is None:CHILD.terminate()
   save('launcher_failure.json',dict(traceback=traceback.format_exc()))
  raise
