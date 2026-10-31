"""Owned-service handoff; never stop vLLM until CPU preparation and recovery checks pass."""
import os,time,json,pathlib,subprocess,signal,urllib.request,fcntl,traceback
H=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/data/pxchen/PACT-WAM');O=R/'evaluate_results/soup_tomato_robot41_baseline_repair_20261029';U=os.getuid();UUID='GPU-d152c6a7-1dcf-0ff2-1802-80647045e446'
def save(name,x):
 p=O/name;p.parent.mkdir(exist_ok=True,parents=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def state(s,**kw):save('controller_status.json',dict(state=s,time=time.strftime('%F %T'),**kw))
def identity(pid):
 p=pathlib.Path('/proc')/str(pid);v=(p/'stat').read_text().rsplit(')',1)[1].split();return dict(pid=pid,ppid=int(v[1]),start=v[19],uid=p.stat().st_uid)
def alive(i):
 try:return all(identity(i['pid'])[k]==i[k] for k in ['pid','uid','start'])
 except (OSError,ValueError):return False
def health():
 try:return urllib.request.urlopen('http://127.0.0.1:8097/health',timeout=3).status==200
 except Exception:return False
def gpu():
 text=subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=uuid,memory.total,memory.free','--format=csv,noheader,nounits'],text=True,timeout=20).strip().split(',');assert text[0].strip()==UUID
 apps=subprocess.check_output(['nvidia-smi','-i','4','--query-compute-apps=pid,used_memory,process_name','--format=csv,noheader,nounits'],text=True,timeout=20)
 return dict(total=int(text[1]),free=int(text[2]),apps=[dict(pid=int(a[0]),mib=int(a[1]),name=','.join(a[2:]).strip()) for l in apps.splitlines() if (a:=l.split(',')) and a[0].strip().isdigit() and a[1].strip().isdigit()])
def snapshot():
 g=gpu();es=[a for a in g['apps'] if 'VLLM::EngineCore' in a['name']];assert len(es)==1
 e=identity(es[0]['pid']);p=identity(e['ppid']);assert e['uid']==p['uid']==U
 q=pathlib.Path('/proc')/str(p['pid']);args=[a.decode() for a in (q/'cmdline').read_bytes().split(b'\0') if a]
 assert 'vllm.entrypoints.openai.api_server' in args and args[args.index('--port')+1]=='8097';assert health()
 env=dict(a.decode().split('=',1) for a in (q/'environ').read_bytes().split(b'\0') if b'=' in a)
 assert env['CUDA_VISIBLE_DEVICES'] in ['4',UUID]
 s=dict(parent=p,engine=e,args=args,env=env,cwd=os.readlink(q/'cwd'))
 private=O/'private';private.mkdir(exist_ok=True,mode=0o700);private.chmod(0o700)
 f=private/'service.json';f.write_text(json.dumps(s));f.chmod(0o600);return s

def restore_policy():
 f=O/'restore_policy.json'
 return json.loads(f.read_text()) if f.exists() else dict(gpu_memory_utilization=.97,cpu_offload_gb=0)

def restore_args(args):
 args=list(args);policy=restore_policy()
 for key,value in [('--gpu-memory-utilization',policy['gpu_memory_utilization']),('--cpu-offload-gb',policy.get('cpu_offload_gb',0))]:
  if key in args:args[args.index(key)+1]=str(value)
  elif value:args.extend([key,str(value)])
 save('vllm_restore_plan.json',dict(gpu_memory_utilization=policy['gpu_memory_utilization'],cpu_offload_gb=policy.get('cpu_offload_gb',0),max_model_len=24576,model='MAI-UI-8B',port=8097))
 return args

class AdoptedWorker:
 def __init__(self,item):
  assert item['uid']==U and alive(item)
  self.item=item;self.pid=item['pid']
 def poll(self):
  running=alive(self.item)
  if running:
   try:running=pathlib.Path('/proc',str(self.pid),'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
   except OSError:running=False
  if running:return None
  complete=O/'experiment_completed.json'
  good=complete.exists() and json.loads(complete.read_text()).get('complete',False)
  for arm in ['control']:
   f=O/arm/'summary.json'
   good=good and f.exists() and json.loads(f.read_text()).get('completed')==41 and not (O/arm/'failure.json').exists()
  return 0 if good else 1
 def wait(self,timeout=None):
  start=time.monotonic()
  while self.poll() is None:
   if timeout is not None and time.monotonic()-start>timeout:raise TimeoutError('Adopted worker wait timed out')
   time.sleep(.1)
  return self.poll()
 def terminate(self):
  if alive(self.item):os.kill(self.pid,signal.SIGTERM)

def standby():
 lock=(O/'standby.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save('standby_ready.json',dict(pid=os.getpid(),time=time.strftime('%F %T')))
 while not (O/'vllm_stopped').exists():
  if (O/'cancel_before_stop').exists():return
  time.sleep(.1)
 s=json.loads((O/'private/service.json').read_text());env=s['env'].copy();env['PYTHONPATH']=str(H/'recovery_plugin')+(':'+env['PYTHONPATH'] if env.get('PYTHONPATH') else '');env['VLLM_ENGINE_READY_TIMEOUT_S']='86400'
 while True:
  if health():return
  with (O/'vllm_restore.log').open('ab') as log:p=subprocess.Popen(restore_args(s['args']),cwd=s['cwd'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save('standby_api.json',identity(p.pid))
  while p.poll() is None:
   if health():
    g=gpu();engines=[a for a in g['apps'] if 'VLLM::EngineCore' in a['name'] and identity(a['pid'])['uid']==U and identity(a['pid'])['ppid']==p.pid]
    if len(engines)==1:
     save('vllm_recovery_health.json',dict(state='HEALTHY',api_pid=p.pid,engine_pid=engines[0]['pid'],gpu=4,uuid=UUID,time=time.strftime('%F %T')));return
   time.sleep(1)
  save('vllm_restore_retry.json',dict(returncode=p.returncode,time=time.strftime('%F %T')));time.sleep(2)

def watchdog():
 while not (O/'controller_finished').exists():
  owner=O/'controller_identity.json'
  f=O/'worker_identity.json'
  if (O/'restore_required').exists() and owner.exists() and not alive(json.loads(owner.read_text())):
   if f.exists():
    w=json.loads(f.read_text())
    if alive(w):
     os.kill(w['pid'],signal.SIGTERM);time.sleep(5)
     if alive(w):os.kill(w['pid'],signal.SIGKILL)
   (O/'vllm_stopped').touch();(O/'experiment_gpu_released').touch()
  if (O/'restore_required').exists() and f.exists() and not alive(json.loads(f.read_text())):(O/'experiment_gpu_released').touch()
  if (O/'vllm_stopped').exists() and not (O/'vllm_recovery_health.json').exists():
   p=O/'standby_identity.json'
   if p.exists() and not alive(json.loads(p.read_text())):launch_standby()
  time.sleep(.1)
def launch_standby():
 with (O/'standby.log').open('ab') as log:p=subprocess.Popen(['/usr/bin/python3',str(H/'control.py'),'standby'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save('standby_identity.json',identity(p.pid));return p

def stop_service(s):
 assert alive(s['parent']) and alive(s['engine'])
 allp=[]
 for p in pathlib.Path('/proc').iterdir():
  if p.name.isdigit():
   try:allp.append(identity(int(p.name)))
   except (OSError,ValueError):pass
 chosen={s['parent']['pid']}
 while True:
  old=len(chosen);chosen.update(a['pid'] for a in allp if a['ppid'] in chosen)
  if len(chosen)==old:break
 items=[a for a in allp if a['pid'] in chosen];assert all(a['uid']==U for a in items)
 save('stopped_service_identities.json',items);(O/'restore_required').touch()
 for a in items:
  if alive(a):os.kill(a['pid'],signal.SIGTERM)
 deadline=time.monotonic()+25
 while any(alive(a) for a in items) and time.monotonic()<deadline:time.sleep(.1)
 for a in items:
  if alive(a):os.kill(a['pid'],signal.SIGKILL)
 (O/'vllm_stopped').touch()

def run(adopt=False):
 O.mkdir(exist_ok=True);lock=(O/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (O/'restore_required').exists(),'Used run; inspect before restarting'
 exp=None;s=None
 try:
  state('CHECKING_PREPARATION');subprocess.run(['/usr/bin/python3',str(H/'prepare_remote.py')],check=True)
  s=snapshot();save('controller_identity.json',identity(os.getpid()))
  if adopt:
   exp=AdoptedWorker(json.loads((O/'worker_identity.json').read_text()));state('PRELOADING_CPU_VLLM_UNCHANGED',worker_pid=exp.pid,adopted=True)
  else:
   # Enough real host RAM for T5 transient loading plus WAM/JEPA, before taking service down.
   while True:
    mem=int(next(l.split()[1] for l in pathlib.Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))/1024
    if mem>=40960:break
    state('WAITING_CPU_MEMORY_VLLM_UNCHANGED',available_mib=mem,required_mib=40960);time.sleep(5)
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUID,MUJOCO_EGL_DEVICE_ID='4',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',LIBERO_CONFIG_PATH='/data/pxchen/FastWAM/.libero_plus_config',DIFFSYNTH_MODEL_BASE_PATH='/data/pxchen/FastWAM/checkpoints',DIFFSYNTH_SKIP_DOWNLOAD='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',ROBOT42_ROOT=str(R/'evaluate_results/soup_tomato_robot41_seed7_sameT5_20261028'),SOUP41_RUN_ROOT=str(O),ROBOT42_ARM='weighted',GUIDANCE_SIX_OUT=str(O/'preload'),ACTION_WEIGHTS_REGISTRY=str(O/'weights_registry.json'),PYTHONPATH=':'.join(map(str,[H,R/'evaluate_results/ar2_async_timing_v6_20261027/source_snapshot',R,R/'src',pathlib.Path('/data/pxchen/LIBERO-plus'),R/'artifacts/jepawam_deploy_smoke_20261009/python_deps'])),TZ='Asia/Shanghai')
   for k in list(env):
    if k.startswith('LIBERO_WORKER_'):env.pop(k)
   env['LD_LIBRARY_PATH']='/data/pxchen/envs/jepa-wam/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
   cmd=json.loads((O/'command.json').read_text())
   with (O/'experiment.log').open('wb') as log:exp=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   save('worker_identity.json',identity(exp.pid));state('PRELOADING_CPU_VLLM_UNCHANGED',worker_pid=exp.pid)
  while not (O/'preload/cpu_models_ready.json').exists():
   if exp.poll() is not None:raise RuntimeError('CPU preparation failed; vLLM unchanged')
   time.sleep(1)
  # Ensure original .97 vLLM startup remains feasible after removing only our service.
  while True:
   assert alive(s['engine']) and alive(s['parent']) and health(),'Existing service changed before handoff'
   g=gpu();owned=sum(a['mib'] for a in g['apps'] if a['pid'] in [s['engine']['pid'],s['parent']['pid']]);required=int(g['total']*restore_policy()['gpu_memory_utilization'])+32
   if g['free']+owned>=required:break
   state('CPU_READY_WAITING_RESTORE_CAPACITY_VLLM_UNCHANGED',free_after_own_vllm_stop_mib=g['free']+owned,restore_required_mib=required,other_occupation_mib=g['total']-g['free']-owned);time.sleep(5)
  launch_standby()
  while not (O/'standby_ready.json').exists():time.sleep(.1)
  with (O/'watchdog.log').open('ab') as log:subprocess.Popen(['/usr/bin/python3',str(H/'control.py'),'watchdog'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  state('CPU_READY_STOPPING_OWN_VLLM');stop_service(s)
  deadline=time.monotonic()+30
  while gpu()['free']<18500:
   if time.monotonic()>deadline:raise RuntimeError('GPU4 did not free enough memory')
   time.sleep(.1)
  (O/'release_gpu4').touch();state('RUNNING_GPU4',worker_pid=exp.pid)
  code=exp.wait();save('experiment_exit.json',dict(returncode=code,time=time.strftime('%F %T')));(O/'experiment_gpu_released').touch()
  state('RESTORING_VLLM',experiment_returncode=code)
  while not (O/'vllm_recovery_health.json').exists():time.sleep(1)
  state('COMPLETED_VLLM_HEALTHY' if code==0 else 'EXPERIMENT_FAILED_VLLM_HEALTHY',returncode=code);(O/'controller_finished').touch()
 except BaseException:
  save('controller_failure.json',dict(traceback=traceback.format_exc()))
  if exp is not None and exp.poll() is None:exp.terminate();exp.wait(timeout=30)
  if (O/'restore_required').exists():
   (O/'vllm_stopped').touch();(O/'experiment_gpu_released').touch();state('FAILED_RESTORATION_PENDING')
  else:(O/'cancel_before_stop').touch();state('FAILED_VLLM_UNCHANGED')
if __name__=='__main__':
 import sys
 {'run':run,'adopt':lambda:run(adopt=True),'standby':standby,'watchdog':watchdog}[sys.argv[1] if len(sys.argv)>1 else 'run']()
