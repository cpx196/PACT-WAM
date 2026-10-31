"""One sudo entry; preload as pxchen, stop owned vLLM, capture, restore on exit."""
import os,sys,time,json,pathlib,subprocess,signal,pwd,urllib.request,fcntl,traceback
HERE=pathlib.Path(__file__).resolve().parent;ROOT=pathlib.Path('/data/pxchen/PACT-WAM');OUT=ROOT/'evaluate_results/ar2_gpu4_counter_20261028';PRIVATE=OUT/'private';USER=pwd.getpwnam('pxchen');NSYS='/usr/local/bin/nsys'
def write(name,obj):
 p=OUT/name;q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2));q.replace(p)
def state(name,**kw):write('controller_status.json',dict(state=name,time=time.strftime('%F %T'),**kw))
def read(name):return json.loads((OUT/name).read_text())
def health():
 try:return urllib.request.urlopen('http://127.0.0.1:8097/health',timeout=3).status==200
 except Exception:return False
def ident(pid):
 p=pathlib.Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split();return dict(pid=pid,ppid=int(s[1]),start=s[19],uid=p.stat().st_uid)
def owned_alive(item):
 try:
  now=ident(item['pid']);return all(now[k]==item[k] for k in ['pid','start','uid'])
 except (FileNotFoundError,ProcessLookupError):return False
def snapshot():
 # Resolve GPU4's owned EngineCore parent rather than choosing an unrelated process.
 line=subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 ps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'],text=True)
 engines=[int(x.split(',')[1]) for x in ps.splitlines() if x.split(',')[0].strip()==line and 'VLLM::EngineCore' in x]
 assert len(engines)==1,'Expected exactly one GPU4 vLLM engine'
 ep=ident(engines[0]);parent=ident(ep['ppid']);assert parent['uid']==ep['uid']==USER.pw_uid
 p=pathlib.Path('/proc')/str(parent['pid']);args=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
 assert 'vllm.entrypoints.openai.api_server' in args and '--port' in args and args[args.index('--port')+1]=='8097'
 env=dict(x.decode().split('=',1) for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x)
 assert env.get('CUDA_VISIBLE_DEVICES') in ['4',line]
 assert health(),'Original vLLM health must pass before any action'
 data=dict(parent=parent,engine=ep,args=args,env=env,cwd=os.readlink(p/'cwd'),gpu_uuid=line,port=8097)
 f=PRIVATE/'vllm_snapshot.json';f.write_text(json.dumps(data));f.chmod(0o600)
 return data
def descendants(pid):
 allp=[]
 for p in pathlib.Path('/proc').iterdir():
  if p.name.isdigit():
   try:allp.append(ident(int(p.name)))
   except (OSError,ValueError):pass
 chosen={pid};changed=True
 while changed:
  old=len(chosen);chosen.update(x['pid'] for x in allp if x['ppid'] in chosen);changed=len(chosen)!=old
 result=[x for x in allp if x['pid'] in chosen];assert all(x['uid']==USER.pw_uid for x in result)
 return result
def stop_old(s):
 assert owned_alive(s['parent']) and owned_alive(s['engine']),'Service process changed; abort before stop'
 items=descendants(s['parent']['pid']);write('stopped_service_identities.json',items)
 # Mark before the first signal so the independent watchdog covers partial shutdown.
 (PRIVATE/'restore_required').touch()
 for item in items:
  if owned_alive(item):os.kill(item['pid'],signal.SIGTERM)
 until=time.monotonic()+25
 while time.monotonic()<until and any(owned_alive(i) for i in items):time.sleep(.2)
 for item in items:
  if owned_alive(item):os.kill(item['pid'],signal.SIGKILL)
 until=time.monotonic()+15
 while time.monotonic()<until:
  free=int(subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
  if free>=20000:return
  time.sleep(.2)
 raise RuntimeError('GPU4 did not free 20000MiB; abort and restore vLLM')
def restore():
 if not (PRIVATE/'restore_required').exists():return
 with (PRIVATE/'restore.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if (PRIVATE/'restored').exists():return
  s=json.loads((PRIVATE/'vllm_snapshot.json').read_text())
  if not health():
   # Do not start a duplicate restored process when the watchdog retries.
   running=False
   f=PRIVATE/'restored_identity.json'
   if f.exists():
    try:running=owned_alive(json.loads(f.read_text()))
    except Exception:pass
   if not running:
    with (OUT/'vllm_restored.log').open('ab') as log:
     child=subprocess.Popen(s['args'],cwd=s['cwd'],env=s['env'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,user=USER.pw_uid,group=USER.pw_gid,extra_groups=os.getgrouplist(USER.pw_name,USER.pw_gid))
    f.write_text(json.dumps(ident(child.pid)))
   write('restore_status.json',dict(state='STARTING',time=time.strftime('%F %T')))
   deadline=time.monotonic()+600
   while time.monotonic()<deadline and not health():time.sleep(2)
  if not health():
   write('restore_status.json',dict(state='FAILED_HEALTH_CHECK',time=time.strftime('%F %T')));raise RuntimeError('vLLM restore health timeout; inspect vllm_restored.log')
  (PRIVATE/'restored').touch();write('restore_status.json',dict(state='HEALTHY',port=8097,time=time.strftime('%F %T')))
def terminate_experiment():
 f=PRIVATE/'profiler_pgid.json'
 if f.exists():
  r=json.loads(f.read_text())
  if owned_alive(r):
   try:os.killpg(r['pid'],signal.SIGTERM)
   except ProcessLookupError:pass
   time.sleep(3)
   try:os.killpg(r['pid'],signal.SIGKILL)
   except ProcessLookupError:pass
 # The actual user worker may outlive the profiler. Only kill the tracked exact identity.
 f=OUT/'worker_uid.json'
 if f.exists():
  r=json.loads(f.read_text());pid=r['pid']
  try:
   if r.get('identity') and owned_alive(r['identity']):
    os.kill(pid,signal.SIGTERM);time.sleep(2)
    if owned_alive(r['identity']):os.kill(pid,signal.SIGKILL)
  except ProcessLookupError:pass

def watchdog(parent):
 deadline=time.monotonic()+2700
 while time.monotonic()<deadline:
  if (PRIVATE/'controller_done').exists():return
  try:os.kill(parent,0)
  except ProcessLookupError:break
  # Restore as soon as the user GPU worker exits, while Nsight exports on CPU.
  # This also covers a worker crash; report generation must not leave GPU4 idle.
  wf=OUT/'worker_uid.json'
  if (PRIVATE/'restore_required').exists() and wf.exists() and not (PRIVATE/'restored').exists():
   try:
    tracked=json.loads(wf.read_text()).get('identity')
    if tracked and not owned_alive(tracked):
     restore()
   except Exception:write('watchdog_restore_failure.json',dict(traceback=traceback.format_exc()))
  time.sleep(.2)
 if (PRIVATE/'restore_required').exists():
  terminate_experiment()
  try:restore()
  except Exception:write('watchdog_failure.json',dict(traceback=traceback.format_exc()))
def run():
 lock=(PRIVATE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def interrupted(*unused):raise KeyboardInterrupt('Controller interrupted')
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 exp=None
 try:
  assert not (PRIVATE/'restore_required').exists(),'This run was already used; do not rerun blindly'
  state('CHECKING_COUNTER_PERMISSION')
  helptext=subprocess.run([NSYS,'profile','--gpu-metrics-devices=help'],capture_output=True,text=True,timeout=30)
  (OUT/'metrics_device_help.txt').write_text(helptext.stdout+helptext.stderr)
  assert 'Insufficient privilege' not in helptext.stdout+helptext.stderr,'Counter privilege unavailable'
  s=snapshot();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=s['gpu_uuid'];env['TZ']='Asia/Shanghai'
  # Validate actual metric acquisition before taking the application down.
  probe=[NSYS,'profile','--trace=none','--sample=none','--cpuctxsw=none','--gpu-metrics-devices=cuda-visible','--gpu-metrics-set=ad10x','--gpu-metrics-frequency=10000','--duration=1','--kill=none','--wait=primary','--force-overwrite=true','--output='+str(OUT/'counter_probe'),'/bin/sleep','2']
  with (OUT/'counter_probe.log').open('w') as log:rc=subprocess.run(probe,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=40).returncode
  text=(OUT/'counter_probe.log').read_text();assert rc==0 and (OUT/'counter_probe.nsys-rep').exists() and not any(w in text for w in ['Insufficient privilege','GPU Metrics: None','Failed to enable','No GPU metrics']), 'Counter probe failed; vLLM unchanged'
  export=subprocess.run([NSYS,'export','--type=sqlite','--force-overwrite=true','--output='+str(OUT/'counter_probe.sqlite'),str(OUT/'counter_probe.nsys-rep')],capture_output=True,text=True,timeout=40)
  assert export.returncode==0,'Counter probe export failed; vLLM unchanged'
  import sqlite3
  db=sqlite3.connect(OUT/'counter_probe.sqlite');tables=[x[0] for x in db.execute("select name from sqlite_master where type='table'")]
  samples={t:db.execute('select count(*) from "'+t+'"').fetchone()[0] for t in tables if t in ['GPU_METRICS','GENERIC_EVENTS']};db.close()
  write('counter_probe_validation.json',dict(sample_tables=samples))
  assert any(samples.values()),'No hardware metric samples in counter probe; vLLM unchanged'
  mem=[int(x.split()[1]) for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0]
  assert mem>35*1024*1024,'Insufficient CPU memory for preload; vLLM unchanged'
  state('PRELOADING_CPU_VLLM_STILL_RUNNING')
  cmd=[NSYS,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--gpu-metrics-devices=cuda-visible','--gpu-metrics-set=ad10x','--gpu-metrics-frequency=10000','--capture-range=cudaProfilerApi','--capture-range-end=stop','--kill=none','--wait=primary','--force-overwrite=true','--output='+str(OUT/'contention'),'/usr/bin/python3',str(HERE/'worker_entry.py'),str(OUT)]
  log=(OUT/'experiment.log').open('w');exp=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);(PRIVATE/'profiler_pgid.json').write_text(json.dumps(ident(exp.pid)))
  deadline=time.monotonic()+1200
  while not (OUT/'cpu_models_ready.json').exists():
   if exp.poll() is not None:raise RuntimeError('CPU preload failed; vLLM unchanged')
   if time.monotonic()>deadline:raise TimeoutError('CPU preload timeout; vLLM unchanged')
   time.sleep(1)
  assert read('cpu_models_ready.json')['ready'] and read('worker_uid.json')['uid']==USER.pw_uid
  state('CPU_READY_STOPPING_VLLM');subprocess.Popen(['/usr/bin/python3',str(HERE/'control.py'),'watchdog',str(os.getpid())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
  stop_old(s);(OUT/'release_gpu4').touch();state('EXPERIMENT_RUNNING_GPU4')
  code=exp.wait(timeout=1800)
  write('experiment_exit.json',dict(returncode=code,time=time.strftime('%F %T')))
  assert code==0 and (OUT/'counter_iterations.json').exists(),'Experiment failed; restoring vLLM'
  state('RESTORING_VLLM');restore();state('COMPLETED_VLLM_HEALTHY')
 except BaseException:
  write('controller_failure.json',dict(traceback=traceback.format_exc()))
  if exp is not None:terminate_experiment()
  try:restore()
  except Exception:write('restore_failure.json',dict(traceback=traceback.format_exc()))
  state('FAILED',restore=read('restore_status.json') if (OUT/'restore_status.json').exists() else 'not needed')
 finally:
  (PRIVATE/'controller_done').touch()
  # Make only non-secret reports available to the experiment account.
  for p in OUT.iterdir():
   if p.is_file():os.chown(p,USER.pw_uid,USER.pw_gid)

def main():
 assert os.geteuid()==0,'Run with sudo; do not pass a password to this script'
 OUT.mkdir(exist_ok=True);os.chown(OUT,USER.pw_uid,USER.pw_gid);PRIVATE.mkdir(exist_ok=True,mode=0o700);PRIVATE.chmod(0o700)
 if len(sys.argv)>1 and sys.argv[1]=='watchdog':watchdog(int(sys.argv[2]));return
 if len(sys.argv)>1 and sys.argv[1]=='run':run();return
 assert not (PRIVATE/'controller_started').exists(),'Already started; inspect status before attempting another run'
 (PRIVATE/'controller_started').touch()
 with (OUT/'controller.log').open('a') as log:
  p=subprocess.Popen(['/usr/bin/python3',str(HERE/'control.py'),'run'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 write('controller_started.json',dict(pid=p.pid,time=time.strftime('%F %T')))
 print('Background controller started. PID:',p.pid)
 print('Status:',OUT/'controller_status.json')
 print('vLLM is stopped only after counters pass and CPU model preload completes.')
if __name__=='__main__':main()
