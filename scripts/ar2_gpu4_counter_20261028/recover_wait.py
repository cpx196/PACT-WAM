import os,pathlib,subprocess,time,json,urllib.request,fcntl,signal
HERE=pathlib.Path(__file__).resolve().parent;OUT=pathlib.Path('/data/pxchen/PACT-WAM/evaluate_results/ar2_gpu4_counter_20261028')
lock=(OUT/'recovery_wait.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(OUT/'recovery_wait.pid').write_text(str(os.getpid()))
def healthy():
 try:return urllib.request.urlopen('http://127.0.0.1:8097/health',timeout=2).status==200
 except Exception:return False
while not healthy():
 env=os.environ.copy();env['PYTHONPATH']=str(HERE/'recovery_plugin')+(':'+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
 env['VLLM_ENGINE_READY_TIMEOUT_S']='86400'
 with (OUT/'vllm_recovery_wait.log').open('a') as log:
  p=subprocess.Popen(['/bin/bash','/home/pxchen/workspace/services/mai_ui_8b_gpu4_20261024/launch.sh'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (OUT/'vllm_recovery_wait_process.json').write_text(json.dumps(dict(pid=p.pid,time=time.strftime('%F %T'))))
 while p.poll() is None and not healthy():time.sleep(1)
 if healthy():
  (OUT/'vllm_recovery_health.json').write_text(json.dumps(dict(state='HEALTHY',port=8097,pid=p.pid,time=time.strftime('%F %T'))));break
 time.sleep(2)
