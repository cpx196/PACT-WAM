import fcntl,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/data/pxchen/PACT-WAM');OUT=R/'evaluate_results/feedback_weights_libero10_remaining7_20261031';SRC=OUT/'source_snapshot';OLD=R/'evaluate_results/feedback_weights_6task_seed7_sameT5_20261021/offline/weights_registry.json'
def save(name,x):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 lock=(OUT/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 save('launcher_identity.json',dict(pid=os.getpid(),started=time.strftime('%F %T')))
 if (OUT/'weights_registry_all10.json').exists():raise RuntimeError('Already completed; refusing duplicate work')
 old=json.loads(OLD.read_text())
 for x in old['tasks']:assert sha(x['path'])==x['sha256']
 while True:
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
  busy=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).split())
  candidates=[v for line in rows for v in [[s.strip() for s in line.split(',')]] if v[1] not in busy and int(v[2])>=6500 and int(v[3])<5]
  if candidates:
   chosen=next((v for v in candidates if v[0]=='2'),candidates[0]);break
  save('launcher_status.json',dict(state='WAITING_GPU',required_free_mib=6500,time=time.strftime('%F %T')));time.sleep(5)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=chosen[1],TZ='Asia/Shanghai',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',ACTION_WEIGHTS_OUT=str(OUT/'offline'),PYTHONPATH=':'.join(map(str,[SRC,R,R/'src',R/'artifacts/jepawam_deploy_smoke_20261009/python_deps'])))
 env['LD_LIBRARY_PATH']='/data/pxchen/envs/jepa-wam/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
 with (OUT/'calibration.log').open('w') as log:
  child=subprocess.Popen(['/data/pxchen/envs/jepa-wam/bin/python',str(SRC/'run_capped.py')],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
  save('launcher_status.json',dict(state='CALIBRATING',gpu=int(chosen[0]),uuid=chosen[1],worker_pid=child.pid,time=time.strftime('%F %T')))
  rc=child.wait()
 if rc:raise RuntimeError('Calibration failed, returncode='+str(rc))
 new=json.loads((OUT/'offline/weights_registry.json').read_text());assert new['complete'] and len(new['tasks'])==7 and old['predictor_sha256']==new['predictor_sha256']
 tasks=old['tasks']+new['tasks'];assert len({x['task'] for x in tasks})==10
 for x in tasks:assert sha(x['path'])==x['sha256']
 save('weights_registry_all10.json',dict(complete=True,predictor_sha256=new['predictor_sha256'],tasks=tasks,original_three_unchanged=True))
 save('launcher_status.json',dict(state='COMPLETED',new_tasks=7,total_available=10,time=time.strftime('%F %T')))
if __name__=='__main__':
 try:main()
 except Exception:
  save('failure.json',dict(time=time.strftime('%F %T'),traceback=traceback.format_exc()));raise
