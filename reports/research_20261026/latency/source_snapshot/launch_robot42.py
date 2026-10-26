"""Detached, sequential GPU0 arms. No extra rollouts or automatic retries."""
import fcntl,hashlib,json,os,shutil,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/data/pxchen/PACT-WAM');OUT=ROOT/'evaluate_results/white_mug_robot42_seed7_sameT5_20261025'
PY='/data/pxchen/envs/jepa-wam/bin/python'
OLD=ROOT/'evaluate_results/feedback_weights_6task_seed7_sameT5_20261021'
SNAP=ROOT/'evaluate_results/weighted_six_four_seeds_sameT5_20261022/source_snapshot'
CHILD=None
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(name,x):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(p)
def status(state,**kw):
 save('progress.json',dict(state=state,updated=time.strftime('%F %T'),**kw));print(state,kw,flush=True)
def aggregate():
 groups={};pairs={}
 for arm in ['weighted','control']:
  p=OUT/arm/'summary.json'
  if p.exists():
   s=json.loads(p.read_text());groups[arm]=dict(completed=s['completed'],successes=s['successes'])
   for r in s['rollouts']:pairs.setdefault(str(r['task_id']),{})[arm]=r['successes']
 paired={k:v for k,v in pairs.items() if len(v)==2}
 save('summary.json',dict(groups=groups,completed=sum(v['completed'] for v in groups.values()),total=84,paired_completed=len(paired),rescue=[int(k) for k,v in paired.items() if v['weighted'] and not v['control']],harm=[int(k) for k,v in paired.items() if v['control'] and not v['weighted']],partial=len(paired)!=42))
def available():
 text=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 free,util=map(int,text.strip().split(','));return free>=17800 and util<10,free,util
def main():
 global CHILD
 OUT.mkdir(exist_ok=False);lock=(OUT/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (OUT/'launcher.pid').write_text(str(os.getpid()));src=OUT/'source_snapshot';src.mkdir()
 for p in SNAP.iterdir():
  if p.is_file():shutil.copyfile(p,src/p.name)
 for name in ['eval_robot42.py','launch_robot42.py']:shutil.copyfile(ROOT/'scripts'/name,src/name)
 expected=json.loads((ROOT/'artifacts/white_mug_robot42_inventory_20261025.json').read_text())
 official=json.loads(Path('/data/pxchen/LIBERO-plus/libero/libero/benchmark/task_classification.json').read_text())['libero_10']
 rows=[r for r in official if 'LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate' in r['name'] and r['category']=='Robot Initial States']
 assert rows==expected and len(rows)==42
 tasks=[dict(task_id=r['id']-1,classification_id=r['id'],name=r['name'],difficulty_level=r['difficulty_level'],seed=7,trial=0,initial_state_index=0) for r in rows]
 reg=json.loads((OLD/'offline/weights_registry.json').read_text());w=next(r for r in reg['tasks'] if 559 in r['case_ids']);assert sha(w['path'])==w['sha256']
 w['case_ids']=[r['task_id'] for r in tasks]
 save('weights_registry.json',dict(complete=True,predictor_sha256=reg['predictor_sha256'],tasks=[w]))
 native_sha=sha(ROOT/'experiments/libero/eval_libero_single.py');assert native_sha=='25d1940d14035d9416ac26e42fb01d7350f61a8a330a09c48fc12c49b9d93d1b'
 ref=json.loads((ROOT/'evaluate_results/t4_seed7_baseline_reference_20261021/manifest.json').read_text())
 assert sha(ref['text_cache_path'])==ref['text_cache_sha256']
 save('manifest.json',dict(base_task='white_mug_plate_chocolate_pudding_right',perturbation='Robot Initial States',tasks=tasks,seed=7,total_episodes=84,arm_order=['weighted','control'],gpu=0,native_sha256=native_sha,historical_t5_path=ref['text_cache_path'],historical_t5_sha256=ref['text_cache_sha256'],fixed_weight_path=w['path'],fixed_weight_sha256=w['sha256'],predictor_sha256=reg['predictor_sha256'],settings=dict(replan=8,denoise=10,plan=32,max_controls=700,guidance_nodes=[7,8,9],raw_gradient_scale=4,delta_rms_cap=.02,beta=1),t5_policy='Encode42 once at batch1; replace559 with exact historical tensor; immutable shared cache across arms',source_sha256={p.name:sha(p) for p in src.iterdir() if p.is_file()}))
 aggregate()
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',MUJOCO_EGL_DEVICE_ID='0',TZ='Asia/Shanghai',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',LIBERO_CONFIG_PATH='/data/pxchen/FastWAM/.libero_plus_config',DIFFSYNTH_MODEL_BASE_PATH='/data/pxchen/FastWAM/checkpoints',DIFFSYNTH_SKIP_DOWNLOAD='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',ROBOT42_ROOT=str(OUT),ACTION_WEIGHTS_REGISTRY=str(OUT/'weights_registry.json'),PYTHONPATH=':'.join(map(str,[src,ROOT,ROOT/'src',Path('/data/pxchen/LIBERO-plus'),ROOT/'artifacts/jepawam_deploy_smoke_20261009/python_deps'])))
 env['LD_LIBRARY_PATH']='/data/pxchen/envs/jepa-wam/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
 for k in list(env):
  if k.startswith('LIBERO_WORKER_'):env.pop(k)
 oldcmd=json.loads((OLD/'command.json').read_text())
 for arm in ['weighted','control']:
  since=None
  while True:
   ok,free,util=available();now=time.monotonic();since=(since or now) if ok else None
   status('WAITING_GPU0',next_arm=arm,free_mib=free,utilization=util)
   if since is not None and now-since>=10:break
   time.sleep(5)
  folder=OUT/arm;folder.mkdir();e=env.copy();e.update(GUIDANCE_SIX_OUT=str(folder),ROBOT42_ARM=arm)
  changes={'EVALUATION.output_dir':str(folder),'EVALUATION.task_id':'525','EVALUATION.vjepa2_ac.enabled':str(arm=='weighted').lower(),'EVALUATION.vjepa2_ac.guidance.enabled':str(arm=='weighted').lower()}
  args=[a.split('=',1)[0]+'='+changes[a.split('=',1)[0]] if a.split('=',1)[0] in changes else a for a in oldcmd[2:] if not a.startswith('MULTIRUN.')]
  cmd=[PY,str(src/'eval_robot42.py'),*args];save(f'{arm}/command.json',cmd)
  with (folder/'run.log').open('w') as log:
   CHILD=subprocess.Popen(cmd,cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT)
   save(f'{arm}/active_process.json',dict(pid=CHILD.pid,gpu=0,started=time.strftime('%F %T'),launcher_pid=os.getpid()))
   stable=False
   while CHILD.poll() is None:
    aggregate();p=folder/'progress.json';progress=json.loads(p.read_text()) if p.exists() else {}
    plans=folder/'plans.jsonl'
    if not stable and plans.exists() and sum(1 for _ in plans.open())>=3:
     stable=True;save(f'{arm}/startup_stability.json',dict(passed=True,updated=time.strftime('%F %T'),at_least_three_formal_replans=True,historical559_preflight_passed=True));status('RUNNING_STABLE',arm=arm,pid=CHILD.pid)
    save('active_progress.json',dict(arm=arm,pid=CHILD.pid,worker_progress=progress,updated=time.strftime('%F %T')))
    time.sleep(10)
   assert CHILD.returncode==0,(arm,CHILD.returncode)
  assert json.loads((folder/'summary.json').read_text())['completed']==42
 aggregate();status('COMPLETED');save('launcher_completed.json',dict(returncode=0,completed=time.strftime('%F %T')))
if __name__=='__main__':
 try:main()
 except BaseException:
  if CHILD is not None and CHILD.poll() is None:CHILD.terminate()
  if OUT.exists():save('launcher_failure.json',dict(traceback=traceback.format_exc()));status('FAILED')
  raise
