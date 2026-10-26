"""Two independent GPUs: 18 weighted +18 no-guidance, reuse seed7 references."""
import fcntl,hashlib,json,os,shutil,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/data/pxchen/PACT-WAM')
OUT=ROOT/'evaluate_results/weighted_six_four_seeds_sameT5_20261022'
PYTHON='/data/pxchen/envs/jepa-wam/bin/python'
OLD=ROOT/'evaluate_results/feedback_weights_6task_seed7_sameT5_20261021'
BASE=ROOT/'evaluate_results/t4_seed7_baseline_reference_20261021'
TASKS=[151,242,260,321,559,590]

def save(name,obj):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,ensure_ascii=False,indent=2));q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def idle():
 busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).split()
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
 return [r[0] for line in rows for r in [[v.strip() for v in line.split(',')]] if r[1] not in busy and int(r[2])>=18432 and int(r[3])<5]

def aggregate(reused):
 rows=list(reused)
 for arm in ('weighted','control'):
  p=OUT/arm/'summary.json'
  if p.exists():rows+=json.loads(p.read_text())['rollouts']
 grouped=[]
 for seed in [7,17,27,37]:
  for arm in ['control','weighted']:
   selected=[r for r in rows if r['seed']==seed and r['arm']==arm]
   grouped.append(dict(seed=seed,arm=arm,completed=len(selected),total=6,successes=sum(r['successes'] for r in selected)))
 save('summary.json',dict(case_set='6task_seed7_sameT5',seeds=[7,17,27,37],completed=len(rows),total=48,reused=12,new_completed=len(rows)-12,new_total=36,partial=len(rows)<48,groups=grouped,rollouts=rows))

def main():
 OUT.mkdir(exist_ok=False)
 lock=(OUT/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (OUT/'launcher.pid').write_text(str(os.getpid()))
 src=OUT/'source_snapshot';src.mkdir()
 files=['scripts/eval_four_seed_arm.py','scripts/launch_four_seed_pairs.py','scripts/eval_weighted_six.py','scripts/weighted_ar2_ranker.py','scripts/fastwam_ar2_eval.py','scripts/test_guidance_gripper_consistency.py','scripts/test_weighted_energy.py','experiments/libero/eval_libero_single.py','experiments/libero/text_context_cache.py','configs/sim_libero.yaml','src/fastwam/models/wan22/fastwam_idm.py','src/fastwam/models/wan22/schedulers/scheduler_continuous.py','artifacts/predictor2.1ac1009/smoke_joint_predictor.py','artifacts/predictor2.1ac1009/vectorized_ac_attention.py']
 hashes={}
 for f in files:
  hashes[f]=sha(ROOT/f);shutil.copyfile(ROOT/f,src/Path(f).name)
 assert hashes['experiments/libero/eval_libero_single.py']=='25d1940d14035d9416ac26e42fb01d7350f61a8a330a09c48fc12c49b9d93d1b'
 ref=json.loads((BASE/'manifest.json').read_text())
 assert sha(ref['text_cache_path'])==ref['text_cache_sha256']=='6102e417e48e5e6d516f18d56d4556fcc09e8562ee7c7f6b6a7d0225e918522a'
 registry=json.loads((OLD/'offline/weights_registry.json').read_text());assert registry['complete']
 for r in registry['tasks']:assert sha(r['path'])==r['sha256']
 reused=[]
 baseline=json.loads((BASE/'baseline_summary.json').read_text())
 weighted=json.loads((OLD/'summary.json').read_text());assert not weighted['partial'] and weighted['successes']==2
 assert json.loads((OLD/'baseline_preflight.json').read_text())['passed']
 assert sha(ROOT/'artifacts/fastwam_predictor21_ar2_full10epoch_bs16_20261014/step19690.pt')==registry['predictor_sha256']
 for arm,collection,source in [('control',baseline['rollouts'],BASE/'baseline_summary.json'),('weighted',weighted['rollouts'],OLD/'summary.json')]:
  for r in collection:
   reused.append(dict(arm=arm,seed=7,task_id=r['task_id'],trial=0,initial_state_index=0,successes=int(r.get('successes',r.get('success',False))),control_steps=r['control_steps'],reused=True,source_summary=str(source),source_summary_sha256=sha(source)))
 assert len(reused)==12
 save('seed7_reused.json',reused)
 for seed in [7,17,27,37]:
  for tid in TASKS:
   weight=next(r for r in registry['tasks'] if tid in r['case_ids'])
   save(f'weight_references/seed{seed}_task{tid}.json',dict(seed=seed,task_id=tid,base_task=weight['task'],path=weight['path'],sha256=weight['sha256'],shape=[2,576,1024],fixed_across_seeds=True))
 save('manifest.json',dict(case_set='6task_seed7_sameT5',task_ids=TASKS,seeds=[7,17,27,37],trial=0,initial_state_index=0,total_episodes=48,reused_seed7=12,new_episodes=36,arms=['weighted','control'],control='No guidance, fixed8',weights='Reuse exactly three frozen base-task tables; 24 explicit per-case/seed references',weight_registry_sha256=sha(OLD/'offline/weights_registry.json'),t5_cache=ref['text_cache_path'],t5_sha256=ref['text_cache_sha256'],source_sha256=hashes))
 aggregate(reused)
 env=os.environ.copy();env.update(TZ='Asia/Shanghai',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',LIBERO_CONFIG_PATH='/data/pxchen/FastWAM/.libero_plus_config',DIFFSYNTH_MODEL_BASE_PATH='/data/pxchen/FastWAM/checkpoints',DIFFSYNTH_SKIP_DOWNLOAD='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',GUIDANCE_SIX_NATIVE_SHA256=hashes['experiments/libero/eval_libero_single.py'],GUIDANCE_SIX_CACHE_SHA256=ref['text_cache_sha256'],ACTION_WEIGHTS_REGISTRY=str(OLD/'offline/weights_registry.json'),FOUR_SEED_ROOT=str(OUT),PYTHONPATH=':'.join(map(str,[src,ROOT,ROOT/'src',Path('/data/pxchen/LIBERO-plus'),ROOT/'artifacts/jepawam_deploy_smoke_20261009/python_deps'])))
 env['LD_LIBRARY_PATH']='/data/pxchen/envs/jepa-wam/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
 for k in list(env):
  if k.startswith('LIBERO_WORKER_'):env.pop(k)
 for name in ['test_guidance_gripper_consistency.py','test_weighted_energy.py']:
  with (OUT/(name+'.log')).open('w') as log:subprocess.run([PYTHON,str(src/name)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 save('progress.json',dict(state='WAITING_TWO_GPUS',updated=time.strftime('%F %T')))
 since={}
 while True:
  now=time.monotonic();available=idle();since={g:since.get(g,now) for g in available};ready=[g for g in available if now-since[g]>=20]
  save('gpu_wait.json',dict(updated=time.strftime('%F %T'),available=available,required_free_mib=18432))
  if len(ready)>=2:gpus=ready[:2];break
  time.sleep(10)
 for name,digest in hashes.items():assert sha(ROOT/name)==digest,('Queued source changed',name)
 oldcmd=json.loads((OLD/'command.json').read_text());children={};logs=[]
 for arm,gpu in zip(['weighted','control'],gpus):
  folder=OUT/arm;folder.mkdir();worker_env=env.copy();worker_env.update(CUDA_VISIBLE_DEVICES=gpu,MUJOCO_EGL_DEVICE_ID=gpu,GUIDANCE_SIX_OUT=str(folder),FOUR_SEED_ARM=arm)
  args=[]
  changes={'seed':'17','EVALUATION.output_dir':str(folder),'EVALUATION.vjepa2_ac.enabled':str(arm=='weighted').lower(),'EVALUATION.vjepa2_ac.guidance.enabled':str(arm=='weighted').lower()}
  for arg in oldcmd[2:]:
   key=arg.split('=',1)[0]
   if key.startswith('MULTIRUN.'):continue
   args.append(key+'='+changes[key] if key in changes else arg)
  cmd=[PYTHON,str(src/'eval_four_seed_arm.py'),*args];save(f'{arm}/command.json',cmd)
  log=(folder/'run.log').open('w');logs.append(log)
  child=subprocess.Popen(cmd,cwd=ROOT,env=worker_env,stdout=log,stderr=subprocess.STDOUT);children[arm]=child
  save(f'{arm}/active_process.json',dict(pid=child.pid,gpu=int(gpu),launcher_pid=os.getpid(),started=time.strftime('%F %T')))
 save('active_process.json',{arm:dict(pid=p.pid,gpu=int(gpus[i])) for i,(arm,p) in enumerate(children.items())})
 stable=False
 while True:
  aggregate(reused)
  failures={arm:p.returncode for arm,p in children.items() if p.poll() is not None and p.returncode!=0}
  if failures:
   for arm,p in children.items():
    if p.poll() is None:p.terminate()
   raise RuntimeError(('Worker failed; paired run stopped',failures))
  if not stable and all((OUT/arm/'smoke_summary.json').exists() and (OUT/arm/'plans.jsonl').exists() for arm in children):
   stable=True;save('startup_stability.json',dict(passed=True,updated=time.strftime('%F %T'),both_arms_passed_seed7_and_seed17_preflight=True,both_formal_episodes_started=True))
  save('progress.json',dict(state='RUNNING_STABLE' if stable else 'LOADING_AND_PREFLIGHT',updated=time.strftime('%F %T'),new_completed=json.loads((OUT/'summary.json').read_text())['new_completed']))
  if all(p.poll() is not None for p in children.values()):break
  time.sleep(10)
 for log in logs:log.close()
 aggregate(reused);assert json.loads((OUT/'summary.json').read_text())['completed']==48
 save('progress.json',dict(state='COMPLETED',updated=time.strftime('%F %T'),new_completed=36))
 save('launcher_completed.json',dict(returncode=0,completed=time.strftime('%F %T')))

if __name__=='__main__':
 try:main()
 except BaseException:
  if OUT.exists():save('launcher_failure.json',dict(traceback=traceback.format_exc()))
  raise
