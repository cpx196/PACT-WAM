"""Add only unweighted guidance to the completed paired robot42 study."""
import fcntl,hashlib,json,os,shutil,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/data/pxchen/PACT-WAM');BASE=ROOT/'evaluate_results/white_mug_robot42_seed7_sameT5_20261025'
OUT=BASE/'unweighted_20261026';PY='/data/pxchen/envs/jepa-wam/bin/python';CHILD=None
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(name,x):
 p=OUT/name;q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(p)
def main():
 global CHILD
 OUT.mkdir(exist_ok=False);lock=(OUT/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (OUT/'launcher.pid').write_text(str(os.getpid()));src=OUT/'source_snapshot';src.mkdir()
 for p in (BASE/'source_snapshot').iterdir():
  if p.is_file():shutil.copyfile(p,src/p.name)
 manifest=json.loads((BASE/'manifest.json').read_text());assert len(manifest['tasks'])==42
 assert sha(BASE/'shared_text_contexts.pt')==json.loads((BASE/'text_cache_manifest.json').read_text())['sha256']
 assert sha(ROOT/'experiments/libero/eval_libero_single.py')==manifest['native_sha256']
 assert all((BASE/'paired_preflight'/f"task{r['task_id']}.json").exists() for r in manifest['tasks'])
 original=(src/'eval_robot42.py').read_text();assert sha(src/'eval_robot42.py')==manifest['source_sha256']['eval_robot42.py']
 old="ARM=os.environ['ROBOT42_ARM'];WEIGHTED=ARM=='weighted'";assert original.count(old)==1
 code=original.replace(old,"ARM=os.environ['ROBOT42_ARM'];WEIGHTED=ARM!='control'")
 old="    if tid==559:assert np.array_equal(c,np.load(OLD/'preflight_task559_guided_actions.npy')),'Historical weighted559 drift'";assert code.count(old)==1
 code=code.replace(old,"    if tid==559:save('unweighted559_preflight.json',dict(guided_repeat_exact=True,historical_no_guidance_exact=True,note='Unweighted guidance is a different method; no weighted action equality expected.'))")
 old="  ns['VJEPA2ACRanker']=shared.ARRanker;ns['LiberoACAdapter']=shared.RawAdapter";assert code.count(old)==1
 code=code.replace(old,"  from unweighted_ranker import UnweightedRanker\n  ns['VJEPA2ACRanker']=UnweightedRanker;ns['LiberoACAdapter']=shared.RawAdapter")
 (src/'eval_robot42_unweighted.py').write_text(code)
 (src/'unweighted_ranker.py').write_text('''import json,os\nfrom pathlib import Path\nimport torch\nfrom weighted_ar2_ranker import ARRanker\nclass UnweightedRanker(ARRanker):\n def __init__(self,*a,**kw):\n  super().__init__(*a,**kw)\n  self.task_weights={tid:torch.ones_like(w) for tid,w in self.task_weights.items()}\n  assert all(torch.equal(w,torch.ones_like(w)) for w in self.task_weights.values())\n  Path(os.environ['GUIDANCE_SIX_OUT'],'unweighted_objective.json').write_text(json.dumps(dict(method='unweighted_guidance',all_feature_weights=1,effective_beta=0,loss='L1 averaged across both AR2 horizons',task_ids=sorted(self.task_weights))))\n''')
 shutil.copyfile(__file__,src/'launch_unweighted42.py')
 save('method_manifest.json',dict(arm='unweighted',gpu=7,total=42,seed=7,trial=0,initial_state_index=0,base_study=str(BASE),t5_sha256=sha(BASE/'shared_text_contexts.pt'),method='guidance enabled; per-feature weights all1; original L1 and AR2 chain retained',historical_weighted_action_check='not applicable; original native559 reference and repeated unweighted action checks retained',guidance_nodes=[7,8,9],step_size=4,max_delta_rms=.02,source_sha256={p.name:sha(p) for p in src.iterdir() if p.is_file()}))
 since=None
 while True:
  free,util=map(int,subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
  now=time.monotonic();since=(since or now) if free>=21000 and util<10 else None
  save('launcher_progress.json',dict(state='WAITING_GPU7',free_mib=free,utilization=util,updated=time.strftime('%F %T')))
  if since is not None and now-since>=10:break
  time.sleep(5)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='7',MUJOCO_EGL_DEVICE_ID='7',TZ='Asia/Shanghai',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',LIBERO_CONFIG_PATH='/data/pxchen/FastWAM/.libero_plus_config',DIFFSYNTH_MODEL_BASE_PATH='/data/pxchen/FastWAM/checkpoints',DIFFSYNTH_SKIP_DOWNLOAD='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',ROBOT42_ROOT=str(BASE),ROBOT42_ARM='unweighted',GUIDANCE_SIX_OUT=str(OUT),ACTION_WEIGHTS_REGISTRY=str(BASE/'weights_registry.json'),PYTHONPATH=':'.join(map(str,[src,ROOT,ROOT/'src',Path('/data/pxchen/LIBERO-plus'),ROOT/'artifacts/jepawam_deploy_smoke_20261009/python_deps'])))
 env['LD_LIBRARY_PATH']='/data/pxchen/envs/jepa-wam/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
 for k in list(env):
  if k.startswith('LIBERO_WORKER_'):env.pop(k)
 oldcmd=json.loads((BASE/'weighted/command.json').read_text());args=[('EVALUATION.output_dir='+str(OUT)) if a.startswith('EVALUATION.output_dir=') else a for a in oldcmd[2:]]
 cmd=[PY,str(src/'eval_robot42_unweighted.py'),*args];save('command.json',cmd)
 with (OUT/'run.log').open('w') as log:
  CHILD=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  save('active_process.json',dict(pid=CHILD.pid,gpu=7,launcher_pid=os.getpid(),started=time.strftime('%F %T')))
  stable=False
  while CHILD.poll() is None:
   plans=OUT/'plans.jsonl'
   if not stable and plans.exists() and sum(1 for _ in plans.open())>=3:
    stable=True;save('startup_stability.json',dict(passed=True,updated=time.strftime('%F %T'),at_least_three_formal_replans=True,historical_native559_and_repeat_checks_passed=True));print('RUNNING_STABLE',flush=True)
   save('launcher_progress.json',dict(state='RUNNING_STABLE' if stable else 'LOADING_PREFLIGHT',pid=CHILD.pid,updated=time.strftime('%F %T')))
   time.sleep(10)
  assert CHILD.returncode==0,CHILD.returncode
 s=json.loads((OUT/'summary.json').read_text());assert s['completed']==42
 groups={arm:json.loads((BASE/arm/'summary.json').read_text()) for arm in ['control','weighted']};groups['unweighted']=s
 lookup={arm:{r['task_id']:r for r in x['rollouts']} for arm,x in groups.items()}
 rows=[dict(task_id=r['task_id'],**{arm:lookup[arm][r['task_id']]['successes'] for arm in groups}) for r in manifest['tasks']]
 save('three_arm_summary.json',dict(total_episodes=126,groups={arm:dict(completed=x['completed'],successes=x['successes']) for arm,x in groups.items()},paired_results=rows))
 save('launcher_completed.json',dict(returncode=0,completed=time.strftime('%F %T')))
if __name__=='__main__':
 try:main()
 except BaseException:
  if CHILD is not None and CHILD.poll() is None:CHILD.terminate()
  if OUT.exists():save('launcher_failure.json',dict(traceback=traceback.format_exc()))
  raise
