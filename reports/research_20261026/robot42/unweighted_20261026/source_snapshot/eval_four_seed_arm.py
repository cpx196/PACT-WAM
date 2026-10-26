"""One arm, pinned T5, three additional seeds; native execution and fixed weights."""
from wand.api import library as _wand_library
import fcntl,functools,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
import eval_weighted_six as shared
ROOT=Path('/data/pxchen/PACT-WAM')
OUT=Path(os.environ['GUIDANCE_SIX_OUT'])
PARENT=Path(os.environ['FOUR_SEED_ROOT'])
ARM=os.environ['FOUR_SEED_ARM']
WEIGHTED=ARM=='weighted'
TASKS=(151,242,260,321,559,590)
SEEDS=(17,27,37)
OLD=ROOT/'evaluate_results/feedback_weights_6task_seed7_sameT5_20261021'
REFERENCE=ROOT/'evaluate_results/t4_seed7_baseline_reference_20261021'
ROWS=[];CURRENT={};GOLD={};NS=None
shared.OUT=OUT

def save(name,obj):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
def status(state,**kw):
 save('progress.json',dict(state=state,arm=ARM,completed=len(ROWS),total=18,updated=time.strftime('%F %T'),**kw))
 print(ARM,state,kw,flush=True)
def summarize():
 save('summary.json',dict(arm=ARM,partial=len(ROWS)<18,seeds=list(SEEDS),completed=len(ROWS),total=18,successes=sum(r['successes'] for r in ROWS),rollouts=ROWS,audit=dict(shared.AUDIT)))
def paired(seed,tid,obs,actions):
 folder=PARENT/'paired_preflight';folder.mkdir(exist_ok=True)
 value=dict(seed=seed,task_id=tid,observation=shared.observation_digest(obs),unguided_full32_sha256=shared.digest(actions))
 p=folder/f'seed{seed}_task{tid}.json'
 with (folder/'lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if p.exists():assert json.loads(p.read_text())==value,('Paired input/action mismatch',seed,tid)
  else:p.write_text(json.dumps(value,indent=2))

def driver(ns,runner,**kw):
 cfg=kw['cfg'];predict=ns['_predict_action_chunk'];bench=ns['benchmark'].get_benchmark_dict()['libero_10'](task_order_index=0)
 assert cfg.EVALUATION.num_trials==1 and cfg.EVALUATION.replan_steps==8 and kw['action_horizon']==32
 assert cfg.EVALUATION.num_inference_steps==10 and cfg.EVALUATION.num_steps_wait==30
 assert not cfg.EVALUATION.compile_action_infer and not cfg.EVALUATION.use_action_ensembler
 assert not cfg.EVALUATION.trigger_probe.enabled and cfg.EVALUATION.binarize_gripper
 g=cfg.EVALUATION.vjepa2_ac.guidance
 assert bool(g.enabled)==WEIGHTED
 if WEIGHTED:assert list(g.after_flow_steps)==[7,8,9] and g.step_size==4 and g.max_delta_rms==.02 and not g.normalize_gradient
 save('runtime_config.json',ns['OmegaConf'].to_container(cfg,resolve=True))
 refs=[json.loads(l) for l in (REFERENCE/'baseline_windows.jsonl').read_text().splitlines()]
 checked=[]

 def preflight(seed):
  shared.STAGE='PREFLIGHT'
  for tid in TASKS:
   cfg.seed=seed;cfg.EVALUATION.task_id=tid;ns['set_global_seed'](seed,get_worker_init_fn=False)
   env,description=ns['get_libero_env'](bench.get_task(tid),ns['LIBERO_ENV_RESOLUTION'],seed)
   try:
    env.reset();ns['apply_libero_camera'](env,cfg.EVALUATION.vjepa2_ac.camera)
    obs=env.set_init_state(bench.get_task_init_states(tid)[0])
    for _ in range(30):obs,_,_,_=env.step(ns['get_libero_dummy_action']())
    args=dict(obs=obs,previous_jepa_image=None,task_description=description,model=kw['model'],processor=kw['processor'],cfg=cfg,action_horizon=32,input_w=kw['input_w'],input_h=kw['input_h'],model_device=kw['model_device'],executed_horizon=8,replan_index=0)
    cfg.EVALUATION.vjepa2_ac.enabled=False;cfg.EVALUATION.visualize_future_video=False
    a=predict(**args)[0]
    cfg.EVALUATION.visualize_future_video=True;b=predict(**args)[0]
    assert np.array_equal(a,b),('Decode-dependent action drift',seed,tid)
    paired(seed,tid,obs,a)
    if seed==7:
     r=next(r for r in refs if r['task_id']==tid and r['control_start']==0)
     assert np.array_equal(a[:4],np.asarray(r['commands'],np.float32)),('Historical seed7 baseline drift',tid)
     assert np.array_equal(ns['_extract_sim_state'](obs),np.asarray(r['state'],np.float32))
    cfg.EVALUATION.vjepa2_ac.enabled=WEIGHTED
    shared.CURRENT=dict(seed=seed,replan_index=0)
    if WEIGHTED:
     c=predict(**args,ac_ranker=kw['ac_ranker'],ac_adapter=kw['ac_adapter'])[0]
     d=predict(**args,ac_ranker=kw['ac_ranker'],ac_adapter=kw['ac_adapter'])[0]
     assert np.array_equal(c,d),('Guided repeat drift',seed,tid)
     if seed==7:assert np.array_equal(c,np.load(OLD/f'preflight_task{tid}_guided_actions.npy')),('Seed7 weighted reference drift',tid)
    else:
     c=b;d=predict(**args)[0];assert np.array_equal(c,d)
    GOLD[(seed,tid)]=dict(action=c.copy(),obs=shared.observation_digest(obs))
    checked.append(dict(seed=seed,task_id=tid,initial_observation=GOLD[(seed,tid)]['obs'],unguided_action_sha256=shared.digest(a),arm_action_sha256=shared.digest(c),passed=True))
    save('preflight.json',dict(passed=True,rows=checked));status('PREFLIGHT_PASSED',seed=seed,task_id=tid)
   finally:
    cfg.EVALUATION.vjepa2_ac.enabled=WEIGHTED;cfg.EVALUATION.visualize_future_video=True;env.close()

 preflight(7)
 @functools.wraps(predict)
 def recorded(*args,**kwargs):
  cfg=kwargs['cfg'];seed=int(cfg.seed);tid=int(cfg.EVALUATION.task_id);idx=int(kwargs.get('replan_index',0))
  shared.CURRENT=dict(seed=seed,replan_index=idx)
  result=predict(*args,**kwargs);commands=result[0]
  assert np.isfinite(commands).all()
  if idx==0:
   assert shared.observation_digest(kwargs['obs'])==GOLD[(seed,tid)]['obs']
   assert np.array_equal(commands,GOLD[(seed,tid)]['action']),('Formal/preflight mismatch',seed,tid)
  shared.append('plans.jsonl',dict(seed=seed,task_id=tid,replan_index=idx,first8_commands=commands[:8].tolist(),full32_sha256=shared.digest(commands),timing=result[3]))
  if idx<3 or idx%10==0:status('RUNNING',seed=seed,task_id=tid,replans_completed=idx+1,policy_replan_s=result[3].get('total_s'))
  return result
 for seed in SEEDS:
  # Preflight uses the unwrapped predictor, then formal execution uses the recorder.
  ns['_predict_action_chunk']=predict;preflight(seed)
  save('smoke_summary.json',dict(passed=True,seed7_reference_reproduced=True,current_seed=seed,repeat_exact=True,paired_observations_and_unguided_actions=True,audit=dict(shared.AUDIT)))
  ns['_predict_action_chunk']=recorded;shared.STAGE='FORMAL'
  for tid in TASKS:
   cfg.seed=seed;cfg.EVALUATION.task_id=tid;ns['set_global_seed'](seed,get_worker_init_fn=False)
   status('START_EPISODE',seed=seed,task_id=tid)
   args=dict(kw);args.update(task_id=tid,suite_name='libero_10',output_root=OUT/f'seed_{seed}',worker_id=None)
   path,result=runner(**args);timing=result['episode_timings'][0]
   row=dict(arm=ARM,seed=seed,task_id=tid,trial=0,initial_state_index=0,successes=result['successes'],control_steps=timing['low_level_action_steps'],wam_calls=timing['replan_count'],duration=result['duration'],result_path=str(path),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,reused=False)
   ROWS.append(row);save(f'seed{seed}_task{tid}_result.json',row);summarize();status('EPISODE_COMPLETED',seed=seed,task_id=tid,success=bool(result['successes']))
 summarize();status('COMPLETED',successes=sum(r['successes'] for r in ROWS))
 return None,dict(successes=sum(r['successes'] for r in ROWS),duration=sum(r['duration'] for r in ROWS))

def main():
 native=ROOT/'experiments/libero/eval_libero_single.py'
 assert hashlib.sha256(native.read_bytes()).hexdigest()==os.environ['GUIDANCE_SIX_NATIVE_SHA256']
 code=native.read_text();entry='if __name__ == "__main__":\n    eval_single_process()';assert code.count(entry)==1
 def install(ns):
  global NS
  NS=ns;shared.NS=ns
  ns['VJEPA2ACRanker']=shared.ARRanker;ns['LiberoACAdapter']=shared.RawAdapter
  ns['_evaluation_task_keys']=lambda cfg:[('libero_10',t) for t in TASKS]
  ns['_preencode_prompt_contexts']=shared.preencode
  if WEIGHTED:shared.install_gripper_audit(ns)
  original=ns['_run_task_to_file'];ns['_run_task_to_file']=lambda **kw:driver(ns,original,**kw)
 code=code.replace(entry,'if __name__ == "__main__":\n    install_arm(globals())\n    eval_single_process()')
 exec(compile(code,str(native),'exec'),dict(__name__='__main__',__file__=str(native),install_arm=install))

if __name__=='__main__':
 try:main()
 except BaseException:save('failure.json',dict(traceback=traceback.format_exc()));status('FAILED');raise
