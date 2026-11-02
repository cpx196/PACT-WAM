"""Single arm: official robot-pose variants, fixed seed and immutable inputs."""
from wand.api import library as _wand_library
import functools,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
import eval_weighted_six as shared
ROOT=Path('/data/pxchen/PACT-WAM')
PARENT=Path(os.environ['ROBOT42_ROOT']);OUT=Path(os.environ['GUIDANCE_SIX_OUT']);RUN=Path(os.environ['SOUP41_RUN_ROOT'])
ARM=os.environ['ROBOT42_ARM'];WEIGHTED=ARM!='control'
MAN=json.loads((PARENT/'manifest.json').read_text());TASKS=tuple(r['task_id'] for r in MAN['tasks'])
OLD=ROOT/'evaluate_results/feedback_weights_6task_seed7_sameT5_20261021'
REF=ROOT/'evaluate_results/t4_seed7_baseline_reference_20261021'
NS=None;ROWS=[];NATIVE_PREENCODE=None
shared.OUT=OUT

def save(name,value):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
def status(state,**kw):
 save('progress.json',dict(state=state,arm=ARM,completed=len(ROWS),total=len(TASKS),updated=time.strftime('%F %T'),**kw));print(state,kw,flush=True)
def summary():
 save('summary.json',dict(arm=ARM,partial=len(ROWS)!=len(TASKS),completed=len(ROWS),total=len(TASKS),successes=sum(r['successes'] for r in ROWS),rollouts=ROWS,audit=dict(shared.AUDIT)))
def preencode(cfg,*,device,dtype):
 from experiments.libero.text_context_cache import cache_identity,save_cache,load_cache,record_audit,file_sha256
 path=PARENT/'shared_text_contexts.pt'
 assert path.exists(), 'Baseline repair must reuse existing T5; refuse re-encoding'
 _,tc,_,tk=NS['_resolve_configs'](model_id=str(cfg.model.model_id),tokenizer_model_id=str(cfg.model.tokenizer_model_id),redirect_common_files=bool(cfg.model.get('redirect_common_files',True)))
 tc.download_if_necessary();tk.download_if_necessary()
 identity=cache_identity(tc.path,tk.path,int(cfg.model.get('tokenizer_max_len',cfg.data.train.get('context_len',128))),dtype)
 identity['encoding_batch_size']='gpu_batch1_except_historical_task321'
 prompts=NS['_evaluation_prompts'](cfg)
 if path.exists():
  checksum=json.loads((PARENT/'text_cache_manifest.json').read_text())['sha256']
  cache,checksum=load_cache(path,identity,prompts,checksum)
 else:
  assert ARM=='weighted'
  cache=NATIVE_PREENCODE(cfg,device=device,dtype=dtype)
  oldpath=Path(MAN['historical_t5_path']);assert file_sha256(oldpath)==MAN['historical_t5_sha256']
  old=torch.load(oldpath,map_location='cpu',weights_only=True)['contexts']
  chosen=[p for p in cache if 'initstate 401' in p];assert len(chosen)==1 and chosen[0] in old
  cache[chosen[0]]=old[chosen[0]]
  checksum=save_cache(cache,path,identity)
  (PARENT/'text_cache_manifest.json').write_text(json.dumps(dict(path=str(path),sha256=checksum,identity=identity,prompt_count=len(cache),historical_321_preserved=True),indent=2))
 record_audit(OUT,cache,identity,'robot42_shared_immutable',path,checksum)
 assert len(cache)==len(TASKS)
 return cache

def native_predictor(ns):
 predict=ns['_soup41_native_predict']
 ns['_predict_action_chunk']=predict
 return predict

def driver(ns,runner,**kw):
 cfg=kw['cfg'];predict=native_predictor(ns);bench=ns['benchmark'].get_benchmark_dict()['libero_10'](task_order_index=0)
 assert cfg.seed==7 and cfg.EVALUATION.num_trials==1 and cfg.EVALUATION.max_steps_override==700
 assert cfg.EVALUATION.replan_steps==8 and kw['action_horizon']==32 and cfg.EVALUATION.num_inference_steps==10
 assert not cfg.EVALUATION.compile_action_infer and not cfg.EVALUATION.use_action_ensembler
 assert not cfg.EVALUATION.trigger_probe.enabled and cfg.EVALUATION.binarize_gripper
 g=cfg.EVALUATION.vjepa2_ac.guidance
 assert g.enabled==WEIGHTED and list(g.after_flow_steps)==[7,8,9] and g.step_size==4 and g.max_delta_rms==.02 and not g.normalize_gradient
 save('runtime_config.json',ns['OmegaConf'].to_container(cfg,resolve=True))
 gold={};prechecks=[]
 def preflight(tid):
  shared.STAGE='PREFLIGHT';cfg.EVALUATION.task_id=tid;ns['set_global_seed'](7,get_worker_init_fn=False)
  env,desc=ns['get_libero_env'](bench.get_task(tid),ns['LIBERO_ENV_RESOLUTION'],7)
  try:
   env.reset();ns['apply_libero_camera'](env,cfg.EVALUATION.vjepa2_ac.camera)
   obs=env.set_init_state(bench.get_task_init_states(tid)[0])
   for _ in range(30):obs,_,_,_=env.step(ns['get_libero_dummy_action']())
   args=dict(obs=obs,previous_jepa_image=None,task_description=desc,model=kw['model'],processor=kw['processor'],cfg=cfg,action_horizon=32,input_w=kw['input_w'],input_h=kw['input_h'],model_device=kw['model_device'],executed_horizon=8,replan_index=0)
   cfg.EVALUATION.vjepa2_ac.enabled=False;cfg.EVALUATION.visualize_future_video=False
   a=predict(**args)[0];cfg.EVALUATION.visualize_future_video=True;b=predict(**args)[0]
   assert np.array_equal(a,b),('Decode/repeated native drift',tid)
   pair=dict(seed=7,task_id=tid,observation=shared.observation_digest(obs),unguided_full32_sha256=shared.digest(a))
   p=PARENT/'paired_preflight'/f'task{tid}.json';p.parent.mkdir(exist_ok=True)
   if p.exists():assert json.loads(p.read_text())==pair,('Paired input mismatch',tid)
   else:assert WEIGHTED;p.write_text(json.dumps(pair,indent=2))
   if tid==321:
    r=next(json.loads(l) for l in (REF/'baseline_windows.jsonl').read_text().splitlines() if json.loads(l)['task_id']==321 and json.loads(l)['control_start']==0)
    assert np.array_equal(a[:4],np.asarray(r['commands'],np.float32)),'Historical no-guidance559 drift'
   cfg.EVALUATION.vjepa2_ac.enabled=WEIGHTED;shared.CURRENT=dict(seed=7,replan_index=0)
   if WEIGHTED:
    c=predict(**args,ac_ranker=kw['ac_ranker'],ac_adapter=kw['ac_adapter'])[0]
    d=predict(**args,ac_ranker=kw['ac_ranker'],ac_adapter=kw['ac_adapter'])[0]
    assert np.array_equal(c,d),('Guided repeated drift',tid)
   else:c=a
   gold[tid]=dict(action=c.copy(),observation=pair['observation'])
   prechecks.append(dict(task_id=tid,passed=True,unguided_full32_sha256=pair['unguided_full32_sha256'],arm_full32_sha256=shared.digest(c)))
   save('preflight.json',dict(rows=prechecks));status('PREFLIGHT_PASSED',task_id=tid)
  finally:cfg.EVALUATION.vjepa2_ac.enabled=WEIGHTED;cfg.EVALUATION.visualize_future_video=True;env.close()
 preflight(321)
 @functools.wraps(predict)
 def recorded(*a,**k):
  tid=int(cfg.EVALUATION.task_id);idx=int(k.get('replan_index',0));shared.CURRENT=dict(seed=7,replan_index=idx)
  result=predict(*a,**k);commands=result[0];assert np.isfinite(commands).all()
  if idx==0:
   assert shared.observation_digest(k['obs'])==gold[tid]['observation']
   assert np.array_equal(commands,gold[tid]['action']),('Formal first plan drift',tid)
  shared.append('plans.jsonl',dict(seed=7,task_id=tid,replan_index=idx,first8_commands=commands[:8].tolist(),full32_sha256=shared.digest(commands),timing=result[3]))
  if idx<3 or idx%10==0:status('RUNNING',task_id=tid,replans=idx+1)
  return result
 for tid in TASKS:
  ns['_predict_action_chunk']=predict
  if tid not in gold:preflight(tid)
  ns['_predict_action_chunk']=recorded;shared.STAGE='FORMAL';cfg.EVALUATION.task_id=tid;ns['set_global_seed'](7,get_worker_init_fn=False)
  args=dict(kw);args.update(task_id=tid,suite_name='libero_10',output_root=OUT/'formal',worker_id=None)
  path,result=runner(**args);t=result['episode_timings'][0]
  ROWS.append(dict(task_id=tid,seed=7,successes=result['successes'],control_steps=t['low_level_action_steps'],wam_calls=t['replan_count'],duration=result['duration'],result_path=str(path),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
  summary();status('EPISODE_COMPLETED',task_id=tid)
 summary();status('COMPLETED');ns['_predict_action_chunk']=predict
 return None,dict(successes=sum(r['successes'] for r in ROWS),duration=sum(r['duration'] for r in ROWS))

def acquire(kw):
 import gc
 model=kw['model'];ranker=kw['ac_ranker']
 assert all(p.device.type=='cpu' for p in model.parameters())
 assert all(p.device.type=='cpu' for p in ranker.encoder.parameters())
 assert all(p.device.type=='cpu' for p in ranker.predictor.parameters())
 gc.collect()
 save('cpu_models_ready.json',dict(ready=True,pid=os.getpid(),time=time.strftime('%F %T'),wam_parameters=sum(p.numel() for p in model.parameters()),encoder_parameters=sum(p.numel() for p in ranker.encoder.parameters()),predictor_parameters=sum(p.numel() for p in ranker.predictor.parameters())))
 while not (RUN/'release_gpu4').exists():time.sleep(.1)
 target=torch.device('cuda:0');torch.cuda.set_device(0);started=time.monotonic()
 # T5 is the only model on GPU at this stage; native preencode saves CPU
 # contexts and deletes/empties the standalone encoder before WAM transfer.
 save('gpu_t5_status.json',dict(state='LOADING_SHARED_T5_CACHE',pid=os.getpid(),time=time.strftime('%F %T')))
 cache=preencode(kw['cfg'],device=str(target),dtype=next(model.parameters()).dtype)
 model._libero_prompt_context_cache=cache
 save('gpu_t5_status.json',dict(state='T5_SAVED_AND_UNLOADED',prompts=len(cache),pid=os.getpid(),time=time.strftime('%F %T')))
 model.to(target);ranker.encoder.to(target);ranker.predictor.to(target)
 stats={k:tuple(v.to(target) for v in values) for k,values in ranker.stats.items()}
 weights={};copies={}
 for tid,w in ranker.task_weights.items():
  if id(w) not in copies:copies[id(w)]=w.to(target)
  weights[tid]=copies[id(w)]
 ranker.stats=stats;ranker.task_weights=weights;ranker.device=target;model.device=target;kw['model_device']=str(target)
 torch.cuda.synchronize()
 save('gpu_ready.json',dict(pid=os.getpid(),gpu=int(os.environ['PHYSICAL_GPU']),transfer_s=time.monotonic()-started,allocated_mib=torch.cuda.memory_allocated()/2**20,time=time.strftime('%F %T')))
 return kw

def run_both(ns,runner,**kw):
 global ARM,WEIGHTED,OUT,ROWS
 kw=acquire(kw);cfg=kw['cfg'];results={}
 for arm in ['unweighted']:
  ARM=arm;WEIGHTED=arm=='weighted';OUT=RUN/arm;OUT.mkdir(exist_ok=True);shared.OUT=OUT;ROWS=[]
  kw['ac_ranker'].task_weights={tid:torch.ones_like(w) for tid,w in kw['ac_ranker'].task_weights.items()}
  assert all(torch.equal(w,torch.ones_like(w)) for w in kw['ac_ranker'].task_weights.values())
  cfg.EVALUATION.output_dir=str(OUT);cfg.EVALUATION.vjepa2_ac.enabled=WEIGHTED;cfg.EVALUATION.vjepa2_ac.guidance.enabled=WEIGHTED
  _,results[arm]=driver(ns,runner,**kw)
  (RUN/'arms_progress.json').write_text(json.dumps(dict(completed_arms=list(results),current=arm,time=time.strftime('%F %T'))))
 (RUN/'experiment_completed.json').write_text(json.dumps(dict(complete=True,total_episodes=41,results=results,time=time.strftime('%F %T'))))
 return None,dict(successes=sum(x['successes'] for x in results.values()),duration=sum(x['duration'] for x in results.values()))


def main():
 native=ROOT/'experiments/libero/eval_libero_single.py'
 assert hashlib.sha256(native.read_bytes()).hexdigest()==MAN['native_sha256']
 code=native.read_text();entry='if __name__ == "__main__":\n    eval_single_process()';assert code.count(entry)==1
 def install(ns):
  global NS,NATIVE_PREENCODE
  NS=ns;shared.NS=ns;NATIVE_PREENCODE=ns['_preencode_prompt_contexts']
  ns['_soup41_native_predict']=ns['_predict_action_chunk']
  ns['VJEPA2ACRanker']=shared.ARRanker;ns['LiberoACAdapter']=shared.RawAdapter
  ns['_evaluation_task_keys']=lambda cfg:[('libero_10',t) for t in TASKS]
  # Defer actual text encoding until GPU4 handoff; preload only WAM/JEPA on CPU.
  ns['_preencode_prompt_contexts']=lambda cfg,**kw:{}
  if WEIGHTED:shared.install_gripper_audit(ns)
  original=ns['_run_task_to_file']
  ns['_resolve_eval_device']=lambda cfg:'cpu'
  ns['_run_task_to_file']=lambda **kw:run_both(ns,original,**kw)
  load=ns['_load_model_checkpoint']
  def checked(model,checkpoint):
   original=model.mot.load_state_dict
   def verify(*a,**k):
    result=original(*a,**k);assert not result.missing_keys and not result.unexpected_keys,result
    save('checkpoint_keys.json',dict(missing=result.missing_keys,unexpected=result.unexpected_keys));return result
   model.mot.load_state_dict=verify
   try:return load(model,checkpoint)
   finally:model.mot.load_state_dict=original
  ns['_load_model_checkpoint']=checked
 code=code.replace(entry,'if __name__ == "__main__":\n    install_robot42(globals())\n    eval_single_process()')
 exec(compile(code,str(native),'exec'),dict(__name__='__main__',__file__=str(native),install_robot42=install))
if __name__=='__main__':
 try:main()
 except BaseException:save('failure.json',dict(traceback=traceback.format_exc()));status('FAILED');raise
