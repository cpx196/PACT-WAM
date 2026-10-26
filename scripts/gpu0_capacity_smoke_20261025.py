"""Isolated GPU0 capacity smoke: one pinned case, 48 controls, then exit."""
import os,json,time,hashlib,traceback
from pathlib import Path
import numpy as np
import torch
import eval_weighted_six as shared
OUT=Path(os.environ['GUIDANCE_SIX_OUT'])
OLD=Path('/data/pxchen/PACT-WAM/evaluate_results/feedback_weights_6task_seed7_sameT5_20261021')
shared.TASKS=(559,)

def driver(ns,runner,**kw):
 shared.NS=ns;cfg=kw['cfg'];tid=559
 assert cfg.seed==7 and cfg.EVALUATION.max_steps_override==48
 assert cfg.EVALUATION.replan_steps==8 and cfg.EVALUATION.vjepa2_ac.guidance.enabled
 shared.save('runtime_config.json',ns['OmegaConf'].to_container(cfg,resolve=True))
 bench=ns['benchmark'].get_benchmark_dict()['libero_10'](task_order_index=0)
 ns['set_global_seed'](7,get_worker_init_fn=False)
 env,desc=ns['get_libero_env'](bench.get_task(tid),ns['LIBERO_ENV_RESOLUTION'],7)
 predict=ns['_predict_action_chunk'];times=[]
 try:
  env.reset();ns['apply_libero_camera'](env,cfg.EVALUATION.vjepa2_ac.camera)
  obs=env.set_init_state(bench.get_task_init_states(tid)[0])
  for _ in range(30):obs,_,_,_=env.step(ns['get_libero_dummy_action']())
  args=dict(obs=obs,previous_jepa_image=None,task_description=desc,model=kw['model'],processor=kw['processor'],cfg=cfg,action_horizon=32,input_w=kw['input_w'],input_h=kw['input_h'],model_device=kw['model_device'],executed_horizon=8,replan_index=0)
  shared.CURRENT=dict(seed=7,replan_index=0)
  a=predict(**args,ac_ranker=kw['ac_ranker'],ac_adapter=kw['ac_adapter'])[0]
  b=predict(**args,ac_ranker=kw['ac_ranker'],ac_adapter=kw['ac_adapter'])[0]
  assert np.array_equal(a,b),'Guided repeated inference mismatch'
  assert np.array_equal(a,np.load(OLD/'preflight_task559_guided_actions.npy')),'Historical guided action mismatch'
  digest=shared.observation_digest(obs)
 finally:env.close()
 shared.STAGE='SMOKE'
 def recorded(*a,**k):
  idx=int(k.get('replan_index',0));shared.CURRENT=dict(seed=7,replan_index=idx)
  r=predict(*a,**k);assert np.isfinite(r[0]).all()
  if idx==0:
   assert np.array_equal(r[0],b),'Smoke first action differs from preflight'
   assert shared.observation_digest(k['obs'])==digest
  times.append(r[3]);shared.status('SMOKE_RUNNING',replans=len(times))
  return r
 ns['_predict_action_chunk']=recorded
 ns['set_global_seed'](7,get_worker_init_fn=False)
 args=dict(kw);args.update(task_id=tid,suite_name='libero_10',output_root=OUT/'smoke',worker_id=None)
 path,result=runner(**args)
 assert len(times)>=6 and shared.AUDIT['backwards']>=24
 shared.save('smoke_summary.json',dict(passed=True,task_id=559,seed=7,max_control_steps=48,replans=len(times),result_path=str(path),audit=dict(shared.AUDIT),guided_repeat_exact=True,historical_guided_full32_exact=True,formal_first_action_and_observation_exact=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,timings=times,note='Capacity/inference smoke only; not a task-success evaluation.'))
 shared.status('SMOKE_COMPLETED')
 return result

shared.driver=driver
try:shared.main()
except BaseException:
 shared.save('failure.json',dict(traceback=traceback.format_exc()));raise
