"""Six seed7 guidance rollouts; historical baseline contexts and gripper fix audited."""
from wand.api import library as _wand_library
import functools,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from weighted_ar2_ranker import ARRanker,RawAdapter
ROOT=Path('/data/pxchen/PACT-WAM')
OUT=Path(os.environ['GUIDANCE_SIX_OUT'])
REFERENCE=ROOT/'evaluate_results/t4_seed7_baseline_reference_20261021'
TASKS=(151,242,260,321,559,590)
NS=None
STAGE='PREFLIGHT'
ROWS=[]
CURRENT={}
AUDIT=dict(forwards=0,backwards=0)

def save(name,value):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
def status(state,**kw):
 d=dict(state=state,stage=STAGE,updated=time.strftime('%F %T'),completed=len(ROWS),total=6,**kw);save('progress.json',d);print('GUIDANCE_SIX '+json.dumps(d,ensure_ascii=False),flush=True)
def append(name,value):
 with (OUT/name).open('a') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
def digest(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def observation_digest(obs):
 return {k:digest(obs[k]) for k in ('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')}

def preencode(cfg,*,device,dtype):
 from experiments.libero.text_context_cache import cache_identity,load_cache,record_audit
 manifest=json.loads((REFERENCE/'manifest.json').read_text())
 checksum=os.environ['GUIDANCE_SIX_CACHE_SHA256'];assert manifest['text_cache_sha256']==checksum
 path=Path(manifest['text_cache_path']);identity=manifest['encoding_provenance']
 _,text,_,token=NS['_resolve_configs'](model_id=str(cfg.model.model_id),tokenizer_model_id=str(cfg.model.tokenizer_model_id),redirect_common_files=bool(cfg.model.get('redirect_common_files',True)))
 text.download_if_necessary();token.download_if_necessary()
 actual=cache_identity(text.path,token.path,int(cfg.model.get('tokenizer_max_len',cfg.data.train.get('context_len',128))),dtype)
 for key,value in actual.items():
  if key!='encoding_batch_size':assert identity[key]==value,('Cache assets/config mismatch',key)
 assert identity['encoding_batch_size']=='historical_mixed_1_and_5'
 cache,checksum=load_cache(path,identity,NS['_evaluation_prompts'](cfg),checksum)
 record_audit(OUT,cache,identity,'explicit_frozen_baseline_compatibility',path,checksum)
 return cache

def install_gripper_audit(ns):
 original=ns['_build_jepa_action_guidance']
 @functools.wraps(original)
 def build(**kw):
  energy,horizon=original(**kw);ranker=kw['ac_ranker'];cfg=kw['cfg'];processor=kw['processor']
  ranker.active_task_id=int(cfg.EVALUATION.task_id)
  def checked(clean):
   expected=ns['_denormalize_action'](clean.detach()[:,:horizon],processor)[...,6:7]
   expected=-(expected*2.-1.)
   if cfg.EVALUATION.get('binarize_gripper',False):expected=np.sign(expected)
   check=dict(stage=STAGE,task_id=int(cfg.EVALUATION.task_id),**CURRENT,requires_grad=bool(clean.requires_grad))
   native_energy=ranker.differentiable_energy
   def check_input(*args,**kwargs):
    actual=kwargs['ac_actions'][...,6:7]
    target=torch.as_tensor(expected,device=actual.device,dtype=actual.dtype)
    assert torch.equal(actual,target),'Predictor gripper differs from execution postprocess'
    assert torch.isfinite(actual).all()
    check['predictor_gripper_exact']=True
    return native_energy(*args,**kwargs)
   ranker.differentiable_energy=check_input
   try:loss=energy(clean)
   finally:ranker.differentiable_energy=native_energy
   AUDIT['forwards']+=1;check['loss']=float(loss.detach());append('gripper_forward_checks.jsonl',check)
   if clean.requires_grad:
    def gradient_check(gradient):
     assert torch.isfinite(gradient).all()
     maximum=float(gradient[...,6].detach().abs().max())
     assert maximum==0.,('Nonzero gripper guidance gradient',maximum)
     AUDIT['backwards']+=1
     append('gripper_gradient_checks.jsonl',dict(stage=STAGE,task_id=int(cfg.EVALUATION.task_id),**CURRENT,gripper_gradient_max=maximum,pose_gradient_rms=float(gradient[:,:horizon,:6].detach().square().mean().sqrt())))
     return gradient
    clean.register_hook(gradient_check)
   return loss
  return checked,horizon
 ns['_build_jepa_action_guidance']=build

def summarize(partial=True):
 baseline=json.loads((REFERENCE/'baseline_summary.json').read_text())
 save('summary.json',dict(partial=partial,seed=7,task_ids=list(TASKS),completed=len(ROWS),total=6,successes=sum(r['successes'] for r in ROWS),rollouts=ROWS,baseline_reference=str(REFERENCE),baseline_successes=baseline['successes'],gripper_fix=True,original_ar_gradient_unchanged=True,original_denoising_schedule_unchanged=True,audit=dict(AUDIT),case_set='6task_seed7_sameT5',weight_registry=os.environ['ACTION_WEIGHTS_REGISTRY'],weight_shape=[2,576,1024],beta=1.,loss='weighted_L1',guidance_reference=str(ROOT/'evaluate_results/fastwam_guidance789_ar2_seed7_six_gripperfix_20261021'),note='Paper Eq14/15 per-element weights; existing L1 and AR chain retained. Compare against gripper-fixed pinned-input guidance (1/6) and retained no-guidance (0/6). No online feedback or expert-memory target.'))
 def cell(r):return ('成功' if r['successes'] else '失败')+f"；{r['control_steps']}步/{r['wam_calls']}次WAM" if r else '待完成'
 lines=['# 逐元素动作加权 seed7 六任务 guidance',f'\n完成{len(ROWS)}/6。固定执行8步，guidance节点7/8/9，raw gradient倍率4，RMS上限0.02，保留原AR2梯度链。','\n|Task|保留fixed8基准|逐元素加权guidance|','|---|---|---|']
 for tid in TASKS:
  old=next(r for r in baseline['rollouts'] if r['task_id']==tid)
  lines.append('|'+str(tid)+'|'+('成功' if old['success'] else '失败')+'|'+cell(next((r for r in ROWS if r['task_id']==tid),None))+'|')
 lines+=['\n六任务历史文本缓存固定SHA256；正式运行前核对基准首4步动作、初始状态、启用解码前后动作等价、guided首段重复一致。正式首段动作/观测还须与该任务preflight一致。','\n每次guidance energy核对Predictor夹爪命令等于执行端后处理；每次真实反传确认夹爪梯度精确为0。直接guidance更新不改夹爪，后续原生去噪仍可能间接改变夹爪预测。','\n任务为已选择的六条诊断case，单seed/init，不外推总体成功率。']
 (OUT/'report_zh.md').write_text('\n'.join(lines))

def driver(ns,original_task_runner,**kw):
 global NS,STAGE,CURRENT
 NS=ns;cfg=kw['cfg'];model=kw['model']
 g=cfg.EVALUATION.vjepa2_ac.guidance
 assert cfg.seed==7 and int(cfg.EVALUATION.replan_steps)==8 and kw['action_horizon']==32
 assert list(g.after_flow_steps)==[7,8,9] and g.step_size==4 and g.max_delta_rms==.02 and not g.normalize_gradient and g.ac_steps==2
 assert g.enabled and not g.trigger.enabled and cfg.EVALUATION.binarize_gripper
 assert not cfg.EVALUATION.trigger_probe.enabled and not cfg.EVALUATION.use_action_ensembler
 assert int(cfg.EVALUATION.num_steps_wait)==30 and int(cfg.EVALUATION.num_inference_steps)==10
 save('runtime_config.json',ns['OmegaConf'].to_container(cfg,resolve=True))
 bench=ns['benchmark'].get_benchmark_dict()['libero_10'](task_order_index=0)
 refs=[json.loads(l) for l in (REFERENCE/'baseline_windows.jsonl').read_text().splitlines()]
 predict=ns['_predict_action_chunk'];gold={};checks=[]
 status('VERIFYING_ALL_SIX_BASELINE_INPUTS')
 for tid in TASKS:
  ns['set_global_seed'](7,get_worker_init_fn=False);cfg.EVALUATION.task_id=tid
  env,description=ns['get_libero_env'](bench.get_task(tid),ns['LIBERO_ENV_RESOLUTION'],7)
  try:
   env.reset();ns['apply_libero_camera'](env,cfg.EVALUATION.vjepa2_ac.camera)
   obs=env.set_init_state(bench.get_task_init_states(tid)[0])
   for _ in range(30):obs,_,_,_=env.step(ns['get_libero_dummy_action']())
   base=dict(obs=obs,previous_jepa_image=None,task_description=description,model=model,processor=kw['processor'],cfg=cfg,action_horizon=kw['action_horizon'],input_w=kw['input_w'],input_h=kw['input_h'],model_device=kw['model_device'],executed_horizon=8,replan_index=0)
   reference=next(r for r in refs if r['task_id']==tid and r['control_start']==0)
   assert np.array_equal(ns['_extract_sim_state'](obs),np.asarray(reference['state'],np.float32)),('Initial state mismatch',tid)
   cfg.EVALUATION.vjepa2_ac.enabled=False;cfg.EVALUATION.visualize_future_video=False
   a=predict(**base)[0]
   cfg.EVALUATION.visualize_future_video=True
   b=predict(**base)[0]
   assert np.array_equal(a[:4],np.asarray(reference['commands'],np.float32)),('Historical baseline drift',tid)
   assert np.array_equal(a,b),('Decode path changes native actions',tid)
   cfg.EVALUATION.vjepa2_ac.enabled=True
   CURRENT=dict(replan_index=0)
   start=time.perf_counter();c=predict(**base,ac_ranker=kw['ac_ranker'],ac_adapter=kw['ac_adapter'])[0]
   guided_s=time.perf_counter()-start
   d=predict(**base,ac_ranker=kw['ac_ranker'],ac_adapter=kw['ac_adapter'])[0]
   assert np.array_equal(c,d),('Guidance not reproducible',tid)
   gold[tid]=dict(action=c.copy(),obs=observation_digest(obs))
   np.save(OUT/f'preflight_task{tid}_guided_actions.npy',c)
   row=dict(task_id=tid,initial_state_exact=True,historical_first4_exact=True,decode_on_off_full32_exact=True,guided_repeat_full32_exact=True,guided_replan_s=guided_s,initial_observation_sha256=gold[tid]['obs'],guided_action_sha256=digest(c))
   checks.append(row);save('baseline_preflight.json',dict(passed=False,tasks=checks));status('TASK_PREFLIGHT_PASSED',task_id=tid,guided_replan_s=guided_s)
  finally:
   cfg.EVALUATION.vjepa2_ac.enabled=True;cfg.EVALUATION.visualize_future_video=True;env.close()
 save('baseline_preflight.json',dict(passed=True,tasks=checks))
 assert AUDIT['backwards']>=36 and AUDIT['forwards']>=72
 save('smoke_summary.json',dict(passed=True,all_six_baseline_and_guided_repeat_exact=True,gripper_inputs_exact=True,gripper_gradient_zero=True,audit=dict(AUDIT),mean_guided_replan_s=sum(r['guided_replan_s'] for r in checks)/6,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
 STAGE='FORMAL';CURRENT={};summarize()
 @functools.wraps(predict)
 def recorded(*a,**k):
  global CURRENT
  tid=int(k['cfg'].EVALUATION.task_id);index=int(k.get('replan_index',0));CURRENT=dict(replan_index=index)
  result=predict(*a,**k);commands=result[0]
  assert np.isfinite(commands).all()
  if index==0:
   assert observation_digest(k['obs'])==gold[tid]['obs'],('Formal initial observation changed',tid)
   assert np.array_equal(commands,gold[tid]['action']),('Formal initial guided actions changed',tid)
  append('plans.jsonl',dict(task_id=tid,seed=7,replan_index=index,first8_commands=commands[:8].tolist(),full32_sha256=digest(commands),timing=result[3]))
  if index<3 or index%10==0:status('RUNNING',task_id=tid,replans_completed=index+1,policy_replan_s=result[3].get('total_s'),audit=dict(AUDIT))
  return result
 ns['_predict_action_chunk']=recorded
 for tid in TASKS:
  ns['set_global_seed'](7,get_worker_init_fn=False);cfg.seed=7;cfg.EVALUATION.task_id=tid
  CURRENT={};status('START_EPISODE',task_id=tid)
  task_kw=dict(kw);task_kw.update(task_id=tid,suite_name='libero_10',output_root=OUT/'formal',worker_id=None)
  path,result=original_task_runner(**task_kw)
  timings=result['episode_timings'][0]
  row=dict(task_id=tid,seed=7,trial=0,initial_state_index=0,successes=result['successes'],control_steps=timings['low_level_action_steps'],wam_calls=timings['replan_count'],episode_wall_s=timings['episode_wall_s'],duration=result['duration'],result_path=str(path),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
  ROWS.append(row);save(f'task{tid}_result.json',row);summarize();status('EPISODE_COMPLETED',task_id=tid,success=bool(result['successes']))
 summarize(False);STAGE='COMPLETED';status('COMPLETED',successes=sum(r['successes'] for r in ROWS),audit=dict(AUDIT))
 return dict(successes=sum(r['successes'] for r in ROWS),duration=sum(r['duration'] for r in ROWS))

def main():
 native=ROOT/'experiments/libero/eval_libero_single.py'
 assert hashlib.sha256(native.read_bytes()).hexdigest()==os.environ['GUIDANCE_SIX_NATIVE_SHA256']
 code=native.read_text();needle='if __name__ == "__main__":\n    eval_single_process()';assert code.count(needle)==1
 def install(ns):
  global NS
  NS=ns;ns['VJEPA2ACRanker']=ARRanker;ns['LiberoACAdapter']=RawAdapter
  ns['_evaluation_task_keys']=lambda cfg:[('libero_10',t) for t in TASKS]
  ns['_preencode_prompt_contexts']=preencode
  install_gripper_audit(ns)
  runner=ns['_run_task_to_file'];ns['_run_task_to_file']=lambda **kw:(None,driver(ns,runner,**kw))
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
 code=code.replace(needle,'if __name__ == "__main__":\n    install_guidance_six(globals())\n    eval_single_process()')
 exec(compile(code,str(native),'exec'),dict(__name__='__main__',__file__=str(native),install_guidance_six=install))
if __name__=='__main__':
 try:main()
 except BaseException:save('failure.json',dict(traceback=traceback.format_exc()));status('FAILED');raise
