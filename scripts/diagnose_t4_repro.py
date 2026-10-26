"""Single-variable diagnostic: fixed model/observation/noise, vary T5 prompt batching."""
import functools,gc,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path('/data/pxchen/PACT-WAM')
OUT=ROOT/'evaluate_results/t4_repro_audit_20261021'
NS=None
GROUPS={}
TEXT_ROWS=[]
TASKS=(151,242,260,321,559,590)

def save(name,data):
 p=OUT/name;q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
def status(state,**kw):
 d=dict(state=state,updated=time.strftime('%F %T'),**kw);save('progress.json',d);print(json.dumps(d),flush=True)
def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def delta(a,b):
 a=a.float();b=b.float()
 return dict(equal=bool(torch.equal(a,b)),max_abs=float((a-b).abs().max()),mean_abs=float((a-b).abs().mean()),different_elements=int((a!=b).sum()),elements=a.numel())

def preencode(cfg,*,device,dtype):
 ns=NS;status('ENCODING_T5_CONTROLLED_GROUPS')
 bench=ns['benchmark'].get_benchmark_dict()['libero_10'](task_order_index=0)
 prompts={tid:ns['DEFAULT_PROMPT'].format(task=bench.get_task(tid).language) for tid in TASKS}
 _,text_cfg,_,token_cfg=ns['_resolve_configs'](model_id=str(cfg.model.model_id),tokenizer_model_id=str(cfg.model.tokenizer_model_id),redirect_common_files=bool(cfg.model.get('redirect_common_files',True)))
 text_cfg.download_if_necessary();token_cfg.download_if_necessary()
 enc=ns['_load_registered_model'](text_cfg.path,'wan_video_text_encoder',torch_dtype=dtype,device=device).eval()
 tokenizer=ns['HuggingfaceTokenizer'](name=token_cfg.path,seq_len=int(cfg.model.get('tokenizer_max_len',cfg.data.train.get('context_len',128))),clean='whitespace')
 specs=[('single151',(151,)),('single151_repeat',(151,)),('single559',(559,)),('old5',(242,260,321,559,590)),('global4',(151,242,559,590)),('six',TASKS)]
 def encode(label,tids):
  with torch.inference_mode():
   ids,mask=tokenizer([prompts[t] for t in tids],return_mask=True,add_special_tokens=True)
   token_hash=digest(ids);mask_hash=digest(mask)
   ids=ids.to(device);mask=mask.to(device=device,dtype=torch.bool)
   context=enc(ids,mask);lens=mask.gt(0).sum(dim=1).long()
   result={}
   for i,tid in enumerate(tids):
    context[i,lens[i]:]=0
    result[tid]=(context[i:i+1].to(device='cpu',dtype=dtype).contiguous().clone(),torch.ones_like(mask[i:i+1],device='cpu'))
   TEXT_ROWS.append(dict(group=label,tasks=list(tids),token_shape=list(ids.shape),context_shape=list(context.shape),token_hash=token_hash,mask_hash=mask_hash,contexts={str(t):digest(result[t][0]) for t in tids}))
   status('T5_GROUP_ENCODED',group=label,tasks=list(tids))
   return result
 for label,tids in specs:GROUPS[label]=encode(label,tids)
 GROUPS['six_sequential']={}
 for tid in TASKS:GROUPS['six_sequential'].update(encode(f'canonical_single_{tid}',(tid,)))
 comparisons=[]
 for tid in (151,559):
  base=GROUPS['single151' if tid==151 else 'single559'][tid][0]
  for label,group in GROUPS.items():
   if tid in group:comparisons.append(dict(task_id=tid,group=label,vs='singleton',**delta(base,group[tid][0])))
 save('text_context_audit.json',dict(groups=TEXT_ROWS,comparisons=comparisons,prompts=prompts))
 torch.save(dict(prompts=prompts,groups=GROUPS),OUT/'controlled_contexts.pt')
 del enc,tokenizer;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 return {prompts[tid]:GROUPS['six_sequential'][tid] for tid in TASKS}

def driver(ns,**kw):
 cfg=kw['cfg'];model=kw['model'];processor=kw['processor']
 cfg.seed=7
 bench=ns['benchmark'].get_benchmark_dict()['libero_10'](task_order_index=0)
 ns['OmegaConf'].save(cfg,OUT/'runtime_config.yaml',resolve=True)
 torch.save({ns['DEFAULT_PROMPT'].format(task=bench.get_task(t).language):GROUPS['six_sequential'][t] for t in TASKS},OUT/'canonical_context_cache.pt')
 specs={151:[('single151','old'),('global4','global'),('six','task'),('six_sequential',None),('single151_repeat',None)],559:[('old5','old'),('global4','global'),('six','task'),('six_sequential',None),('single559',None)]}
 previous={151:dict(old='fastwam_t4_seed7_task151_20261020',global_='fastwam_t4_seed7_recalibrated_a48_20261020',task='fastwam_t4_per_task_offline_calibration_20261020'),559:dict(old='fastwam_t4_seed7_remaining5_20261020',global_='fastwam_t4_seed7_recalibrated_a48_20261020',task='fastwam_t4_per_task_offline_calibration_20261020')}
 reference={}
 for tid in (151,559):
  for mode,folder in previous[tid].items():
   arm='fixed8' if mode=='old' else 'adaptive48'
   rows=[json.loads(l) for l in (ROOT/'evaluate_results'/folder/'windows.jsonl').read_text().splitlines()]
   w=next(w for w in rows if w['task_id']==tid and w['seed']==7 and w['arm']==arm and w['control_start']==0)
   reference[tid,'global' if mode=='global_' else mode]=w
 results=[]
 for tid in (151,559):
  ns['set_global_seed'](7,get_worker_init_fn=False);cfg.EVALUATION.task_id=tid
  env,description=ns['get_libero_env'](bench.get_task(tid),ns['LIBERO_ENV_RESOLUTION'],7)
  try:
   env.reset();ns['apply_libero_camera'](env,cfg.EVALUATION.vjepa2_ac.camera);obs=env.set_init_state(bench.get_task_init_states(tid)[0])
   for _ in range(30):obs,_,_,_=env.step(ns['get_libero_dummy_action']())
   assert np.array_equal(ns['_extract_sim_state'](obs),np.array(reference[tid,'old']['state'],np.float32)),'Initial state mismatch'
   np.savez_compressed(OUT/f'initial_task{tid}_obs.npz',**{k:obs[k] for k in ('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')})
   imgs=ns['get_libero_image'](obs,image_rotation_degrees=180)
   Image.fromarray(ns['_agentview_for_jepa'](imgs,processor)).save(OUT/f'task{tid}_primary.png')
   Image.fromarray(imgs['wrist_image']).save(OUT/f'task{tid}_wrist.png')
   original=model.infer_action;trace={}
   @functools.wraps(original)
   def traced(*a,**k):
    trace.clear()
    for key in ('input_image','proprio','context','context_mask'):
     if torch.is_tensor(k.get(key)):trace[key]=dict(shape=list(k[key].shape),dtype=str(k[key].dtype),sha256=digest(k[key]))
    trace.update(seed=k.get('seed'),rand_device=k.get('rand_device'))
    output=original(*a,**k);trace['normalized_action_sha256']=digest(output['action']);return output
   model.infer_action=traced
   prompt=ns['DEFAULT_PROMPT'].format(task=description)
   for group,refkey in specs[tid]:
    actions=[];traces=[]
    for repeat in range(2):
     ns['set_global_seed'](7,get_worker_init_fn=False)
     cache=dict(model._libero_prompt_context_cache);cache[prompt]=GROUPS[group][tid];model._libero_prompt_context_cache=cache
     action,_,frames,_=ns['_predict_action_chunk'](obs=obs,previous_jepa_image=None,task_description=description,model=model,processor=processor,cfg=cfg,action_horizon=kw['action_horizon'],input_w=kw['input_w'],input_h=kw['input_h'],model_device=kw['model_device'],executed_horizon=8,replan_index=0)
     assert frames is None
     actions.append(action.copy());traces.append(dict(trace))
    r=dict(task_id=tid,group=group,repeats_equal=bool(np.array_equal(actions[0],actions[1])),repeat_max_abs=float(np.abs(actions[0]-actions[1]).max()),input_traces=traces,first4_commands=actions[0][:4].tolist())
    if refkey:r.update(historical_reference=refkey,historical_max_abs=float(np.abs(actions[0][:4]-np.array(reference[tid,refkey]['commands'],np.float32)).max()))
    results.append(r);np.save(OUT/f'task{tid}_{group}_actions.npy',actions[0]);save('action_audit.json',results);status('ACTION_GROUP_COMPLETED',task_id=tid,group=group,repeats_equal=r['repeats_equal'],historical_max_abs=r.get('historical_max_abs'))
   model.infer_action=original
  finally:env.close()
 save('summary.json',dict(text=json.loads((OUT/'text_context_audit.json').read_text()),action=results,torch_version=torch.__version__,cuda=torch.version.cuda,flags=dict(cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)))
 status('COMPLETED');return dict(successes=0,duration=0)

def main():
 source=ROOT/'experiments/libero/eval_libero_single.py';code=source.read_text()
 needle='if __name__ == "__main__":\n    eval_single_process()';assert code.count(needle)==1
 def install(ns):
  global NS
  NS=ns;ns['_preencode_prompt_contexts']=preencode;ns['_run_task_to_file']=lambda **kw:(None,driver(ns,**kw))
 code=code.replace(needle,'if __name__ == "__main__":\n    install_repro(globals())\n    eval_single_process()')
 exec(compile(code,str(source),'exec'),dict(__name__='__main__',__file__=str(source),install_repro=install))
if __name__=='__main__':
 try:main()
 except BaseException:save('failure.json',dict(traceback=traceback.format_exc()));status('FAILED');raise
