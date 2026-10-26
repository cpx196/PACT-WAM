"""Fresh-process regression against saved initial observations, no rollouts."""
import functools,json,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/data/pxchen/PACT-WAM')
BASE=ROOT/'evaluate_results/t4_repro_audit_20261021'
OUT=ROOT/'evaluate_results/t4_repro_repaired_regression_20261021'
TASKS=(151,242,260,321,559,590)
def save(name,value):
 (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
def driver(ns,**kw):
 from experiments.libero.text_context_cache import load_cache,cache_identity,tensor_sha256
 cfg=kw['cfg'];cfg.seed=7;model=kw['model'];processor=kw['processor']
 ns['OmegaConf'].save(cfg,OUT/'runtime_config.yaml',resolve=True)
 bench=ns['benchmark'].get_benchmark_dict()['libero_10'](task_order_index=0)
 prompts={t:ns['DEFAULT_PROMPT'].format(task=bench.get_task(t).language) for t in TASKS}
 path=Path(cfg.EVALUATION.text_context_cache_path)
 identity=json.loads(path.with_suffix('.pt.json').read_text())['identity']
 results=[]
 for tid in (151,559):
  obs=dict(np.load(BASE/f'initial_task{tid}_obs.npz'))
  description=bench.get_task(tid).language
  reference=np.load(BASE/f'task{tid}_six_sequential_actions.npy')
  groups=[(tid,), (151,242,559,590), (242,260,321,559,590), TASKS]
  for tids in groups:
   if tid not in tids:continue
   cache,_=load_cache(path,identity,[prompts[t] for t in tids],cfg.EVALUATION.text_context_cache_sha256)
   model._libero_prompt_context_cache=cache
   for repeat in range(2):
    ns['set_global_seed'](7,get_worker_init_fn=False)
    action,_,frames,_=ns['_predict_action_chunk'](obs=obs,previous_jepa_image=None,task_description=description,model=model,processor=processor,cfg=cfg,action_horizon=kw['action_horizon'],input_w=kw['input_w'],input_h=kw['input_h'],model_device=kw['model_device'],executed_horizon=8,replan_index=0)
    row=dict(task_id=tid,group=list(tids),repeat=repeat,action_shape=list(action.shape),equal=bool(np.array_equal(action,reference)),max_abs=float(np.abs(action-reference).max()),context_sha256=tensor_sha256(cache[prompts[tid]][0]))
    results.append(row);save('action_regression.json',results)
    print(json.dumps(row),flush=True)
    assert row['equal'],row
 save('summary.json',dict(status='PASSED',cross_process_full_action_equal=True,model_construction_seed=57,inference_seed=7,prior_process_model_construction_seed=7,checks=len(results),results=results,completed=time.strftime('%F %T')))
 save('progress.json',dict(state='COMPLETED',updated=time.strftime('%F %T')))
 return dict(successes=0,duration=0)
def main():
 source=ROOT/'experiments/libero/eval_libero_single.py';code=source.read_text()
 needle='if __name__ == "__main__":\n    eval_single_process()';assert code.count(needle)==1
 def install(ns):
  ns['_evaluation_task_keys']=lambda cfg:[('libero_10',t) for t in TASKS]
  ns['_run_task_to_file']=lambda **kw:(None,driver(ns,**kw))
  old=ns['_load_model_checkpoint']
  def checked(model,ckpt):
   original=model.mot.load_state_dict
   def load(*a,**k):
    result=original(*a,**k)
    save('checkpoint_keys.json',dict(missing_keys=result.missing_keys,unexpected_keys=result.unexpected_keys))
    assert not result.missing_keys,result.missing_keys
    return result
   model.mot.load_state_dict=load
   try:return old(model,ckpt)
   finally:model.mot.load_state_dict=original
  ns['_load_model_checkpoint']=checked
 code=code.replace(needle,'if __name__ == "__main__":\n    install_repro(globals())\n    eval_single_process()')
 exec(compile(code,str(source),'exec'),dict(__name__='__main__',__file__=str(source),install_repro=install))
if __name__=='__main__':
 try:main()
 except BaseException:
  save('failure.json',dict(traceback=traceback.format_exc()));raise
