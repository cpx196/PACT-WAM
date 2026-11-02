from pathlib import Path
import ast,json,hashlib,shutil,subprocess,os
R=Path('/data/pxchen/PACT-WAM');OUT=R/'evaluate_results/libero10_remaining7_robotpose_three_arm_seed7_20261031'
assert not OUT.exists();OUT.mkdir();SRC=OUT/'source_snapshot';SRC.mkdir()
old=R/'evaluate_results/soup_tomato_robot41_baseline_repair_20261029/source_snapshot'
if not old.exists():old=R/'scripts/soup_tomato_robot41_baseline_repair_20261029'
for name in ['eval_weighted_six.py','weighted_ar2_ranker.py','fastwam_ar2_eval.py','vectorized_ac_attention.py']:
 p=old/name;assert p.exists(),p;shutil.copy2(p,SRC/name)
for name in ['eval_three.py','launcher.py']:
 p=Path('/tmp')/name;ast.parse(p.read_text());shutil.copy2(p,SRC/name)
# Regression of the wrapper-reset mechanism that broke the earlier baseline.
tree=ast.parse((SRC/'eval_three.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='native_predictor')
ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<reset-test>','exec'),ns)
f=lambda:None;d={'_soup41_native_predict':f,'_predict_action_chunk':lambda:1}
for _ in range(3):
 assert ns['native_predictor'](d) is f and d['_predict_action_chunk'] is f
 d['_predict_action_chunk']=lambda:2
(OUT/'cpu_checks.json').write_text(json.dumps(dict(python_syntax=True,native_predictor_reset_three_arms=True)))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
native=R/'experiments/libero/eval_libero_single.py';assert sha(native)=='25d1940d14035d9416ac26e42fb01d7350f61a8a330a09c48fc12c49b9d93d1b'
reg=json.loads((R/'evaluate_results/feedback_weights_libero10_remaining7_20261031/offline/weights_registry.json').read_text());assert reg['complete'] and len(reg['tasks'])==7
rows=json.loads(Path('/data/pxchen/LIBERO-plus/libero/libero/benchmark/task_classification.json').read_text())['libero_10']
historical=json.loads((R/'evaluate_results/t4_seed7_baseline_reference_20261021/manifest.json').read_text())
items=[]
for i,w in enumerate(reg['tasks']):
 assert sha(w['path'])==w['sha256']
 b=w['task'].removesuffix('_demo');a=[r for r in rows if r['name'].startswith(b+'_') and r['category']=='Robot Initial States'];assert a
 taskout=OUT/f'task{i+1:02d}';taskout.mkdir()
 tasks=[dict(task_id=r['id']-1,name=r['name'],difficulty_level=r['difficulty_level'],seed=7,trial=0,initial_state_index=0) for r in a]
 weight=dict(w);weight['case_ids']=[r['task_id'] for r in tasks]
 (taskout/'weights_registry.json').write_text(json.dumps(dict(complete=True,predictor_sha256=reg['predictor_sha256'],tasks=[weight]),indent=2))
 m=dict(base_task=b,tasks=tasks,seed=7,total_episodes=len(a)*3,arm_order=['weighted','control','unweighted'],native_sha256=sha(native),historical_t5_path=historical['text_cache_path'],historical_t5_sha256=historical['text_cache_sha256'],weight_sha256=w['sha256'],t5_policy='GPU batch1 once, preserve historical overlapping prompts, unload T5, immutable cache shared across three arms')
 (taskout/'manifest.json').write_text(json.dumps(m,indent=2))
 items.append(dict(base_task=b,variants=len(a),first_task_id=tasks[0]['task_id'],output=str(taskout)))
assert sum(x['variants'] for x in items)==271
(OUT/'manifest.json').write_text(json.dumps(dict(seed=7,total_episodes=813,total_variants=271,tasks=items,source_sha256={p.name:sha(p) for p in SRC.glob('*.py')},settings=dict(replan=8,plan=32,denoise=10,max_controls=700,guidance_nodes=[7,8,9],scale=4,delta_rms_cap=.02)),indent=2))
with (OUT/'launcher.log').open('w') as log:
 p=subprocess.Popen(['/usr/bin/python3',str(SRC/'launcher.py')],cwd=R,env={**os.environ,'TZ':'Asia/Shanghai'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(launcher_pid=p.pid,output=str(OUT),episodes=813)))
