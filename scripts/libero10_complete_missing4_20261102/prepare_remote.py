from pathlib import Path
import os,json,hashlib,shutil,subprocess,ast
R=Path('/data/pxchen/PACT-WAM');O=R/'evaluate_results/libero10_missing4_robotpose_seed7_20261102';assert not O.exists();O.mkdir();S=O/'source_snapshot';S.mkdir()
old=R/'evaluate_results/libero10_remaining7_robotpose_three_arm_seed7_20261031/source_snapshot'
for name in ['eval_weighted_six.py','weighted_ar2_ranker.py','fastwam_ar2_eval.py','vectorized_ac_attention.py']:shutil.copy2(old/name,S/name)
for name in ['eval_soup_unweighted.py','eval_cream_three.py','launcher.py','merge_results.py']:
 p=Path('/tmp')/name;ast.parse(p.read_text());shutil.copy2(p,S/name)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
# First arm reuses the exact old experiment cache and paired native action fingerprints.
soup=R/'evaluate_results/soup_tomato_robot41_seed7_sameT5_20261028';m=json.loads((soup/'manifest.json').read_text());tc=json.loads((soup/'text_cache_manifest.json').read_text());assert sha(soup/'shared_text_contexts.pt')==tc['sha256']
t1=O/'task01';t1.mkdir();shutil.copy2(soup/'manifest.json',t1/'manifest.json');shutil.copy2(soup/'weights_registry.json',t1/'weights_registry.json')
items=[dict(base_task=m['base_task'],variants=41,first_task_id=289,output=str(t1),parent=str(soup),worker_script='eval_soup_unweighted.py',arms=['unweighted'])]
# Second arm family has its own once-encoded cache, shared by all three groups.
registry=json.loads((R/'evaluate_results/feedback_weights_libero10_remaining7_20261031/weights_registry_all10.json').read_text());w=next(x for x in registry['tasks'] if 'LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box' in x['task']);assert sha(w['path'])==w['sha256']
b=w['task'].removesuffix('_demo');rows=json.loads(Path('/data/pxchen/LIBERO-plus/libero/libero/benchmark/task_classification.json').read_text())['libero_10'];a=[x for x in rows if x['category']=='Robot Initial States' and x['name'].startswith(b+'_')];assert len(a)==39
ref=json.loads((R/'evaluate_results/t4_seed7_baseline_reference_20261021/manifest.json').read_text());native=R/'experiments/libero/eval_libero_single.py';assert sha(native)==m['native_sha256']
t2=O/'task02';t2.mkdir();tasks=[dict(task_id=x['id']-1,name=x['name'],difficulty_level=x['difficulty_level'],seed=7,trial=0,initial_state_index=0) for x in a];w=dict(w);w['case_ids']=[x['task_id'] for x in tasks]
(t2/'weights_registry.json').write_text(json.dumps(dict(complete=True,predictor_sha256=registry['predictor_sha256'],tasks=[w]),indent=2))
(t2/'manifest.json').write_text(json.dumps(dict(base_task=b,tasks=tasks,native_sha256=sha(native),historical_t5_path=ref['text_cache_path'],historical_t5_sha256=ref['text_cache_sha256'],seed=7,total_episodes=117),indent=2))
items.append(dict(base_task=b,variants=39,first_task_id=tasks[0]['task_id'],output=str(t2),parent=str(t2),worker_script='eval_cream_three.py',arms=['weighted','control','unweighted']))
# Source regression: all guided arms use real guidance and native wrapper reset is retained.
for name in ['eval_soup_unweighted.py','eval_cream_three.py']:
 s=(S/name).read_text();assert "WEIGHTED=ARM!='control'" in s and "ns['_soup41_native_predict']=ns['_predict_action_chunk']" in s
assert "for arm in ['unweighted']" in (S/'eval_soup_unweighted.py').read_text()
assert 'torch.ones_like(w)' in (S/'eval_soup_unweighted.py').read_text()
(O/'manifest.json').write_text(json.dumps(dict(seed=7,total_episodes=158,tasks=items,soup_t5_sha256=tc['sha256'],source_sha256={p.name:sha(p) for p in S.glob('*.py')},poll_seconds=5,settings=dict(plan=32,replan=8,denoise=10,max_controls=700,guidance_nodes=[7,8,9],scale=4,delta_rms_cap=.02)),indent=2))
with (O/'launcher.log').open('w') as log:
 child=subprocess.Popen(['/usr/bin/python3',str(S/'launcher.py')],cwd=R,env={**os.environ,'TZ':'Asia/Shanghai'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(launcher_pid=child.pid,total_episodes=158,output=str(O))))
