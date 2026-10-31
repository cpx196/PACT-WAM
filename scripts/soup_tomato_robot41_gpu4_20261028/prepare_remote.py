import json,pathlib,hashlib,shutil
R=pathlib.Path('/data/pxchen/PACT-WAM');H=pathlib.Path(__file__).resolve().parent;O=R/'evaluate_results/soup_tomato_robot41_seed7_sameT5_20261028'
O.mkdir(exist_ok=True);(O/'weighted').mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
b='LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket'
a=[r for r in json.loads(pathlib.Path('/data/pxchen/LIBERO-plus/libero/libero/benchmark/task_classification.json').read_text())['libero_10'] if r['name'].startswith(b+'_') and r['category']=='Robot Initial States']
assert len(a)==41 and [r['id']-1 for r in a]==list(range(289,330))
old=R/'evaluate_results/feedback_weights_6task_seed7_sameT5_20261021';reg=json.loads((old/'offline/weights_registry.json').read_text());w=next(r for r in reg['tasks'] if 321 in r['case_ids'])
assert sha(w['path'])==w['sha256']=='6be17dc8a6fe29e08163d775c6db6d5ae69a0e46d5d357e8ff329cf5d854de4e'
w['case_ids']=list(range(289,330));(O/'weights_registry.json').write_text(json.dumps(dict(complete=True,predictor_sha256=reg['predictor_sha256'],tasks=[w]),indent=2))
ref=json.loads((R/'evaluate_results/t4_seed7_baseline_reference_20261021/manifest.json').read_text());assert sha(ref['text_cache_path'])==ref['text_cache_sha256']
native=R/'experiments/libero/eval_libero_single.py';assert sha(native)=='25d1940d14035d9416ac26e42fb01d7350f61a8a330a09c48fc12c49b9d93d1b'
manifest=dict(base_task=b,perturbation='Robot Initial States',tasks=[dict(task_id=r['id']-1,name=r['name'],difficulty_level=r['difficulty_level'],seed=7,trial=0,initial_state_index=0) for r in a],seed=7,total_episodes=82,arm_order=['weighted','control'],gpu=4,native_sha256=sha(native),historical_t5_path=ref['text_cache_path'],historical_t5_sha256=ref['text_cache_sha256'],fixed_weight_path=w['path'],fixed_weight_sha256=w['sha256'],settings=dict(replan=8,denoise=10,plan=32,max_controls=700,guidance_nodes=[7,8,9],scale=4,delta_rms_cap=.02,beta=1),t5_policy='GPU4 batch1 once after CPU model preload; unload T5 before WAM transfer; preserve historical321 exactly; shared immutable cache across both arms',source_sha256={p.name:sha(p) for p in H.glob('*.py')})
(O/'manifest.json').write_text(json.dumps(manifest,indent=2))
base=R/'evaluate_results/white_mug_robot42_seed7_sameT5_20261025';cmd=json.loads((base/'weighted/command.json').read_text())
changes={'EVALUATION.output_dir':str(O/'weighted'),'EVALUATION.task_id':'289','EVALUATION.vjepa2_ac.device':'cpu'}
args=[a.split('=',1)[0]+'='+changes[a.split('=',1)[0]] if a.split('=',1)[0] in changes else a for a in cmd[2:]]
if not any(a.startswith('EVALUATION.vjepa2_ac.device=') for a in args):args.append('EVALUATION.vjepa2_ac.device=cpu')
(O/'command.json').write_text(json.dumps(['/data/pxchen/envs/jepa-wam/bin/python',str(H/'eval_soup41.py'),*args],indent=2))
print(json.dumps(dict(prepared=True,total_episodes=82,weight_file=w['path'],sha256=w['sha256'],output=str(O))))
