import pathlib,json,hashlib
R=pathlib.Path('/data/pxchen/PACT-WAM');H=pathlib.Path(__file__).resolve().parent;O=R/'evaluate_results/soup_tomato_robot41_baseline_repair_20261029';BASE=R/'evaluate_results/soup_tomato_robot41_seed7_sameT5_20261028'
O.mkdir(exist_ok=True);(O/'preload').mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
manifest=json.loads((BASE/'manifest.json').read_text());weighted=json.loads((BASE/'weighted/summary.json').read_text());assert weighted['completed']==41 and not weighted['partial']
cache=json.loads((BASE/'text_cache_manifest.json').read_text());assert sha(cache['path'])==cache['sha256']=='e67b010adae11754e4715a615e3203b0ee65257c2cd64db54c1349ceeffa997b'
assert len(list((BASE/'paired_preflight').glob('task*.json')))==41
assert sha(R/'experiments/libero/eval_libero_single.py')==manifest['native_sha256']
reg=json.loads((BASE/'weights_registry.json').read_text());assert all(sha(w['path'])==w['sha256'] for w in reg['tasks'])
(O/'weights_registry.json').write_text(json.dumps(reg,indent=2));(O/'restore_policy.json').write_text(json.dumps(dict(gpu_memory_utilization=.90,cpu_offload_gb=2),indent=2))
(O/'manifest.json').write_text(json.dumps(dict(total_episodes=41,arm_order=['control'],paired_weighted_reference=str(BASE/'weighted/summary.json'),original_run=str(BASE),shared_t5_path=cache['path'],shared_t5_sha256=cache['sha256'],seed=7,task_ids=list(range(289,330)),fix='Reset predictor from native function at arm entry; retain all paired input/action assertions',source_sha256={p.name:sha(p) for p in H.glob('*.py')}),indent=2))
old=json.loads((BASE/'command.json').read_text());args=[('EVALUATION.output_dir='+str(O/'preload')) if a.startswith('EVALUATION.output_dir=') else a for a in old[2:]]
(O/'command.json').write_text(json.dumps(['/data/pxchen/envs/jepa-wam/bin/python',str(H/'eval_soup41.py'),*args],indent=2))
print(json.dumps(dict(prepared=True,total_episodes=41,arm='control',T5_reused=True,T5_sha256=cache['sha256'],weighted_result_unchanged=True)),flush=True)
