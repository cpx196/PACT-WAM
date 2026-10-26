import json,hashlib,shutil
from pathlib import Path
import torch
from experiments.libero.text_context_cache import save_cache,entry_fingerprints
ROOT=Path('/data/pxchen/PACT-WAM')
BASE=ROOT/'evaluate_results/t4_repro_audit_20261021'
OUT=ROOT/'evaluate_results/t4_seed7_baseline_reference_20261021'
OUT.mkdir(exist_ok=False)
controlled=torch.load(BASE/'controlled_contexts.pt',map_location='cpu',weights_only=True)
prompts=controlled['prompts'];groups=controlled['groups']
cache={prompts[t]:groups['single151' if t==151 else 'old5'][t] for t in (151,242,260,321,559,590)}
identity=json.loads((BASE/'frozen_canonical_text_cache.pt.json').read_text())['identity']
# This is a compatibility cache reproducing the historical batch paths, not newly encoded batch1 text.
identity={**identity,'encoding_batch_size':'historical_mixed_1_and_5','encoding_provenance':'task151=single151; remaining five=old5; controlled_contexts.pt'}
sha=save_cache(cache,OUT/'historical_baseline_text_contexts.pt',identity)
rows=[];windows=[];sources={}
for folder in ('fastwam_t4_seed7_task151_20261020','fastwam_t4_seed7_remaining5_20261020'):
 p=ROOT/'evaluate_results'/folder
 for name in ('summary.json','windows.jsonl','command.json','manifest.json'):
  f=p/name
  if f.exists():sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
 s=json.loads((p/'summary.json').read_text())
 rows.extend({**r,'source_dir':str(p)} for r in s['rollouts'] if r['arm']=='fixed8')
 windows.extend(json.loads(line) for line in (p/'windows.jsonl').read_text().splitlines() if json.loads(line)['arm']=='fixed8')
assert len(rows)==6 and {r['task_id'] for r in rows}=={151,242,260,321,559,590}
assert all(r['seed']==7 and r['trial']==0 and r['initial_state_index']==0 for r in rows)
(OUT/'baseline_summary.json').write_text(json.dumps(dict(seed=7,guidance=False,execution_horizon=8,plan_horizon=32,successes=sum(r['success'] for r in rows),total=6,rollouts=rows),ensure_ascii=False,indent=2))
(OUT/'baseline_windows.jsonl').write_text(''.join(json.dumps(w,ensure_ascii=False)+'\n' for w in windows))
manifest=dict(status='FROZEN_HISTORICAL_BASELINE_REFERENCE',user_selected=True,task_ids=[151,242,260,321,559,590],seed=7,trial=0,initial_state_index=0,guidance=False,arm='fixed8',execution_horizon=8,plan_horizon=32,source_sha256=sources,text_cache_path=str(OUT/'historical_baseline_text_contexts.pt'),text_cache_sha256=sha,text_context_fingerprints=entry_fingerprints(cache),encoding_provenance=identity,first_action_historical_reproduction_verified_tasks=[151,559],remaining_tasks_reproduction_pending=[242,260,321,590],full_trajectory_reproduction_verified=False,comparison_requirement='Use exact historical cached tensors and matched observations/seed/model settings; verify matching initial actions before effectiveness comparison.',prior_cross_cohort_effectiveness_claims_still_invalid=True)
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps(dict(status=manifest['status'],cache_sha256=sha,windows=len(windows),tasks=len(rows))))
