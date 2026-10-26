"""Recompute all 126 published episodes and verify fixed-input artifacts; CPU only."""
from pathlib import Path
import argparse,hashlib,json,collections

def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args();root=args.root
 base=root/'reports/research_20261026/robot42';expected={'control':30,'unweighted':30,'weighted':34};folders={'control':'control','unweighted':'unweighted_20261026','weighted':'weighted'}
 tasks=set(range(525,567));results={};preflights={};summaries={}
 for arm,folder in folders.items():
  path=base/folder;files=list((path/'formal/libero_10').glob('*_results.json'));assert len(files)==42,(arm,len(files))
  rows={}
  for p in files:
   x=read(p);assert x['task_id'] not in rows and x['total_episodes']==1 and x['successes'] in (0,1),p
   rows[x['task_id']]=x['successes']
  assert set(rows)==tasks,(arm,set(rows)^tasks)
  summary=read(path/'summary.json');assert summary['completed']==42 and not summary['partial'];assert sum(rows.values())==summary['successes']==expected[arm]
  assert {r['task_id']:r['successes'] for r in summary['rollouts']}==rows
  pre=read(path/'preflight.json')['rows'];assert all(r['passed'] for r in pre);pre={r['task_id']:r for r in pre};assert set(pre)==tasks
  results[arm]=rows;preflights[arm]=pre;summaries[arm]=summary
 for tid in tasks:
  assert len({p[tid]['unguided_full32_sha256'] for p in preflights.values()})==1,('Unguided plan mismatch',tid)
 three=read(base/'unweighted_20261026/three_arm_summary.json');assert three['total_episodes']==126 and len(three['paired_results'])==42
 for row in three['paired_results']:
  for arm in expected:assert row[arm]==results[arm][row['task_id']]
 manifest=read(base/'manifest.json');assert sha(root/'experiments/libero/eval_libero_single.py')==manifest['native_sha256']
 artifacts=read(root/'reproducibility/robot42_20261026/artifact_manifest.json')
 for item in artifacts:
  p=root/item['repository_path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],p
 t5=root/'reproducibility/robot42_20261026/shared_text_contexts.pt';t5hash=sha(t5)
 assert t5hash==read(base/'text_cache_manifest.json')['sha256']==read(base/'unweighted_20261026/method_manifest.json')['t5_sha256']
 audits={}
 for arm in ['weighted','unweighted']:
  folder=base/folders[arm];count=0
  for line in (folder/'gripper_gradient_checks.jsonl').open():
   row=json.loads(line);assert row['gripper_gradient_max']==0,row['task_id'];count+=1
  assert count==summaries[arm]['audit']['backwards'];audits[arm]=count
 for arm,n in expected.items():print(f'{arm}: {n}/42 ({n/42:.2%})')
 for control in ['control','unweighted']:
  rescue=sorted(t for t in tasks if results['weighted'][t] and not results[control][t]);harm=sorted(t for t in tasks if results[control][t] and not results['weighted'][t]);print(f'weighted vs {control}: rescued={rescue}; harmed={harm}; net={len(rescue)-len(harm)}')
 print(f'PASS: 126 formal episodes, 42 paired unguided plans, native-source SHA256, {len(artifacts)} artifact hashes, T5 identity, zero gripper gradient checks {audits}')
if __name__=='__main__':main()
