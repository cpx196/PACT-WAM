import json
from pathlib import Path

source = Path('/data/pxchen/PACT-WAM/evaluate_results/long800_paired_seed7_20260927')
out = Path('/data/pxchen/PACT-WAM/evaluate_results/harm35_euler_fixed_20260928')
out.mkdir(exist_ok=True)
groups = [{(x['task_suite'], x['task_id']): x
           for p in (source/g).glob('*/*results.json')
           for x in [json.loads(p.read_text())]}
          for g in ('control_noguidance', 'experiment_rot180_guidance4x')]
a, b = groups
keys = sorted(k for k in a.keys() & b.keys() if a[k]['successes'] and not b[k]['successes'])
assert len(keys) == 35, keys
tasks = ''.join(f'{s},{t}\n' for s,t in keys)
path = out/'tasks.txt'
if path.exists():
    assert path.read_text() == tasks
else:
    path.write_text(tasks)
manifest = dict(source=str(source), paired=len(a.keys() & b.keys()), count=35,
                tasks=[dict(suite=s, task_id=t, control=a[(s,t)], old_guidance=b[(s,t)]) for s,t in keys],
                old_duration_hours=sum(b[k]['duration'] for k in keys)/3600,
                selection='control success AND old guidance failure; selected diagnostic cohort, not unbiased accuracy')
(out/'task_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(dict(count=35, task_ids=[t for s,t in keys], old_duration_hours=manifest['old_duration_hours'])))
