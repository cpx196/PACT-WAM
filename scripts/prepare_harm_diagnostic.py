"""Freeze a diagnostic cohort from the first 600 paired long800 tasks."""
import collections
import hashlib
import json
from pathlib import Path

source = Path('/data/pxchen/PACT-WAM/evaluate_results/long800_paired_seed7_20260927')
root = Path('/data/pxchen/PACT-WAM/evaluate_results/harm_diagnostic_20260928')
root.mkdir(parents=True, exist_ok=True)
manifest = json.loads((source/'task_manifest.json').read_text())
def records(group):
    return {int(r['task_id']): r for p in (source/group).glob('libero_10/*_results.json')
            for r in [json.loads(p.read_text())]}
control, guided = records('control_noguidance'), records('experiment_rot180_guidance4x')
groups = collections.defaultdict(list)
for item in manifest['tasks'][:600]:
    tid = int(item['task_id'])
    c, g = bool(control[tid]['successes']), bool(guided[tid]['successes'])
    label = 'harm' if c and not g else 'rescue' if g and not c else 'stable_success' if c else 'stable_failure'
    row = dict(item, original_label=label, control_success=c, original_4x_success=g,
               control_duration_s=control[tid]['duration'], original_4x_duration_s=guided[tid]['duration'])
    groups[label].append(row)
assert len(groups['harm']) == 29 and len(groups['rescue']) == 29
all600 = [r for rows in groups.values() for r in rows]
stratified = {}
for field in ('category', 'base_task'):
    stratified[field] = {}
    for value in sorted({r[field] for r in all600}):
        rows = [r for r in all600 if r[field] == value]
        stratified[field][value] = {
            'n': len(rows), 'control_success': sum(r['control_success'] for r in rows),
            'original_4x_success': sum(r['original_4x_success'] for r in rows),
            'harm': sum(r['original_label'] == 'harm' for r in rows),
            'rescue': sum(r['original_label'] == 'rescue' for r in rows)}
# Include 14 stable-success controls spread across base tasks / OOD categories.
selected, counts = [], collections.Counter()
candidates = list(groups['stable_success'])
for _ in range(14):
    row = min(candidates, key=lambda r: (counts[r['base_task']] + counts[r['category']],
                                        hashlib.sha256(str(r['task_id']).encode()).hexdigest()))
    selected.append(row)
    candidates.remove(row)
    counts[row['base_task']] += 1
    counts[row['category']] += 1
groups['stable_success'] = selected
# Interleave sensitive labels so partial results cannot contain only harm.
tasks = []
for index in range(29):
    for label in ('harm', 'rescue', 'stable_success'):
        if index < len(groups[label]):
            tasks.append(groups[label][index])
text = ''.join(f"libero_10,{r['task_id']}\n" for r in tasks)
taskfile = root/'tasks.txt'
if taskfile.exists():
    assert taskfile.read_text() == text, 'Refuse to replace an existing cohort'
else:
    taskfile.write_text(text)
report = {'source': str(source), 'cohort': 'first600 paired; selected diagnostic cohort, NOT unbiased test',
          'counts': dict(collections.Counter(r['original_label'] for r in tasks)),
          'task_list_sha256': hashlib.sha256(text.encode()).hexdigest(), 'tasks': tasks,
          'first600_stratified': stratified,
          'sum_original_4x_duration_hours': sum(r['original_4x_duration_s'] for r in tasks)/3600}
(root/'task_manifest.json').write_text(json.dumps(report, indent=2))
(root/'tasks_statecompat18.txt').write_text(''.join(f"libero_10,{r['task_id']}\n" for r in tasks[:18]))
print(json.dumps({k:v for k,v in report.items() if k != 'tasks'}, indent=2))
