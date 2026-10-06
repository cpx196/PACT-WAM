"""Report sensitive-cohort outcomes separately from active paired-state telemetry."""
import collections
import json
import statistics
from pathlib import Path

root = Path('/data/pxchen/PACT-WAM/evaluate_results/harm_diagnostic_20260928')
manifest = json.loads((root/'task_manifest.json').read_text())
labels = {int(row['task_id']): row['original_label'] for row in manifest['tasks']}
report = {}
for group in ('guidance1x', 'guidance2x', 'guidance4x_protected_gripper',
              'guidance4x_checkpoint_state_compat'):
    folder = root/group
    outcomes = {int(row['task_id']): row for p in folder.glob('libero_10/*_results.json')
                for row in [json.loads(p.read_text())]}
    summary = {'completed': len(outcomes), 'total': 18 if group.endswith('state_compat') else len(labels)}
    for label in ('harm', 'rescue', 'stable_success'):
        subset = [r for tid, r in outcomes.items() if labels[tid] == label]
        summary[label] = {'completed': len(subset), 'successes': sum(int(r['successes']) for r in subset),
                          'tasks': {str(r['task_id']): int(r['successes']) for r in subset}}
    rows = []
    for p in folder.glob('**/diagnostics/*.jsonl'):
        for line in p.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # Reader may encounter an in-progress final write.
    summary['telemetry_including_active_tasks'] = {
        'paired_replans': len(rows), 'unique_tasks': len({r['task_id'] for r in rows}),
        'final_loss_rises': sum(r['final_loss_drop_pct'] < 0 for r in rows),
        'median_final_loss_drop_pct': statistics.median([r['final_loss_drop_pct'] for r in rows]) if rows else None,
        'gripper_binary_switches': sum(r['gripper_binary_switches_executed_prefix'] for r in rows),
        'median_final_delta_rms_7d': [statistics.median(r['final_delta_rms_7d'][i] for r in rows)
                                      for i in range(7)] if rows else None,
    }
    report[group] = summary
(root/'diagnostic_summary.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
