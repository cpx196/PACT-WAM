import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

root = Path('/data/pxchen/PACT-WAM/evaluate_results/plus_short315_paired_rot180_20260926')
archive = root / 'restart_archive_1234'
manifest = root / 'task_manifest.json'
data = json.loads(manifest.read_text())
assert hashlib.sha256((root / 'tasks.txt').read_bytes()).hexdigest() == data['settings']['task_list_sha256']
if data['settings']['experiment'].get('gpus') == [2, 3]:
    raise RuntimeError('Topology change already recorded')
for name in ('task_manifest.json', 'TASKS_AND_SETTINGS.md', 'launch_plus_short315_paired.sh'):
    shutil.copy2(root / name, archive / name)
for group in ('control_noguidance', 'experiment_rot180_guidance4x'):
    shutil.copy2(root / group / 'manager_config.yaml', archive / f'{group}_config.yaml')
data['settings']['control']['gpus'] = [1]
initial_gpu = data['settings']['experiment'].pop('gpu')
data['settings']['experiment']['initial_gpu'] = initial_gpu
data['settings']['experiment']['gpus'] = [2, 3]
data['settings']['restart_history'] = [{'time_hkt': datetime.now().isoformat(timespec='seconds'),
    'reason': 'User requested two experimental workers plus one control worker',
    'completed_preserved': {'control': 10, 'experiment': 5},
    'partial_tasks': 'requeued; complete result JSONs are skipped by the manager',
    'parameters_changed': ['MULTIRUN.num_gpus=2 for experiment', 'experimental worker GPUs=2,3'],
    'archive': str(archive)}]
manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
md_path = root / 'TASKS_AND_SETTINGS.md'
md = md_path.read_text().replace('Experiment: GPU 3,', 'Experiment: GPUs 2+3,')
md += '\n\n## Three-GPU restart\n\nControl uses GPU 1; experiment uses GPUs 2 and 3 with a shared task queue. All 315 task IDs and model parameters are unchanged. Ten completed control results and five completed experimental results were retained. In-progress tasks are requeued. Original logs/configurations are retained in restart_archive_1234.\n'
md_path.write_text(md)
print('Metadata updated; task-list checksum unchanged.')
