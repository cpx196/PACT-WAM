"""CPU-only window indexing; preserve trajectory-level train/validation split."""
import argparse
import json
from collections import Counter
from pathlib import Path

import h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    assert len(source) == 500
    assert len({(r['task_file'], r['demo']) for r in source}) == 500
    args.out.mkdir(parents=True, exist_ok=True)
    manifests = {stride: [] for stride in (1, 4, 8, 32)}
    trajectory_rows = []
    for task in sorted({r['task_file'] for r in source}):
        with h5py.File(args.data / task, 'r') as file:
            for row in (r for r in source if r['task_file'] == task):
                group = file['data'][row['demo']]
                n = len(group['actions'])
                assert n == row['n'] and group['actions'].shape == (n, 7)
                required = ('agentview_rgb', 'ee_states', 'gripper_states')
                lengths = {key: len(group['obs'][key]) for key in required}
                # t+8 must be a stored real observation, not just an action endpoint.
                upper = min(n - 8, min(lengths.values()) - 8)
                assert upper > 8
                split = 'val' if int(row['demo'].split('_')[-1]) % 10 == 0 else 'train'
                assert split == row['split']
                trajectory_rows.append(dict(task_file=task, demo=row['demo'], split=split,
                                            actions=n, observation_lengths=lengths,
                                            eligible_windows=upper - 8))
                for stride, entries in manifests.items():
                    for t in range(8, upper, stride):
                        entries.append(dict(task_file=task, demo=row['demo'], t=t, n=n,
                                            language=row['language'], split=split))
    summary = dict(trajectories=len(source), tasks=len({r['task_file'] for r in source}),
                   action_steps=sum(r['actions'] for r in trajectory_rows),
                   contract=dict(context_offsets=[-8, 0], future_offsets=[4, 8],
                                 action_slice='[t:t+8]', la_segments=[[0, 4], [4, 8]],
                                 split='demo index modulo 10 == 0 is validation',
                                 selected_stride=32, image_orientation='existing rot180 contract; no images changed',
                                 processing='index only; no image/latent/WAM caches generated'),
                   variants={})
    for stride, entries in manifests.items():
        counts = Counter(r['split'] for r in entries)
        per_task = {}
        for task in sorted({r['task_file'] for r in entries}):
            per_task[task] = dict(Counter(r['split'] for r in entries if r['task_file'] == task))
        summary['variants'][str(stride)] = dict(total=len(entries), **counts, per_task=per_task)
        path = args.out / f'windows_stride{stride}.json'
        # Never overwrite a different existing manifest on repeat execution.
        if path.exists():
            assert json.loads(path.read_text()) == entries, path
        else:
            path.write_text(json.dumps(entries, ensure_ascii=False))
    (args.out / 'trajectory_audit.json').write_text(json.dumps(trajectory_rows, indent=2))
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
