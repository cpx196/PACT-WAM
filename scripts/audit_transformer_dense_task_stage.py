"""CPU-only validation decomposition, paired by demonstration trajectory."""
import json
from pathlib import Path

import numpy as np
import torch
from scripts.train_libero10_transformer_la450 import Corrector

ROOT = Path('/data/pxchen/PACT-WAM')
RUN = ROOT/'artifacts/libero10_transformer_la_stride32_idm29_20261004'
OLD = ROOT/'artifacts/libero10_transformer_la_stride32_20261004'
OUT = ROOT/'artifacts/libero10_transformer_bottleneck_20261007'


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    data = torch.load(RUN/'dense_features.pt', map_location='cpu', weights_only=False)
    old = torch.load(OLD/'dense_features.pt', map_location='cpu', weights_only=False)
    assert data['rows'] == old['rows']
    assert torch.equal(data['action'], old['action']) and torch.equal(data['expert'], old['expert'])
    train, ids = data['train_ids'], data['val_ids']
    assert len(ids) == 436
    a, y = data['action'][ids], data['expert'][ids]
    mean = data['action'][train].flatten(1).mean(0)
    std = data['action'][train].flatten(1).std(0).clamp_min(1e-4)
    predictions = {'raw': a[:, :, :6]}
    for name, source, feature in (
        ('new_idm_la', RUN/'wam_la/best.pt', data['wam_la'][ids]),
        ('no_la', RUN/'zero_la/best.pt', torch.zeros_like(data['wam_la'][ids])),
        ('old_idm_la', OLD/'wam_la/best.pt', old['wam_la'][ids])):
        model = Corrector(mean, std).eval()
        ck = torch.load(source, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        with torch.no_grad():
            predictions[name] = torch.cat([model(a[start:start+32], feature[start:start+32])[0]
                                          for start in range(0, len(ids), 32)])
        assert abs(float((predictions[name]-y).square().mean())-ck['best_val']) < 2e-7
    errors = {name: (pred-y).square().numpy() for name, pred in predictions.items()}
    rows = [data['rows'][i] for i in ids]
    def aggregate(indices):
        return dict(count=len(indices), mse={k: float(v[indices].mean()) for k,v in errors.items()},
                    dimension_mse={k: v[indices].mean((0,1)).tolist() for k,v in errors.items()},
                    segment_mse={k: v[indices].reshape(-1,2,4,6).mean((0,2,3)).tolist()
                                 for k,v in errors.items()})
    task = {name: aggregate([j for j,r in enumerate(rows) if r['task_file']==name])
            for name in sorted({r['task_file'] for r in rows})}
    stage = {}
    for lo,hi in ((0,.25),(.25,.5),(.5,.75),(.75,1.)):
        selected = [j for j,r in enumerate(rows) if lo <= r['t']/r['n'] < hi]
        if selected:
            stage[f'{lo:.2f}-{hi:.2f}'] = aggregate(selected)
    per_window = {k:v.mean((1,2)) for k,v in errors.items()}
    trajectories = sorted({(r['task_file'],r['demo']) for r in rows})
    trajectory_ids = [[j for j,r in enumerate(rows) if (r['task_file'],r['demo'])==key]
                      for key in trajectories]
    rng = np.random.default_rng(239)
    draws = rng.integers(len(trajectories),size=(10000,len(trajectories)))
    counts = np.array([len(ix) for ix in trajectory_ids])
    bootstrap = {}
    for control in ('no_la','raw','old_idm_la'):
        gap = per_window['new_idm_la']-per_window[control]
        sums = np.array([gap[ix].sum() for ix in trajectory_ids])
        sampled = sums[draws].sum(1)/counts[draws].sum(1)
        bootstrap[control] = dict(mean_mse_gap=float(gap.mean()),
                                  trajectory_cluster_bootstrap95=np.quantile(sampled,[.025,.975]).tolist(),
                                  better_windows=int((gap<0).sum()), worse_windows=int((gap>0).sum()))
    records = [dict(index=ids[j], **r, errors={k:float(v[j]) for k,v in per_window.items()})
               for j,r in enumerate(rows)]
    report = dict(overall=aggregate(list(range(len(ids)))), per_task=task, per_stage=stage,
                  stage_definition='demonstration progress t/n, not online replan index',
                  dimensions=['x','y','z','rotation_command_1','rotation_command_2','rotation_command_3'],
                  bootstrap=bootstrap, validation_trajectories=len(trajectories),
                  rows=records, limitation='Single seed; checkpoint selection uses this validation set; descriptive only')
    (OUT/'decomposition.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2),flush=True)


if __name__ == '__main__':
    main()
