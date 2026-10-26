"""Read-only model/data audit; no optimizer, training, or real future input."""
import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from scripts.train_libero10_transformer_la450 import Corrector

ROOT = Path('/data/pxchen/PACT-WAM')
OUT = ROOT/'artifacts/libero10_arb_compare_idm30_vs_zero_20261007'
DATA = ROOT/'artifacts/libero10_guidance_backward_smoke_20260929/data/libero_10'


def main():
    torch.set_num_threads(4)
    started = time.monotonic()
    data = torch.load(OUT/'dense_features.pt', map_location='cpu', weights_only=False)
    assert data['future_source'] == 'compare' and data['idm_epoch'] == 30
    rows, train, val = data['rows'], data['train_ids'], data['val_ids']
    assert {(rows[i]['task_file'], rows[i]['demo']) for i in train}.isdisjoint(
        {(rows[i]['task_file'], rows[i]['demo']) for i in val})
    a, y, la = data['action'], data['expert'], data['la']
    audit_ids = np.linspace(0, len(rows)-1, 20, dtype=int).tolist()
    max_label_error = 0.
    for i in audit_ids:
        row = rows[i]
        item = torch.load(ROOT/'artifacts/libero10_idm_cache_stride32_20261004/tokens'/f'{i:05d}.pt',
                          map_location='cpu', weights_only=False)
        assert item['row'] == row and torch.equal(item['action'], a[i])
        with h5py.File(DATA/row['task_file'], 'r') as f:
            expert = torch.tensor(np.asarray(f['data'][row['demo']]['actions'][row['t']:row['t']+8]))
        assert torch.allclose(expert.float(), item['expert_executable'], atol=1e-6, rtol=1e-6)
        expected = (expert.float()*item['scale']+item['offset'])[:, :6]
        max_label_error = max(max_label_error, float((expected-y[i]).abs().max()))
        assert torch.allclose(expected, y[i], atol=1e-6, rtol=1e-6)
    mean = a[train].flatten(1).mean(0)
    std = a[train].flatten(1).std(0).clamp_min(1e-4)
    model = Corrector(mean, std).eval().requires_grad_(False)

    @torch.no_grad()
    def measure(ids, features=None):
        deltas = []
        for start in range(0, len(ids), 64):
            ix = ids[start:start+64]
            feat = la[ix] if features is None else features[start:start+64]
            _, delta = model(a[ix], feat)
            deltas.append(delta)
        delta = torch.cat(deltas)
        residual = y[ids]-a[ids, :, :6]
        base = float(residual.square().mean())
        d2 = float(delta.square().mean())
        dot = float((delta*residual).mean())
        optimal = dot/max(d2, 1e-12)
        return dict(baseline_mse=base, corrected_mse=base+d2-2*dot,
                    delta_squared_mean=d2, alignment_dot=dot, cross_term=2*dot,
                    optimal_scalar_on_this_set=optimal,
                    mse_at_optimal_scalar=base-2*optimal*dot+optimal*optimal*d2,
                    scalar_sweep={str(s): base-2*s*dot+s*s*d2 for s in [0,.1,.25,.5,.75,1]},
                    correction_rms=float(delta.square().mean((1, 2)).sqrt().mean()),
                    per_dimension_delta2=delta.square().mean((0, 1)).tolist(),
                    per_dimension_dot=(delta*residual).mean((0, 1)).tolist())

    results = {}
    for name in ['best', 'last']:
        ckpt = torch.load(OUT/'compare_la'/f'{name}.pt', map_location='cpu', weights_only=False)
        assert ckpt['config']['idm_epoch'] == 30 and ckpt['config']['real_future_used'] is False
        model.load_state_dict(ckpt['model'])
        assert all(torch.isfinite(p).all() for p in model.parameters())
        result = dict(epoch=ckpt['epoch'], step=ckpt['step'], parameters_finite=True,
                      train=measure(train), val=measure(val))
        rng = torch.Generator().manual_seed(239)
        result['zero_la_inference'] = measure(val, torch.zeros_like(la[val]))
        result['mean_la_inference'] = measure(val, la[train].mean(0, keepdim=True).expand(len(val), -1, -1))
        result['shuffled_la_inference'] = measure(val, la[val][torch.randperm(len(val), generator=rng)])
        results[name] = result
        print(json.dumps(dict(event='CHECKPOINT_AUDITED', checkpoint=name, **result)), flush=True)
    metrics = [json.loads(x) for x in (OUT/'compare_la/metrics.jsonl').read_text().splitlines()]
    historic = ROOT/'artifacts/libero10_transformer_la_stride32_20261004'
    historical_control = None
    if (historic/'zero_la/best.pt').exists():
        old = torch.load(historic/'dense_features.pt', map_location='cpu', weights_only=False)
        assert old['rows'] == rows and old['train_ids'] == train and old['val_ids'] == val
        assert torch.equal(old['action'], a) and torch.equal(old['expert'], y)
        ckpt = torch.load(historic/'zero_la/best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        historical_control = dict(epoch=ckpt['epoch'], config=ckpt['config'],
                                  input_labels_splits_exactly_match=True,
                                  val=measure(val, torch.zeros_like(la[val])),
                                  caveat='Historical independently trained zero-LA control, not a completed variant of this stopped run.')
        print(json.dumps(dict(event='HISTORICAL_ZERO_LA_AUDITED', **historical_control)), flush=True)
    report = dict(label_audit_samples=len(audit_ids), max_label_error=max_label_error,
                  split_overlap=False, config=json.loads((OUT/'config.json').read_text()),
                  checkpoints=results, clipping_rates=[r['clip_rate'] for r in metrics],
                  last_completed_epoch=metrics[-1]['epoch'],
                  zero_la_training_started=(OUT/'zero_la/metrics.jsonl').exists(),
                  historical_zero_la_control=historical_control,
                  caveat='Feature overrides are inference ablations, NOT a separately trained zero-LA control; scalar optima use held-out labels and are diagnostic only.',
                  seconds=time.monotonic()-started)
    (OUT/'stopped_diagnosis.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(event='AUDIT_COMPLETED', seconds=report['seconds'],
                         label_max_error=max_label_error, last_completed_epoch=report['last_completed_epoch'])), flush=True)


if __name__ == '__main__':
    main()
