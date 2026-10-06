"""CPU-only paired future-source audit; export unmodified source frames for inspection."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr
from scripts.train_libero10_idm_two_stage450 import InverseDynamics, ResidualHead, REAL

ROOT = Path('/data/pxchen/PACT-WAM')
PILOT = ROOT/'artifacts/libero10_idm_oracle_comparison450_20261004'
PRETRAIN = ROOT/'artifacts/libero10_idm_pretrain_then_mlp450_20261004'
BASE = ROOT/'artifacts/libero10_predictor_action_oneepoch_20260930'
DATA = ROOT/'artifacts/libero10_guidance_backward_smoke_20260929/data/libero_10'


def main():
    torch.set_num_threads(4)
    dataset = torch.load(ROOT/'artifacts/libero10_idm21m_mlp_pilot450_20261004/dataset.pt',
                         map_location='cpu', weights_only=False)
    cached = torch.load(PILOT/'latent_actions.pt', map_location='cpu', weights_only=False)
    ids = dataset['val_ids']
    a, y = dataset['action'][ids], dataset['expert'][ids]
    inverse = InverseDynamics().eval()
    ck = torch.load(PRETRAIN/'stage1/best.pt', map_location='cpu', weights_only=False)
    inverse.load_state_dict(ck['model'])
    with torch.no_grad():
        p_real = inverse.head(cached['real_la'][ids]).reshape(-1, 8, 6)
        p_wam = inverse.head(cached['wam_la'][ids]).reshape(-1, 8, 6)
    models = {}
    mean = dataset['action'][dataset['train_ids']].flatten(1).mean(0)
    std = dataset['action'][dataset['train_ids']].flatten(1).std(0).clamp_min(1e-4)
    for name in ('real_oracle', 'wam_future', 'no_jepa'):
        model = ResidualHead(mean, std).eval()
        ck = torch.load(PILOT/name/'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        with torch.no_grad():
            feature = cached['real_la'][ids] if name == 'real_oracle' else cached['wam_la'][ids]
            if name == 'no_jepa':
                feature = torch.zeros_like(feature)
            models[name] = model(a, feature)[0]
            if name == 'real_oracle':
                models['oracle_with_wam_input'] = model(a, cached['wam_la'][ids])[0]
    predictions = dict(idm_real=p_real, idm_wam=p_wam, raw=a[:, :, :6], **models)
    errors = {k: (v-y).square() for k, v in predictions.items()}
    per_sample = {k: v.mean((1, 2)).numpy() for k, v in errors.items()}
    dimensions = {k: v.mean((0, 1)).tolist() for k, v in errors.items()}
    segments = {k: v.reshape(-1, 2, 4, 6).mean((0, 2, 3)).tolist() for k, v in errors.items()}
    token_l1 = []
    for i in ids:
        real = torch.load(REAL/f'{i:04d}.pt', map_location='cpu', weights_only=True)
        token_l1.append(float((real[1:]-dataset['tokens'][i, 1:]).abs().mean()))
    switch_gap = per_sample['oracle_with_wam_input']-per_sample['real_oracle']
    rng = np.random.default_rng(239)
    ci = np.quantile(switch_gap[rng.integers(0, len(ids), size=(10000, len(ids)))].mean(1), [0.025, 0.975])
    tasks = {}
    for task in sorted({dataset['rows'][i]['task_file'] for i in ids}):
        ix = [j for j, i in enumerate(ids) if dataset['rows'][i]['task_file'] == task]
        tasks[task] = {k: float(v[ix].mean()) for k, v in per_sample.items()}
    rows = []
    for j, i in enumerate(ids):
        rows.append(dict(index=i, **dataset['rows'][i], token_l1=token_l1[j],
                         switch_gap=float(switch_gap[j]),
                         errors={k: float(v[j]) for k, v in per_sample.items()}))
    worst = sorted(rows, key=lambda r: r['switch_gap'], reverse=True)[:3]
    exported = []
    frame_dir = PILOT/'inspection_frames'
    frame_dir.mkdir(exist_ok=True)
    for row in worst:
        i, t = row['index'], row['t']
        cached_raw = torch.load(BASE/'wam'/f'{i:04d}.pt', map_location='cpu', weights_only=False)
        assert cached_raw['row'] == dataset['rows'][i]
        # Export original arrays unchanged; orientation differences are documented.
        for k, frame in enumerate(cached_raw['future']):
            path = frame_dir/f'{i:04d}_wam_{k}.png'
            Image.fromarray(np.asarray(frame)).save(path)
            exported.append(str(path))
        with h5py.File(DATA/row['task_file'], 'r') as file:
            rgb = file['data'][row['demo']]['obs']['agentview_rgb']
            for k, tick in enumerate((t, t+4, t+8)):
                path = frame_dir/f'{i:04d}_real_{k}.png'
                Image.fromarray(np.asarray(rgb[tick])).save(path)
                exported.append(str(path))
    report = dict(dimension_order=['x', 'y', 'z', 'rotation_command_1', 'rotation_command_2', 'rotation_command_3'],
                  dimension_mse=dimensions, segment_mse=segments, per_task=tasks,
                  fixed_oracle_switch=dict(mean_mse_gap=float(switch_gap.mean()),
                      worse_windows=int((switch_gap>0).sum()), better_windows=int((switch_gap<0).sum()),
                      paired_window_bootstrap95_mse_gap=ci.tolist(),
                      token_l1_spearman=float(spearmanr(token_l1, switch_gap).statistic)),
                  worst_examples=worst, all_examples=rows, exported_frames=exported,
                  frame_orientation='WAM frames rot180 and two views; HDF5 real images stored upright; exported unchanged',
                  caveat='expert future comparison does not establish whether WAM future is physically invalid')
    (PILOT/'gap_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ('all_examples', 'exported_frames')}, indent=2))


if __name__ == '__main__':
    main()
