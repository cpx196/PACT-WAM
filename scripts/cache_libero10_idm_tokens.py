"""Two-shard, resumable frozen-model cache. Full spatial tokens; no training."""
import argparse
import functools
import gc
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/data/pxchen/PACT-WAM')
OUT = ROOT/'artifacts/libero10_idm_cache_stride32_20261004'
MANIFEST = ROOT/'artifacts/libero10_idm_dense_windows_20261004/windows_stride32.json'
DATA = ROOT/'artifacts/libero10_guidance_backward_smoke_20260929/data/libero_10'
SOURCE = ROOT/'checkpoints/vjepa2-ac-libero90-euler-fixed-rot180-step2000-inference.pt'


def emit(event, **kw):
    print(json.dumps(dict(time=time.strftime('%F %T'), event=event, **kw)), flush=True)


def save(data, path):
    tmp = path.with_suffix('.tmp')
    torch.save(data, tmp)
    os.replace(tmp, path)


def verify(item, row):
    assert item['row'] == row
    assert item['action'].shape == (8, 7)
    assert item['expert_executable'].shape == (8, 7)
    assert torch.isfinite(item['action']).all()


def wam(rows, ids):
    from scripts.train_guidance_libero10_one_epoch import make_wam, hdf_sample
    from experiments.libero import eval_libero_single as ev
    started = time.monotonic()
    model, processor, cfg = make_wam(need_prompts=False)
    # These exact ten language instructions were already encoded in the old cache.
    old = ROOT/'artifacts/libero10_predictor_action_oneepoch_20260930'
    old_rows = json.loads((old/'windows.json').read_text())
    prompts = {}
    for i, row in enumerate(old_rows):
        prompt = ev.DEFAULT_PROMPT.format(task=row['language'])
        if prompt not in prompts:
            cached = torch.load(old/'wam'/f'{i:04d}.pt', map_location='cpu', weights_only=False)
            prompts[prompt] = (cached['kw']['context'], cached['kw']['context_mask'])
    assert len(prompts) == 10
    model._libero_prompt_context_cache = prompts
    cfg.EVALUATION.offload_text_encoder = True
    assert not cfg.EVALUATION.vjepa2_ac.enabled
    normalizer = processor.normalizer.normalizers['action'][processor.shape_meta['action'][0]['key']]
    scale = normalizer.scale.detach().float().cpu().reshape(-1)
    offset = normalizer.offset.detach().float().cpu().reshape(-1)
    capture = {}
    original = model.infer_action_from_video
    @functools.wraps(original)
    def wrapped(**kw):
        result = original(**kw)
        action = result['action']
        if action.ndim == 3:
            action = action[0]
        capture['action'] = action[:8].detach().float().cpu()
        return result
    model.infer_action_from_video = wrapped
    emit('WAM_LOADED', seconds=time.monotonic()-started, checkpoint=str(cfg.ckpt))
    done = 0
    loop = time.monotonic()
    for i in ids:
        dest = OUT/'raw'/f'{i:05d}.pt'
        if dest.exists():
            verify(torch.load(dest, map_location='cpu', weights_only=False), rows[i])
            continue
        torch.manual_seed(239+i)
        np.random.seed(239+i)
        cfg.seed = 239+i
        obs, prev, expert = hdf_sample(rows[i])
        capture.clear()
        _, _, future, _ = ev._predict_action_chunk(
            obs, prev, rows[i]['language'], model, processor, cfg,
            action_horizon=int(cfg.data.train.num_frames)-1,
            input_w=int(cfg.data.train.video_size[1]), input_h=int(cfg.data.train.video_size[0]),
            model_device='cuda', executed_horizon=8)
        assert len(future) >= 3 and 'action' in capture
        item = dict(row=rows[i], index=i, seed=239+i, action=capture['action'], obs=obs,
                    context=[prev, np.rot90(obs['agentview_image'], 2).copy()],
                    future=[np.asarray(x) for x in future[:3]], expert_executable=expert,
                    expert_normalized_first6=(expert*scale+offset)[:, :6], scale=scale, offset=offset)
        verify(item, rows[i])
        save(item, dest)
        done += 1
        if done <= 2 or done % 25 == 0:
            emit('WAM_PROGRESS', generated=done, shard_total=len(ids), index=i,
                 seconds=time.monotonic()-loop, seconds_per_window=(time.monotonic()-loop)/done)
    emit('WAM_DONE', generated=done, shard_total=len(ids), seconds=time.monotonic()-started)


def tokens(rows, ids):
    from experiments.libero.vjepa2_ac_ranker import VJEPA2ACRanker, LiberoACAdapter
    started = time.monotonic()
    ranker = VJEPA2ACRanker(checkpoint_path=str(SOURCE), vjepa2_repo='/data/pxchen/vjepa2',
                           device='cuda', dtype=torch.float32, max_ac_steps=2)
    adapter = LiberoACAdapter(low_level_steps_per_ac_step=4)
    emit('JEPA_LOADED', seconds=time.monotonic()-started)
    done = 0
    loop = time.monotonic()
    for i in ids:
        dest = OUT/'tokens'/f'{i:05d}.pt'
        if dest.exists():
            item = torch.load(dest, map_location='cpu', weights_only=False)
            assert item['row'] == rows[i] and item['current'].shape == (256, 1408)
            continue
        raw = torch.load(OUT/'raw'/f'{i:05d}.pt', map_location='cpu', weights_only=False)
        verify(raw, rows[i])
        row = rows[i]
        t = row['t']
        with h5py.File(DATA/row['task_file'], 'r') as file:
            group = file['data'][row['demo']]
            rgb = group['obs']['agentview_rgb']
            real = [np.rot90(np.asarray(rgb[j]), 2).copy() for j in (t, t+4, t+8)]
            ee = torch.tensor(np.asarray(group['obs']['ee_states'][[t, t+4, t+8]]), dtype=torch.float32)
            from scipy.spatial.transform import Rotation
            endpoint_xyz_euler = torch.cat((ee[:, :3], torch.tensor(
                Rotation.from_rotvec(ee[:, 3:6].numpy()).as_euler('xyz'), dtype=torch.float32)), -1)
            expert = torch.tensor(np.asarray(group['actions'][t:t+8]), dtype=torch.float32)
        assert torch.equal(expert, raw['expert_executable'])
        w = raw['future']
        state = torch.tensor(adapter.state_from_observation(raw['obs']))
        a = (raw['action']-raw['offset'])/raw['scale']
        executable = torch.cat((a[:, :6], 1-2*a[:, 6:7]), -1)
        ac = adapter.convert_torch(executable[None], state)
        with torch.no_grad():
            z = ranker.encode_clips([raw['context'], [w[0], w[1]], [w[1], w[2]],
                                    [real[0], real[1]], [real[1], real[2]]])
            assert z.shape == (5, 256, 1408)
            history = z[0].unsqueeze(0)
            predictions = []
            actions, states = ac.actions.cuda(), ac.states.cuda()
            for k in range(2):
                prediction = ranker.predictor(history, actions[:, :k+1], states[:, :k+1])
                prediction = F.layer_norm(prediction[:, -ranker.tokens_per_frame:], (prediction.shape[-1],))
                predictions.append(prediction[0])
                history = torch.cat((history, prediction), dim=1)
            predicted = torch.stack(predictions)
            energy = (predicted-z[1:3]).abs().mean()
            if done == 0:
                reference = ranker.differentiable_energy(current_rep=z[0], target_reps=z[1:3],
                                                        ac_actions=actions, ac_states=states)[0]
                assert abs(float(energy-reference)) < 2e-5
        item = dict(index=i, row=row, current=z[0].float().cpu(), wam_future=z[1:3].float().cpu(),
                    real_future=z[3:5].float().cpu(), predicted_future=predicted.float().cpu(),
                    state=state, ac_actions=ac.actions.cpu(), ac_states=ac.states.cpu(),
                    endpoint_xyz_euler=endpoint_xyz_euler, action=raw['action'],
                    expert_executable=expert, expert_normalized_first6=raw['expert_normalized_first6'],
                    scale=raw['scale'], offset=raw['offset'], energy=float(energy))
        for key in ('current', 'wam_future', 'real_future', 'predicted_future', 'action', 'endpoint_xyz_euler'):
            assert torch.isfinite(item[key]).all(), key
        save(item, dest)
        done += 1
        if done <= 2 or done % 25 == 0:
            emit('TOKENS_PROGRESS', generated=done, shard_total=len(ids), index=i,
                 seconds=time.monotonic()-loop, seconds_per_window=(time.monotonic()-loop)/done,
                 bytes_per_window=dest.stat().st_size, peak_gib=torch.cuda.max_memory_allocated()/2**30)
    emit('TOKENS_DONE', generated=done, shard_total=len(ids), seconds=time.monotonic()-started)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['wam', 'tokens'], required=True)
    p.add_argument('--shard', type=int, required=True)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    assert args.shard in (0, 1)
    rows = json.loads(MANIFEST.read_text())
    assert len(rows) == 4307
    for name in ('raw', 'tokens'):
        (OUT/name).mkdir(parents=True, exist_ok=True)
    from scripts.train_guidance_libero10_one_epoch import config
    from omegaconf import OmegaConf
    metadata = dict(manifest=str(MANIFEST), total=4307, train=3871, val=436,
                    stride=32, action_steps=8, low_level_steps_per_ac_step=4,
                    predictor_steps=2, seed_rule='239 + global window index',
                    jepa_checkpoint=str(SOURCE), tokens_dtype='float32',
                    wam_dtype='bfloat16', image_orientation='rot180, main view only for JEPA',
                    state_orientation='HDF5 rotation vector converted to Euler XYZ',
                    pooling=False, training=False,
                    real_future_usage='supervision only; never supplied to WAM/predictor inference',
                    wam_config=OmegaConf.to_container(config(), resolve=False))
    path = OUT/f'shard{args.shard}_config.json'
    if path.exists():
        assert json.loads(path.read_text()) == metadata
    else:
        path.write_text(json.dumps(metadata, indent=2))
    ids = list(range(args.shard, len(rows), 2))
    if args.limit:
        ids = ids[:args.limit]
    emit('PHASE_START', phase=args.phase, shard=args.shard, total=len(ids))
    (wam if args.phase == 'wam' else tokens)(rows, ids)
    if not args.limit and args.phase == 'tokens':
        for i in ids:
            assert (OUT/'tokens'/f'{i:05d}.pt').exists()
        save(dict(shard=args.shard, completed=len(ids), time=time.strftime('%F %T')),
             OUT/f'shard{args.shard}_complete.pt')


if __name__ == '__main__':
    main()
