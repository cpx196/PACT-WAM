"""Frozen native Action Expert oracle-future test; no JEPA or correction head.

Reuses full WAM video latents and prompt contexts from a held-out demo cache.
True future is allowed ONLY as an offline oracle condition, never a deployment
input. No training, simulator rollout, new model checkpoint, or global patch.
"""
import argparse
import fcntl
import inspect
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path('/data/pxchen/PACT-WAM')
sys.path.insert(0, str(ROOT))
CACHE = ROOT / 'artifacts/libero10_predictor_action_oneepoch_20260930'
DATA = ROOT / 'artifacts/libero10_guidance_backward_smoke_20260929/data/libero_10'
DEFAULT_OUT = ROOT / 'artifacts/fastwam_native_oracle_future30_20261008'
ARMS = ('wam_native', 'wam_roundtrip', 'real_future_oracle')


def emit(event, **values):
    print(json.dumps(dict(time=time.strftime('%F %T'), event=event, **values)), flush=True)


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(tmp, path)


def load_cache(index):
    return torch.load(CACHE / 'wam' / f'{index:04d}.pt', map_location='cpu', weights_only=False)


def select_windows(per_task):
    rows = json.loads((CACHE / 'windows.json').read_text())
    grouped, skipped = defaultdict(list), []
    for index, row in enumerate(rows):
        if row['split'] != 'val':
            continue
        assert int(row['demo'].split('_')[-1]) % 10 == 0
        cached = load_cache(index)
        assert cached['row'] == row
        kw = cached['kw']
        assert 'video_latents' in kw, 'Need FULL native video cache, not JEPA prefix tokens'
        horizon, frames = int(kw['action_horizon']), int(kw['num_video_frames'])
        assert horizon == 32 and frames == 9, (horizon, frames)
        assert horizon % (frames - 1) == 0
        stride = horizon // (frames - 1)
        offsets = list(range(0, horizon + 1, stride))
        with h5py.File(DATA / row['task_file'], 'r') as file:
            obs = file['data'][row['demo']]['obs']
            lengths = [len(obs[key]) for key in ('agentview_rgb', 'eye_in_hand_rgb')]
        if row['t'] + horizon >= min(lengths):
            skipped.append(dict(index=index, row=row, reason='incomplete 32-step true video; no padding'))
            continue
        grouped[row['task_file']].append(dict(index=index, row=row, frame_offsets=offsets))
    expected_tasks = {r['task_file'] for r in rows}
    assert set(grouped) == expected_tasks, 'At least one task has no complete held-out video'
    selected = []
    for task in sorted(grouped):
        eligible = sorted(grouped[task], key=lambda x: x['row']['t'] / x['row']['n'])
        n = min(per_task, len(eligible))
        positions = np.rint(np.linspace(0, len(eligible) - 1, n)).astype(int)
        assert len(set(positions.tolist())) == n
        selected.extend(eligible[p] for p in positions)
    assert len({(s['row']['task_file'], s['row']['demo']) for s in selected}) == len(selected)
    return dict(selected=selected, skipped=skipped,
                available_per_task={k: len(v) for k, v in sorted(grouped.items())},
                contract=dict(split='held-out demos, demo index modulo 10 == 0',
                              sampling='up to 3 per task spanning relative trajectory position',
                              action_horizon=32, scored_horizon=8, video_frames=9,
                              frame_offsets=[0, 4, 8, 12, 16, 20, 24, 28, 32],
                              no_terminal_frame_padding=True,
                              arms=list(ARMS), frozen=True, guidance=False, post_correction=False,
                              purpose='offline native Action Expert oracle, not rollout success'))


def tensor_to_uint8(image):
    return ((image.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte()


def true_video(sample, cached, cfg, processor, ev, model):
    row = sample['row']
    frames = []
    with h5py.File(DATA / row['task_file'], 'r') as file:
        group = file['data'][row['demo']]
        obs_group = group['obs']
        physical_expert = torch.from_numpy(np.asarray(group['actions'][row['t']:row['t'] + 8], dtype=np.float32))
        cached_expert = cached['expert'].float()
        expected_expert = physical_expert * cached['scale'].float().reshape(-1) + cached['offset'].float().reshape(-1)
        assert torch.allclose(expected_expert, cached_expert, rtol=1e-5, atol=1e-6), 'Expert action/time alignment mismatch'
        for offset in sample['frame_offsets']:
            # Exactly the online image transform; no future proprioceptive data.
            obs = dict(cached['obs'])
            obs['agentview_image'] = np.asarray(obs_group['agentview_rgb'][row['t'] + offset])
            obs['robot0_eye_in_hand_image'] = np.asarray(obs_group['eye_in_hand_rgb'][row['t'] + offset])
            image, _, _ = ev._obs_to_model_input(
                obs, cfg=cfg, processor=processor,
                width=int(cfg.data.train.video_size[1]),
                height=int(cfg.data.train.video_size[0]), device='cuda', dtype=model.torch_dtype)
            frames.append(image)
    video = torch.stack(frames, dim=2)
    anchor = cached['kw']['input_image'].to('cuda', dtype=model.torch_dtype)
    assert torch.equal(video[:, :, 0], anchor), 'Cached current image/orientation mismatch'
    return video


def metrics(action, expert, scale, offset):
    assert action.shape == (32, 7)
    a = action[:8].float()
    e = expert[:8].float()
    physical_a = (a - offset) / scale
    physical_e = (e - offset) / scale
    # Historical cache normalized raw expert gripper +/-1 directly. Continuous
    # coordinates are valid; correct gripper labels explicitly for this audit.
    expert_open_probability = (1.0 - physical_e[:, 6]) / 2.0
    expert_gripper_normalized = expert_open_probability * scale[6] + offset[6]
    executed_gripper = 1.0 - 2.0 * physical_a[:, 6]
    err = a - e
    return dict(mse_first6=float(err[:, :6].square().mean()),
                mse_position_normalized=float(err[:, :3].square().mean()),
                mse_rotation_normalized=float(err[:, 3:6].square().mean()),
                mse_gripper_normalized=float((a[:, 6] - expert_gripper_normalized).square().mean()),
                mse_position_executable=float((physical_a[:, :3] - physical_e[:, :3]).square().mean()),
                mse_rotation_executable=float((physical_a[:, 3:6] - physical_e[:, 3:6]).square().mean()),
                gripper_sign_agreement=float(((executed_gripper >= 0) == (physical_e[:, 6] >= 0)).float().mean()))


def paired_change(baseline, candidate, expert):
    correction = (candidate[:8, :6] - baseline[:8, :6]).float().flatten()
    residual = (expert[:8, :6] - baseline[:8, :6]).float().flatten()
    product = float(correction.norm() * residual.norm())
    cosine = float(torch.dot(correction, residual)) / product if product > 1e-12 else None
    return dict(correction_rms=float(correction.square().mean().sqrt()),
                cosine_toward_expert=cosine)


def aggregate(records):
    means = {arm: float(np.mean([r['arms'][arm]['mse_first6'] for r in records])) for arm in ARMS}
    per_task = {}
    for task in sorted({r['row']['task_file'] for r in records}):
        items = [r for r in records if r['row']['task_file'] == task]
        per_task[task] = dict(windows=len(items), **{
            arm: float(np.mean([r['arms'][arm]['mse_first6'] for r in items])) for arm in ARMS})
    comparisons = {}
    for arm in ARMS[1:]:
        diffs = np.array([r['arms'][arm]['mse_first6'] - r['arms']['wam_native']['mse_first6']
                          for r in records])
        cosines = [r['paired'][arm]['cosine_toward_expert'] for r in records
                   if r['paired'][arm]['cosine_toward_expert'] is not None]
        comparisons[arm] = dict(relative_mse_change_percent=100 * (means[arm] / means['wam_native'] - 1),
                                improved_windows=int((diffs < -1e-9).sum()),
                                worsened_windows=int((diffs > 1e-9).sum()),
                                equal_windows=int((np.abs(diffs) <= 1e-9).sum()),
                                mean_cosine_toward_expert=float(np.mean(cosines)) if cosines else None,
                                mean_correction_rms=float(np.mean([r['paired'][arm]['correction_rms']
                                                                  for r in records])))
    task_weighted = {arm: float(np.mean([x[arm] for x in per_task.values()])) for arm in ARMS}
    return dict(windows=len(records), mean_mse_first6=means, equal_task_weighted_mse=task_weighted,
                paired=comparisons, per_task=per_task,
                caution='True future is privileged offline information; MSE is not task success rate.')


def run(args, manifest):
    from experiments.libero import eval_libero_single as ev
    from fastwam.models.wan22.fastwam_idm import FastWAMIDM
    from scripts.train_guidance_libero10_one_epoch import make_wam
    from PIL import Image

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic()
    model, processor, cfg = make_wam(need_prompts=False)
    assert all(not p.requires_grad for p in model.parameters())
    assert int(cfg.EVALUATION.vjepa2_ac.camera.image_rotation_degrees) == 180
    assert int(cfg.data.train.action_video_freq_ratio) == 4
    assert int(processor.num_output_cameras) == 2
    assert str(cfg.data.train.concat_multi_camera) == 'horizontal'
    assert Path(str(cfg.ckpt)).name == 'libero_optional_idm_2cam224.pt', 'Unexpected FastWAM checkpoint'
    allowed = set(inspect.signature(FastWAMIDM.infer_action).parameters) - {'self'}
    write_json(args.out / 'config.json', dict(
        checkpoint=str(cfg.ckpt), source_cache=str(CACHE), data=str(DATA),
        model_dtype=str(model.torch_dtype), guidance=False, post_module=False,
        compile=False, image_transform='exact _obs_to_model_input rot180, main+wrist horizontal',
        decoder='FastWAM VAE, not JEPA decoder', paired_noise='identical explicit action latent',
        full_configuration=OmegaConf.to_container(cfg, resolve=False),
        selected_windows=len(manifest['selected']), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    emit('MODEL_LOADED',seconds=time.monotonic()-start, checkpoint=str(cfg.ckpt), dtype=str(model.torch_dtype))
    records = []
    for j, sample in enumerate(manifest['selected']):
        begin = time.monotonic()
        cached = load_cache(sample['index'])
        kw = cached['kw'].copy()
        assert kw.pop('action_infer_mode', 'idm') == 'idm'
        generated = kw.pop('video_latents').to('cuda', dtype=model.torch_dtype).clone()
        unknown = set(kw) - allowed
        assert not unknown, f'Unhandled native inference arguments: {unknown}'
        kw = {k: v.to('cuda') if torch.is_tensor(v) else v for k, v in kw.items()}
        assert kw.get('action_guidance_fn') is None
        assert int(kw['num_inference_steps']) == 10
        assert float(kw['video_sigma_shift']) == 1.0 and float(kw['action_sigma_shift']) == 1.0
        kw.update(compile_action_infer=False, action_guidance_fn=None,
                  action_guidance_after_steps=None, action_guidance_last_steps=0,
                  action_guidance_step_size=0.0, action_flow_trace=False,
                  action_response_trace=False, num_action_candidates=1)
        kw.pop('initial_video_latents', None)
        kw.pop('precomputed_video_latents', None)
        generator = torch.Generator(device='cpu').manual_seed(int(kw['seed']))
        noise = torch.randn((1, 32, model.action_expert.action_dim), generator=generator,
                            dtype=torch.float32).to('cuda', dtype=model.torch_dtype)
        kw['initial_action_latents'] = noise.clone()
        with torch.no_grad():
            video = true_video(sample, cached, cfg, processor, ev, model)
            real_latents = model.vae.model.encode(video, model.vae.scale).clone()
            decoded = model.decode_video_latents(generated, tiled=False)
            roundtrip_video = torch.stack([
                torch.from_numpy(np.asarray(frame).copy()).permute(2, 0, 1) for frame in decoded
            ], dim=1).unsqueeze(0).to('cuda', dtype=model.torch_dtype) * (2.0 / 255.0) - 1.0
            # Retain the exact current image, not the lossy VAE reconstruction.
            roundtrip_video[:, :, 0] = kw['input_image']
            roundtrip_latents = model.vae.model.encode(roundtrip_video, model.vae.scale).clone()
            assert generated.shape == real_latents.shape == roundtrip_latents.shape
            current_latent = model._encode_input_image_latents_tensor(kw['input_image'])
            first_frame_rms = float((generated[:, :, :1].float() - current_latent.float()).square().mean().sqrt())
            assert first_frame_rms < 1e-4, 'Native cached first frame is not the same current observation'
            anchor_encoding_rms = float((real_latents[:, :, :1].float() - current_latent.float()).square().mean().sqrt())
            # All conditions use literally identical first-frame latent.
            real_latents[:, :, :1] = generated[:, :, :1]
            roundtrip_latents[:, :, :1] = generated[:, :, :1]
            inputs = dict(wam_native=generated, wam_roundtrip=roundtrip_latents,
                          real_future_oracle=real_latents)
            actions = {}
            for arm in ARMS:
                assert torch.equal(kw['initial_action_latents'], noise)
                result = model.infer_action_from_video(video_latents=inputs[arm], **kw)
                assert not result.get('action_guidance'), 'Guidance must remain disabled'
                actions[arm] = result['action'].cpu().float()
                assert torch.isfinite(actions[arm]).all()
            if j == 0:
                repeat = model.infer_action_from_video(video_latents=generated, **kw)['action'].cpu().float()
                repeat_error = float((repeat-actions['wam_native']).abs().max())
                assert repeat_error <= 1e-5, f'Paired determinism check failed: {repeat_error}'
                emit('SMOKE_PASSED', action_repeat_max_abs=repeat_error,
                     first_frame_rms=first_frame_rms, true_encoding_first_frame_rms=anchor_encoding_rms,
                     video_shape=list(video.shape), latent_shape=list(generated.shape), noise_shape=list(noise.shape))
                args.out.joinpath('preview').mkdir(exist_ok=True)
                for index in (0, 1, 2, 8):
                    rgb = tensor_to_uint8(video[0, :, index]).permute(1, 2, 0).numpy()
                    Image.fromarray(rgb).save(args.out / 'preview' / f'real_input_offset{sample["frame_offsets"][index]:02d}.png')
                    decoded[index].save(args.out / 'preview' / f'wam_decoded_offset{sample["frame_offsets"][index]:02d}.png')
        scale, offset = cached['scale'].float().reshape(-1), cached['offset'].float().reshape(-1)
        expert = cached['expert'].float()
        assert torch.isfinite(scale).all() and (scale.abs() > 1e-8).all()
        record = dict(index=sample['index'], row=sample['row'], seed=int(kw['seed']),
                      frame_offsets=sample['frame_offsets'], seconds=time.monotonic()-begin,
                      first_frame_rms=first_frame_rms,
                      arms={arm: metrics(actions[arm], expert, scale, offset) for arm in ARMS},
                      paired={arm: paired_change(actions['wam_native'], actions[arm], expert)
                              for arm in ARMS[1:]})
        records.append(record)
        torch.save(dict(actions=actions, expert=expert, scale=scale, offset=offset,
                        sample=sample, initial_action_latents=noise.cpu()),
                   args.out / f'paired_{j:03d}.pt')
        with (args.out / 'metrics.jsonl').open('a') as file:
            file.write(json.dumps(record, allow_nan=False) + '\n')
        emit('PAIR_COMPLETED', completed=j+1, total=len(manifest['selected']),
             mse={arm: record['arms'][arm]['mse_first6'] for arm in ARMS}, seconds=record['seconds'])
    summary = aggregate(records)
    summary.update(total_seconds=time.monotonic()-start,
                   peak_gpu_memory_mib=torch.cuda.max_memory_allocated()/1024**2,
                   checkpoint=str(cfg.ckpt))
    write_json(args.out / 'summary.json', summary)
    emit('COMPLETED', **summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--per-task', type=int, default=3)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    assert 1 <= args.per_task <= 3
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.out / 'summary.json').exists():
        emit('ALREADY_COMPLETE', path=str(args.out / 'summary.json'))
        return
    assert not (args.out / 'metrics.jsonl').exists(), 'Partial run: inspect before resuming or use a new output directory'
    manifest = select_windows(args.per_task)
    manifest_path = args.out / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    else:
        write_json(manifest_path, manifest)
    emit('PREFLIGHT_PASSED', windows=len(manifest['selected']), tasks=len(manifest['available_per_task']),
         skipped=len(manifest['skipped']))
    if not args.prepare_only:
        try:
            run(args, manifest)
        except Exception:
            write_json(args.out / 'failure.json', dict(time=time.strftime('%F %T'), traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
