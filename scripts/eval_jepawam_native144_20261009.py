"""Evaluate native JEPA-WAM on PACT's fixed, paired 144-episode manifest."""
import argparse
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
import torch
from experiments.robot.libero import run_libero_eval as ev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    assets = Path('/data/pxchen/PACT-WAM/checkpoints/jepa_wam_assets_20261009')
    policy = assets / 'JEPA_WAM/checkpoints/libero/jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90+n1+b32+x7--visual-cosine-projector-allviews--20260723_232305/checkpoints/step-040000-epoch-37-loss=0.0262.pt'
    cfg = ev.GenerateConfig(
        pretrained_checkpoint=str(policy),
        base_vlm=str(assets / 'JEPA_WAM/checkpoints/pretrained_vlm/prism-qwen25-vjepa21-vitl-384px+0_5b+stage-finetune+x7'),
        llm_checkpoint_path=str(assets / 'Qwen2.5-0.5B'),
        vjepa_checkpoint_path=str(assets / 'vjepa2/vjepa2_1_vitl_dist_vitG_384.pt'),
        task_suite_name='libero_10', max_episode_steps=700, num_steps_wait=5,
        num_open_loop_steps=20, save_rollouts=True, seed=7,
    )
    manifest = json.loads(Path(args.manifest).read_text())
    assert len(manifest) == 144
    assert len({(x['suite'], x['task_id'], x['seed']) for x in manifest}) == 144
    assert {x['seed'] for x in manifest} == {7, 17, 27, 37}
    assert {x['suite'] for x in manifest} == {'libero_10'}
    assert torch.cuda.is_available(), 'GPU execution required'
    torch.set_num_threads(4)
    ev.set_seed_everywhere(7)
    (root / 'run_config.json').write_text(json.dumps({
        'policy': str(policy), 'manifest': manifest, 'physical_gpu': 4,
        'max_steps': 700, 'settling_steps': 5, 'initial_state_index': 0,
        'execute_chunk_steps': 20, 'flow_updates': 4, 'env_resolution': 384,
        'guidance': False, 'trigger': False, 'pact_correction': False,
        'dtype': 'native: BF16 encoder, FP32 stored policy parameters, BF16 CUDA autocast',
        'views': ['agentview', 'wrist'], 'rotation': 'native image rot180, proprio axis-angle',
    }, indent=2))
    completed = []
    result_path = root / 'results.jsonl'
    if result_path.exists():
        completed = [json.loads(line) for line in result_path.read_text().splitlines() if line.strip()]
    done_keys = {(x['suite'], x['task_id'], x['seed']) for x in completed}
    if len(done_keys) == 144:
        print('ALREADY_COMPLETE; no GPU model loaded', flush=True)
        return
    print('START', {'total':144, 'resuming_completed':len(done_keys), 'gpu':torch.cuda.get_device_name()}, flush=True)
    start = time.perf_counter()
    model = ev.get_model(cfg)
    torch.cuda.synchronize()
    model_load_s = time.perf_counter() - start
    print('MODEL_LOADED', {'seconds':model_load_s}, flush=True)
    ev._resolve_unnorm_key(cfg, model)
    suite = ev.benchmark.get_benchmark_dict()['libero_10']()
    metadata = ev._load_task_classification('libero_10')
    original_get_action = ev.get_action
    inference_times = []

    def timed_get_action(*a, **kw):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        actions = original_get_action(*a, **kw)
        torch.cuda.synchronize()
        inference_times.append(time.perf_counter() - begin)
        assert np.isfinite(np.asarray(actions)).all(), 'Nonfinite policy actions'
        return actions

    ev.get_action = timed_get_action
    videos = root / 'videos'
    videos.mkdir(exist_ok=True)
    for entry in manifest:
        key = (entry['suite'], entry['task_id'], entry['seed'])
        if key in done_keys:
            continue
        cfg.seed = entry['seed']
        ev.set_seed_everywhere(cfg.seed)
        task = suite.get_task(entry['task_id'])
        inference_times.clear()
        task_start = time.perf_counter()
        print('EPISODE_START', entry, task.language, flush=True)
        initial_states = suite.get_task_init_states(entry['task_id'])
        env, instruction = ev.get_libero_env(task, resolution=cfg.env_img_res)
        env_setup_s = time.perf_counter() - task_start
        episode_start = time.perf_counter()
        try:
            success, frames = ev._run_episode(cfg, env, instruction, model, initial_states[0])
        finally:
            env.close()
        episode_s = time.perf_counter() - episode_start
        video_start = time.perf_counter()
        video = videos / f"task_{entry['task_id']:04d}_seed_{entry['seed']}_success_{int(success)}.mp4"
        if frames:
            with imageio.get_writer(str(video), fps=30) as writer:
                for frame in frames:
                    writer.append_data(frame)
        record = {**entry, 'initial_state_index':0, 'instruction':instruction,
            'task_name':task.name, 'category':metadata.get(task.name,{}).get('category'),
            'success':bool(success), 'control_steps':len(frames), 'replans':len(inference_times),
            'env_setup_s':env_setup_s, 'episode_s':episode_s,
            'policy_total_s':sum(inference_times), 'policy_mean_s':float(np.mean(inference_times)) if inference_times else None,
            'video_s':time.perf_counter()-video_start, 'total_s':time.perf_counter()-task_start,
            'video':str(video) if frames else None,
            'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
        }
        with result_path.open('a') as output:
            output.write(json.dumps(record)+'\n')
            output.flush()
        completed.append(record)
        done_keys.add(key)
        mean_s = float(np.mean([x['total_s'] for x in completed]))
        summary = {'completed':len(done_keys), 'total':144,
            'successes':sum(x['success'] for x in completed),
            'mean_task_s':mean_s, 'estimated_remaining_s':(144-len(done_keys))*mean_s,
            'model_load_s':model_load_s,
        }
        temp = root / 'progress.json.part'
        temp.write_text(json.dumps(summary, indent=2))
        temp.replace(root / 'progress.json')
        print('EPISODE_DONE', json.dumps(record), flush=True)
        print('PROGRESS', json.dumps(summary), flush=True)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('FULL144_COMPLETE', json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
