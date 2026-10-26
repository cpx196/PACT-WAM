"""Isolated harm ablation; never changes the production evaluation entry point."""
import functools
import inspect
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import hydra
from scipy.spatial.transform import Rotation
from experiments.libero import eval_libero_single as ev

original_build = ev._build_jepa_action_guidance
original_predict = ev._predict_action_chunk
predict_signature = inspect.signature(original_predict)


def mask_gripper(action):
    # Same forward value, no direct JEPA gradient on the binary actuator.
    return torch.cat((action[..., :6], action[..., 6:].detach()), dim=-1)


@functools.wraps(original_build)
def build_guidance(**kwargs):
    energy, horizon = original_build(**kwargs)
    if bool(kwargs['cfg'].EVALUATION.get('diagnostic', {}).get('protect_gripper', False)):
        original_energy = energy
        energy = lambda action: original_energy(mask_gripper(action))
    return energy, horizon


def rms_per_dim(x):
    return x.reshape(-1, x.shape[-1]).square().mean(0).sqrt().tolist()


@functools.wraps(original_predict)
def predict(*args, **kwargs):
    bound = predict_signature.bind(*args, **kwargs)
    bound.apply_defaults()
    ctx = bound.arguments
    cfg, model = ctx['cfg'], ctx['model']
    settings = cfg.EVALUATION.get('diagnostic', {})
    protect = bool(settings.get('protect_gripper', False))
    compatibility_state = bool(settings.get('checkpoint_compat_state', False))
    replan = int(ctx['replan_index'])
    interval = max(1, int(settings.get('paired_every', 4)))
    measure = replan < 4 or replan % interval == 0 or protect
    original_infer = model.infer_action_from_video

    @functools.wraps(original_infer)
    def infer(**infer_kwargs):
        if not measure:
            return original_infer(**infer_kwargs)
        started = time.time()
        guided_kwargs = dict(infer_kwargs, action_flow_trace=True)
        guided = original_infer(**guided_kwargs)
        baseline_kwargs = dict(infer_kwargs)
        baseline_kwargs['action_guidance_fn'] = None
        baseline_kwargs['action_flow_trace'] = False
        # Same explicit seed and precomputed video as the guided pass.
        assert 'seed' in baseline_kwargs, 'Paired measurement requires explicit seed'
        baseline = original_infer(**baseline_kwargs)
        action = guided['action'].clone().float()
        base = baseline['action'].float()
        proposed = action.clone()
        if protect:
            # Masking alone cannot prevent indirect cross-dimensional effects.
            # Preserve the policy's final gripper command at this SAME state.
            action[..., 6] = base[..., 6]
            guided['action'] = action
        horizon = min(8, action.shape[-2])
        ranker = ctx['ac_ranker']
        energy = infer_kwargs['action_guidance_fn']
        def final_energy(a):
            a = a.to(device=ranker.device, dtype=ranker.dtype)
            if a.ndim == 2:
                a = a.unsqueeze(0)
            with torch.no_grad():
                return float(energy(a).item())
        base_loss, guided_loss = final_energy(base), final_energy(action)
        updates = []
        for step in guided.get('action_flow_trace', []):
            if step['guidance_applied']:
                delta = (step['x_after'] - step['x_before'])[..., :horizon, :]
                updates.append({'step': step['step'], 'delta_rms_7d': rms_per_dim(delta),
                                'bf16_zero_fraction': float((delta == 0).float().mean())})
        processor = ctx['processor']
        action_key = processor.shape_meta['action'][0]['key']
        normalizer = processor.normalizer.normalizers['action'][action_key]
        scale = normalizer.scale.detach().float().cpu()
        offset = normalizer.offset.detach().float().cpu()
        raw_base, raw_action = (base-offset)/scale, (action-offset)/scale
        switches = ((raw_base[..., :horizon, 6] > .5) !=
                    (raw_action[..., :horizon, 6] > .5)).sum().item()
        row = {'task_id': int(cfg.EVALUATION.task_id), 'suite': str(cfg.EVALUATION.task_suite_name),
               'replan': replan, 'seed': int(baseline_kwargs['seed']), 'protect_gripper': protect,
               'checkpoint_compat_state': compatibility_state,
               'step_size': float(cfg.EVALUATION.vjepa2_ac.guidance.step_size),
               'baseline_final_loss': base_loss, 'guided_final_loss': guided_loss,
               'final_loss_drop_pct': 100*(base_loss-guided_loss)/max(abs(base_loss), 1e-12),
               'final_delta_rms_7d': rms_per_dim((action-base)[..., :horizon, :]),
               'proposed_delta_rms_7d': rms_per_dim((proposed-base)[..., :horizon, :]),
               'gripper_binary_switches_executed_prefix': int(switches),
               'updates': updates, 'local_diagnostics': guided.get('action_guidance', []),
               'elapsed_s': time.time()-started}
        folder = Path(str(cfg.EVALUATION.output_dir)) / 'diagnostics'
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / f'gpu{cfg.gpu_id}.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        return guided

    model.infer_action_from_video = infer
    adapter = ctx['ac_adapter']
    original_state = adapter.state_from_observation
    if compatibility_state:
        def checkpoint_state(obs):
            state = original_state(obs)
            state[3:6] = Rotation.from_quat(obs['robot0_eef_quat']).as_rotvec()
            return state
        # Compatibility ablation ONLY: deliberately mirrors the old training
        # loader's raw rotvec-as-Euler input, including its later integration.
        # This is NOT the canonical fixed training pipeline.
        adapter.state_from_observation = checkpoint_state
    try:
        return original_predict(*args, **kwargs)
    finally:
        model.infer_action_from_video = original_infer
        adapter.state_from_observation = original_state


ev._build_jepa_action_guidance = build_guidance
ev._predict_action_chunk = predict
if __name__ == '__main__':
    # Imported Hydra entrypoints resolve relative paths as Python modules.
    # Use an absolute config directory without modifying production configs.
    hydra.main(version_base='1.3', config_path=str(ROOT / 'configs'),
               config_name='sim_libero.yaml')(ev.eval_single_process.__wrapped__)()
