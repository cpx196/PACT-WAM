"""Isolated real-data, real-WAM guidance double-backward smoke. Never exports production weights."""
import argparse, functools, gc, json, logging, sys, traceback
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from omegaconf import OmegaConf

ROOT = Path('/data/pxchen/PACT-WAM')
OUT = ROOT/'artifacts/libero10_guidance_backward_smoke_20260929'
sys.path.insert(0, str(ROOT))

def log(kind, **kw):
    print(json.dumps(dict(event=kind, **kw)), flush=True)

def cache_wam():
    from experiments.libero import eval_libero_single as ev
    cfg = OmegaConf.load(ROOT/'evaluate_results/guidance789_rawgrad4x_long36_full144_euler_fixed_step2000_20260929/seed_7/manager_config.yaml')
    cfg.EVALUATION.output_dir = str(OUT/'cache')
    cfg.EVALUATION.vjepa2_ac.enabled = False
    cfg.EVALUATION.vjepa2_ac.guidance.enabled = False
    cfg.EVALUATION.vjepa2_ac.num_action_candidates = 1
    cfg.EVALUATION.trigger_probe.enabled = False
    cfg.EVALUATION.action_denoise_trace = False
    cfg.MULTIRUN.task_file = None
    description = 'pick up the book and place it in the back compartment of the caddy'
    ev._evaluation_prompts = lambda cfg: [ev.DEFAULT_PROMPT.format(task=description)]
    manifest = json.loads((OUT/'data_manifest.json').read_text())

    def collect(**ctx):
        model, processor = ctx['model'], ctx['processor']
        model.requires_grad_(False)
        captured = {}
        original = model.infer_action_from_video
        @functools.wraps(original)
        def traced(**kw):
            result = original(**dict(kw, action_flow_trace=True))
            captured['trace'] = result['action_flow_trace'][-1]
            assert captured['trace']['step'] == 9
            return result
        model.infer_action_from_video = traced
        key = processor.shape_meta['action'][0]['key']
        norm = processor.normalizer.normalizers['action'][key]
        scale, offset = norm.scale.detach().float().cpu().reshape(-1), norm.offset.detach().float().cpu().reshape(-1)
        records = []
        with h5py.File(manifest['source'], 'r') as f:
            for window in manifest['windows']:
                g = f['data'][window['demo']]; t = window['t']; o = g['obs']
                ee = np.asarray(o['ee_states'][t])
                obs = dict(agentview_image=np.asarray(o['agentview_rgb'][t]),
                           robot0_eye_in_hand_image=np.asarray(o['eye_in_hand_rgb'][t]),
                           robot0_eef_pos=ee[:3], robot0_eef_quat=Rotation.from_rotvec(ee[3:6]).as_quat(),
                           robot0_gripper_qpos=np.asarray(o['gripper_states'][t]))
                _, _, future, _ = ev._predict_action_chunk(
                    obs, np.rot90(np.asarray(o['agentview_rgb'][t-8]),2).copy(), description,
                    model, processor, cfg, action_horizon=ctx['action_horizon'],
                    input_w=ctx['input_w'], input_h=ctx['input_h'], model_device=ctx['model_device'],
                    executed_horizon=8)
                trace = captured['trace']
                expert = torch.from_numpy(np.asarray(g['actions'][t:t+8],dtype=np.float32))
                expert_norm = expert*scale+offset
                records.append(dict(window=window, obs=obs,
                    context=[np.rot90(np.asarray(o['agentview_rgb'][t-8]),2).copy(), np.rot90(obs['agentview_image'],2).copy()],
                    wam_future=[np.asarray(x) for x in future[:3]],
                    real_future=[np.rot90(np.asarray(o['agentview_rgb'][i]),2).copy() for i in [t,t+4,t+8]],
                    clean=trace['clean_action_estimate'].float(), base=trace['x_base'].float(),
                    expert=expert_norm, scale=scale, offset=offset))
                log('WAM_WINDOW_CACHED', demo=window['demo'],step=int(trace['step']),future_frames=len(future))
        torch.save(records, OUT/'wam_windows.pt')
        log('WAM_CACHE_PASSED', windows=len(records))
        return None, dict(successes=0,duration=0)
    ev._run_task_to_file = collect
    ev.eval_single_process(cfg)

def backward_smoke():
    from experiments.libero.vjepa2_ac_ranker import VJEPA2ACRanker, LiberoACAdapter
    from scripts.train_vjepa2_ac_libero90_euler_fixed import loss_fn
    torch.manual_seed(239)
    ranker = VJEPA2ACRanker(checkpoint_path=str(ROOT/'checkpoints/vjepa2-ac-libero90-euler-fixed-rot180-step2000-inference.pt'),
             vjepa2_repo='/data/pxchen/JEPA_WAM',device='cuda',dtype=torch.float32,max_ac_steps=2)
    adapter = LiberoACAdapter(low_level_steps_per_ac_step=4)
    records = torch.load(OUT/'wam_windows.pt',map_location='cpu',weights_only=False)
    for r in records:
        with torch.no_grad():
            w, v = r['wam_future'], r['real_future']
            z = ranker.encode_clips([r['context'],[w[0],w[1]],[w[1],w[2]],[v[0],v[1]],[v[1],v[2]]]).detach()
        r['reps'] = z.cpu()
        r['state'] = adapter.state_from_observation(r['obs'])
    ranker.encoder.cpu()
    torch.cuda.empty_cache()
    model = ranker.predictor
    model.requires_grad_(True).eval()
    model.extrinsics_encoder.requires_grad_(False)
    model.use_activation_checkpointing = True
    params = [p for p in model.parameters() if p.requires_grad]
    tracked = model.action_encoder.weight
    before = tracked.detach().clone()
    optimizer = torch.optim.AdamW(params,lr=1e-6,weight_decay=0.04)
    metrics = []

    def forward(r):
        a = r['clean'].to('cuda').detach().requires_grad_(True)
        scale, offset = r['scale'].to('cuda'), r['offset'].to('cuda')
        raw = (a[:,:8]-offset)/scale
        ac = adapter.convert_torch(raw, torch.as_tensor(r['state'],device='cuda'))
        reps = r['reps'].to('cuda')
        energy = ranker.differentiable_energy(current_rep=reps[0],target_reps=reps[1:3],ac_actions=ac.actions,ac_states=ac.states).mean()
        grad = torch.autograd.grad(energy,a,create_graph=True)[0]
        delta = -4.0*grad[:,:8]
        rms = delta.square().mean().sqrt().clamp_min(1e-12)
        delta = delta*torch.clamp(0.02/rms,max=1.0)
        guided = r['base'].to('cuda')[:,:8]+delta
        loss = F.mse_loss(guided[...,:6],r['expert'].to('cuda')[None,...,:6])
        return loss, energy, guided, delta

    # Prove the action path alone, initially with default SDPA backend.
    try:
        loss, _, _, _ = forward(records[0])
        probe = torch.autograd.grad(loss, tracked)[0]
        log('DEFAULT_BACKEND_DOUBLE_BACKWARD', passed=True,grad_norm=float(probe.norm()))
    except RuntimeError as e:
        log('DEFAULT_BACKEND_DOUBLE_BACKWARD',passed=False,error=str(e))
    gc.collect(); torch.cuda.empty_cache()
    # Math SDPA is an equivalent attention implementation with double-backward support.
    with torch.backends.cuda.sdp_kernel(enable_flash=False,enable_math=True,enable_mem_efficient=False):
        fixed_before = forward(records[0])[2].detach().cpu()
        for step,r in enumerate(records,1):
            optimizer.zero_grad(set_to_none=True)
            loss, energy, guided, delta = forward(r)
            loss.backward()  # ONLY action supervision: no prediction-loss contribution.
            grad = tracked.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.norm()>0
            for p in params:
                if p.grad is not None: assert torch.isfinite(p.grad).all()
            row=dict(step=step,action_loss=float(loss.detach()),energy=float(energy.detach()),
                     action_encoder_grad_norm=float(grad.norm()),correction_rms=float(delta.detach().square().mean().sqrt()),
                     peak_gib=torch.cuda.max_memory_allocated()/2**30)
            torch.nn.utils.clip_grad_norm_(params,1.0)
            optimizer.step(); metrics.append(row); log('ACTION_ONLY_UPDATE',**row)
            del loss, energy, guided, delta
        optimizer.zero_grad(set_to_none=True)
        fixed_after = forward(records[0])[2].detach().cpu()
        param_change=float((tracked.detach()-before).norm())
        action_change=float((fixed_after-fixed_before).square().mean().sqrt())
        assert param_change>0 and action_change>0
        # Separately check the original real-future prediction-loss path.
        r=records[0]; reps=r['reps'].to('cuda')
        raw=(r['expert']-r['offset'])/r['scale']
        ac=adapter.convert_torch(raw[None].to('cuda'),torch.as_tensor(r['state'],device='cuda'))
        tf,ar=loss_fn(model,torch.stack([reps[0],reps[3],reps[4]])[None],ac.actions,ac.states)
        pred_grad=torch.autograd.grad(tf+ar,tracked)[0]
        assert torch.isfinite(pred_grad).all()
        report=dict(passed=True,updates=len(metrics),metrics=metrics,parameter_change_norm=param_change,
                    fixed_sample_action_change_rms=action_change,prediction_grad_norm=float(pred_grad.norm()),
                    note='mechanical smoke only; no full policy training, no online success claims; FP32 correction before BF16 output cast')
        (OUT/'backward_report.json').write_text(json.dumps(report,indent=2))
        log('SMOKE_BACKWARD_PASSED',**{k:v for k,v in report.items() if k!='metrics'})

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    p=argparse.ArgumentParser(); p.add_argument('--phase',choices=['cache','backward'],required=True)
    args=p.parse_args()
    try:
        (cache_wam if args.phase=='cache' else backward_smoke)()
    except Exception:
        traceback.print_exc(); sys.exit(1)
