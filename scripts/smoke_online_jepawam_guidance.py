"""Paired native/guidance one-task smoke. Never starts the 144 evaluation."""
import os, sys, json, time, fcntl, traceback
from pathlib import Path
from collections import deque
import numpy as np
import torch
from PIL import Image
import imageio.v2 as imageio
sys.path[:0] = ['/data/pxchen/JEPA_WAM', '/data/pxchen/PACT-WAM/scripts',
                '/data/pxchen/PACT-WAM/artifacts/predictor2.1ac1009']
from experiments.robot.libero import run_libero_eval as ev
from prismatic.models import load_vla
from smoke_joint_predictor import JointPredictor, DATA
from vectorized_ac_attention import enable_vectorized
from jepawam_gradient_guidance import JEPAGuidance

OUT = Path('/data/pxchen/PACT-WAM/artifacts/jepawam_gradient_guidance2_cosine_smoke_20261013')
ASSETS = Path('/data/pxchen/PACT-WAM/checkpoints/jepa_wam_assets_20261009')
PCKPT = Path('/data/pxchen/PACT-WAM/artifacts/predictor21_joint20_train10epoch_bs16_gpu3_20261011/step19690.pt')

def save(name, value):
    temp = OUT/(name+'.tmp'); temp.write_text(json.dumps(value, indent=2)); temp.replace(OUT/name)

def emit(event, **data):
    print(json.dumps(dict(event=event, **data)), flush=True)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = open(OUT/'run.lock', 'a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'config.json').exists(): raise RuntimeError('Refusing duplicate smoke')
    torch.set_num_threads(4)
    policy = next((ASSETS/'JEPA_WAM/checkpoints/libero').rglob('step-040000-epoch-37-loss=0.0262.pt'))
    cfg = ev.GenerateConfig(pretrained_checkpoint=str(policy),
        base_vlm=str(ASSETS/'JEPA_WAM/checkpoints/pretrained_vlm/prism-qwen25-vjepa21-vitl-384px+0_5b+stage-finetune+x7'),
        llm_checkpoint_path=str(ASSETS/'Qwen2.5-0.5B'),
        vjepa_checkpoint_path=str(ASSETS/'vjepa2/vjepa2_1_vitl_dist_vitG_384.pt'),
        task_suite_name='libero_10', max_episode_steps=700, num_steps_wait=5, num_open_loop_steps=20, seed=7)
    config = dict(policy=str(policy), predictor=str(PCKPT), physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        task_id=365, seed=7, initial_state_index=0, max_steps=700, execute_steps=20,
        guidance_scale=4., max_delta_rms=.02, normalize_gradient=False,
        flow_steps_zero_based=[2,3], flow_steps_one_based=[3,4], verify_descent=True,
        loss='mean tokenwise 1-cosine(P(current, executable20, state8), fixed W joint latent)',
        correction_coordinates='WAM normalized first6, all20 commands; gripper gradient zero',
        predictor_precision='FP32 forward/backward; frozen parameters',
        policy_precision='native BF16; guided flow accumulator FP32',
        compile=False, trigger=False, arb=False, real_future_input=False,
        note='Two corrections: between flow3/4 and after flow4. Old 2x4 AC interface is replaced by current20 joint Predictor. No 144 launch.')
    save('config.json', config)
    begin=time.perf_counter()
    model=load_vla(policy,load_for_training=False,base_vlm=cfg.base_vlm,llm_checkpoint_path=cfg.llm_checkpoint_path,
        vjepa_checkpoint_path=cfg.vjepa_checkpoint_path,load_visual_token_cosine_head=True).cuda().eval().requires_grad_(False)
    weights=torch.load(policy,map_location='cpu',weights_only=False,mmap=True)['model']['visual_token_cosine_head']
    model.visual_token_cosine_head.load_state_dict(weights,strict=True);del weights
    predictor=JointPredictor().eval().requires_grad_(False)
    ck=torch.load(PCKPT,map_location='cpu',weights_only=False,mmap=True)
    predictor.load_state_dict(ck['model'],strict=True);del ck
    predictor.cuda();enable_vectorized(predictor)
    key=model._check_unnorm_key(model.norm_stats,'libero_10_no_noops')
    adapter=JEPAGuidance(model,predictor,json.load(open(DATA/'train_only_stats.json')),key).install()
    torch.cuda.synchronize();emit('MODELS_LOADED',seconds=time.perf_counter()-begin)
    suite=ev.benchmark.get_benchmark_dict()['libero_10']()
    task=suite.get_task(365)
    env,instruction=ev.get_libero_env(task,resolution=384)
    env.reset();raw=env.set_init_state(suite.get_task_init_states(365)[0])
    for _ in range(5):raw,_,_,_=env.step(ev.get_libero_dummy_action())
    obs=ev._prepare_observation(raw);env.close()
    images=[Image.fromarray(obs['full_image']),Image.fromarray(obs['wrist_image'])]
    def invoke():
        return model.predict_action(image=images,instruction=instruction,unnorm_key=key,proprio=obs['state'])
    # Exactly the same noise; adapter-off versus adapter-zero must be identical.
    ev.set_seed_everywhere(7); adapter.context=None; baseline=invoke()
    adapter.prepare(images[0],obs['state']);adapter.scale=0
    ev.set_seed_everywhere(7); zero=invoke()
    error=float(np.abs(baseline-zero).max());assert error == 0, error
    adapter.scale=4
    adapter.check_gradient=True
    ev.set_seed_everywhere(7); guided=invoke()
    diagnostics=list(adapter.records)
    assert len(diagnostics)==2 and all(r['gradient_rms']>0 for r in diagnostics)
    assert np.isfinite(guided).all()
    assert all(p.grad is None for p in predictor.parameters())
    save('smoke.json',dict(status='passed',zero_scale_max_abs_error=error,gradient_nodes=diagnostics,
        raw_first6_correction_rms=float(np.sqrt(np.mean((guided[:,:6]-baseline[:,:6])**2))),
        frozen_predictor_no_parameter_grads=True))
    emit('GRADIENT_SMOKE_PASSED',zero_scale_max_abs_error=error,nodes=diagnostics)
    results=[]
    for variant in ['native','guidance']:
        ev.set_seed_everywhere(7)
        env,instruction=ev.get_libero_env(task,resolution=384)
        frames=[];queue=deque();timings=[];nodes=[];success=False;started=time.perf_counter()
        try:
            env.reset();raw=env.set_init_state(suite.get_task_init_states(365)[0])
            for step in range(705):
                if step<5: raw,_,done,_=env.step(ev.get_libero_dummy_action())
                else:
                    frames.append(ev.get_libero_image(raw))
                    if not queue:
                        obs=ev._prepare_observation(raw)
                        images=[Image.fromarray(obs['full_image']),Image.fromarray(obs['wrist_image'])]
                        tick=time.perf_counter()
                        if variant=='guidance':adapter.prepare(images[0],obs['state'])
                        else:adapter.context=None
                        a=model.predict_action(image=images,instruction=instruction,unnorm_key=key,proprio=obs['state'])
                        a=ev._process_action(a.astype(np.float32))
                        assert a.shape==(20,7) and np.isfinite(a).all()
                        torch.cuda.synchronize();timings.append(time.perf_counter()-tick)
                        if variant=='guidance':
                            nodes.extend([dict(replan=len(timings),**r) for r in adapter.records])
                        queue.extend(a)
                        emit('REPLAN',variant=variant,replan=len(timings),policy_s=timings[-1],
                             guidance_nodes=adapter.records if variant=='guidance' else [])
                    raw,_,done,_=env.step(np.asarray(queue.popleft()).tolist())
                if done:success=True;break
        finally:env.close()
        video=OUT/f'{variant}_task365_seed7_success{int(success)}.mp4'
        imageio.mimwrite(video,frames,fps=30)
        record=dict(variant=variant,success=success,control_steps=len(frames),replans=len(timings),
            total_s=time.perf_counter()-started,policy_mean_s=float(np.mean(timings)),
            guidance_nodes=nodes,video=str(video))
        results.append(record);save('results.json',results);emit('EPISODE_DONE',**{k:v for k,v in record.items() if k!='guidance_nodes'})
    adapter.close()
    save('summary.json',dict(status='complete',config=config,results=results,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,elapsed_s=time.perf_counter()-begin))
    emit('COMPLETE')

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True);save('failure.json',dict(traceback=traceback.format_exc()));raise
