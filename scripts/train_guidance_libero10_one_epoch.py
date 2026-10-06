"""One-epoch predictor-only action-supervised training; isolated from production."""
import argparse
import functools
import gc
import json
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT=Path('/data/pxchen/PACT-WAM')
DATA=ROOT/'artifacts/libero10_guidance_backward_smoke_20260929/data/libero_10'
EPOCH=int(os.environ.get('PACT_TRAIN_EPOCH','1'))
assert EPOCH in (1,2)
DEFAULT_OUT=ROOT/('artifacts/libero10_predictor_action_oneepoch_20260930' if EPOCH==1
                  else 'artifacts/libero10_predictor_action_epoch2_20260930')
OUT=Path(os.environ.get('PACT_OUTPUT_DIR',DEFAULT_OUT))
CACHE_ROOT=Path(os.environ.get('PACT_CACHE_ROOT',OUT))
LEARNING_RATE=float(os.environ.get('PACT_LEARNING_RATE','1e-6'))
TRAIN_STEPS=int(os.environ.get('PACT_TRAIN_STEPS','450'))
PAIRED_EVAL=os.environ.get('PACT_PAIRED_EVAL','0')=='1'
sys.path.insert(0,str(ROOT))

def emit(event,**kw):
    print(json.dumps(dict(time=time.strftime('%F %T'),event=event,**kw)),flush=True)

def tasks():
    rows=[]
    for path in sorted(DATA.glob('*.hdf5')):
        m=re.match(r'^[A-Z_]+SCENE\d+_(.+)_demo$',path.stem)
        assert m,path
        language=m.group(1).replace('_',' ')
        with h5py.File(path,'r') as f:
            demos=sorted(f['data'],key=lambda x:int(x.split('_')[-1]))
            assert len(demos)==50,(path,len(demos))
            for demo in demos:
                idx=int(demo.split('_')[-1]); n=len(f['data'][demo]['actions'])
                assert n>=17
                # Fixed single draw per demonstration, reproducible across restart.
                rng=np.random.default_rng(239+1000000*(EPOCH-1)+100003*len(rows))
                t=int(rng.integers(8,n-8))
                rows.append(dict(task_file=path.name,demo=demo,t=t,n=n,
                    language=language,split='val' if idx%10==0 else 'train'))
    assert len(rows)==500
    return rows

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=tasks(); manifest=OUT/'windows.json'
    if manifest.exists(): assert json.loads(manifest.read_text())==rows
    else: manifest.write_text(json.dumps(rows,indent=2))
    emit('PREFLIGHT_PASSED',total=len(rows),train=sum(r['split']=='train' for r in rows),
         val=sum(r['split']=='val' for r in rows),tasks=10)

def hdf_sample(row):
    from experiments.libero import eval_libero_single as ev
    path=DATA/row['task_file']; t=row['t']
    with h5py.File(path,'r') as f:
        g=f['data'][row['demo']]; o=g['obs']
        ee=np.asarray(o['ee_states'][t])
        obs=dict(agentview_image=np.asarray(o['agentview_rgb'][t]),
                 robot0_eye_in_hand_image=np.asarray(o['eye_in_hand_rgb'][t]),
                 robot0_eef_pos=ee[:3],robot0_eef_quat=Rotation.from_rotvec(ee[3:6]).as_quat(),
                 robot0_gripper_qpos=np.asarray(o['gripper_states'][t]))
        expert=torch.from_numpy(np.asarray(g['actions'][t:t+8],dtype=np.float32))
        prev=np.rot90(np.asarray(o['agentview_rgb'][t-8]),2).copy()
    return obs,prev,expert

def config():
    cfg=OmegaConf.load(ROOT/'evaluate_results/guidance789_rawgrad4x_long36_full144_euler_fixed_step2000_20260929/seed_7/manager_config.yaml')
    cfg.EVALUATION.output_dir=str(OUT/'load_placeholder')
    cfg.EVALUATION.vjepa2_ac.enabled=False
    cfg.EVALUATION.vjepa2_ac.guidance.enabled=False
    cfg.EVALUATION.vjepa2_ac.num_action_candidates=1
    cfg.EVALUATION.trigger_probe.enabled=False
    cfg.EVALUATION.action_denoise_trace=False
    cfg.MULTIRUN.task_file=None
    cfg.seed=7
    return cfg

def make_wam(need_prompts=False):
    from experiments.libero import eval_libero_single as ev
    cfg=config()
    cache=None
    if need_prompts:
        rows=tasks()
        languages=sorted({r['language'] for r in rows})
        ev._evaluation_prompts=lambda cfg:[ev.DEFAULT_PROMPT.format(task=x) for x in languages]
        cache=ev._preencode_prompt_contexts(cfg,device='cuda',dtype=torch.bfloat16)
    model_cfg=OmegaConf.create(OmegaConf.to_container(cfg.model,resolve=True))
    model_cfg.load_text_encoder=False
    model=instantiate(model_cfg,model_dtype=torch.bfloat16,device='cuda')
    ev._load_model_checkpoint(model,str(cfg.ckpt))
    model=model.to('cuda').eval().requires_grad_(False)
    if cache is not None:model._libero_prompt_context_cache=cache
    stats=ev.load_dataset_stats_from_json(str(ev._resolve_dataset_stats_path(cfg)))
    processor=instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    return model,processor,cfg

def cache_wam(limit=500):
    from experiments.libero import eval_libero_single as ev
    rows=tasks(); model,processor,cfg=make_wam(need_prompts=True)
    original=model.infer_action_from_video; capture={}
    @functools.wraps(original)
    def wrapped(**kw):
        capture['kw']={k:(v.detach().cpu() if torch.is_tensor(v) else v) for k,v in kw.items()}
        return original(**kw)
    model.infer_action_from_video=wrapped
    normalizer=processor.normalizer.normalizers['action'][processor.shape_meta['action'][0]['key']]
    scale=normalizer.scale.detach().float().cpu().reshape(-1)
    offset=normalizer.offset.detach().float().cpu().reshape(-1)
    (OUT/'wam').mkdir(exist_ok=True)
    for i,row in enumerate(rows):
        if i>=limit:break
        dest=OUT/'wam'/f'{i:04d}.pt'
        if dest.exists():continue
        obs,prev,expert=hdf_sample(row)
        _,_,future,_=ev._predict_action_chunk(obs,prev,row['language'],model,processor,cfg,
                  action_horizon=int(cfg.data.train.num_frames)-1,
                  input_w=int(cfg.data.train.video_size[1]),input_h=int(cfg.data.train.video_size[0]),
                  model_device='cuda',executed_horizon=8)
        assert len(future)>=3
        item=dict(obs=obs,context=[prev,np.rot90(obs['agentview_image'],2).copy()],
                  future=[np.asarray(x) for x in future[:3]],
                  expert=expert*scale+offset,scale=scale,offset=offset,
                  kw=capture['kw'],row=row)
        tmp=dest.with_suffix('.tmp');torch.save(item,tmp);os.replace(tmp,dest)
        if (i+1)%10==0 or i==0:emit('WAM_CACHE_PROGRESS',completed=i+1,total=500)
    emit('WAM_CACHE_COMPLETED',total=limit)

def cache_encoder(limit=500):
    from experiments.libero.vjepa2_ac_ranker import VJEPA2ACRanker
    ranker=VJEPA2ACRanker(checkpoint_path=str(ROOT/'checkpoints/vjepa2-ac-libero90-euler-fixed-rot180-step2000-inference.pt'),
           vjepa2_repo='/data/pxchen/JEPA_WAM',device='cuda',dtype=torch.float32,max_ac_steps=2)
    del ranker.predictor;gc.collect();torch.cuda.empty_cache()
    (OUT/'latent').mkdir(exist_ok=True)
    for i in range(limit):
        dest=OUT/'latent'/f'{i:04d}.pt'
        if dest.exists():continue
        r=torch.load(OUT/'wam'/f'{i:04d}.pt',map_location='cpu',weights_only=False)
        w=r['future']
        with torch.no_grad():
            z=ranker.encode_clips([r['context'],[w[0],w[1]],[w[1],w[2]]]).cpu()
        tmp=dest.with_suffix('.tmp');torch.save(z,tmp);os.replace(tmp,dest)
        if (i+1)%20==0 or i==0:emit('ENCODER_CACHE_PROGRESS',completed=i+1,total=500)
    emit('ENCODER_CACHE_COMPLETED',total=limit)

def trainer(smoke=False):
    from experiments.libero.vjepa2_ac_ranker import VJEPA2ACRanker,LiberoACAdapter
    from scripts.smoke_guidance_three_step_joint import make_train_method
    rows=tasks(); model,_,_=make_wam(need_prompts=False)
    train_method=make_train_method(model)
    ranker=VJEPA2ACRanker(checkpoint_path=str(ROOT/'checkpoints/vjepa2-ac-libero90-euler-fixed-rot180-step2000-inference.pt'),
           vjepa2_repo='/data/pxchen/JEPA_WAM',device='cpu',dtype=torch.float32,max_ac_steps=2)
    del ranker.encoder;ranker.device=torch.device('cuda')
    predictor=ranker.predictor.cuda().eval().requires_grad_(True)
    predictor.extrinsics_encoder.requires_grad_(False)
    predictor.use_activation_checkpointing=True
    params=[p for p in predictor.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=LEARNING_RATE,weight_decay=0.04)
    adapter=LiberoACAdapter(low_level_steps_per_ac_step=4)
    trainids=[i for i,r in enumerate(rows) if r['split']=='train']
    valids=[i for i,r in enumerate(rows) if r['split']=='val']
    assert len(trainids)==450 and len(valids)==50
    order=(trainids[:2] if smoke else np.random.default_rng(239+1000000*(EPOCH-1)).permutation(trainids).tolist())
    runroot=OUT/('smoke' if smoke else 'train_one_epoch')
    runroot.mkdir(exist_ok=True)
    logpath=runroot/'metrics.jsonl'
    if not smoke and logpath.exists():
        # Resume is not yet supported; never silently repeat prior optimizer steps.
        raise RuntimeError('Existing training metrics; explicit resume required')
    if smoke and logpath.exists():
        raise RuntimeError('Existing smoke metrics; use a distinct output directory')
    if EPOCH==2 and not smoke:
        source=ROOT/'artifacts/libero10_predictor_action_oneepoch_20260930/train_one_epoch/step_0450.pt'
        checkpoint=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
        assert checkpoint['step']==450 and checkpoint['data_format']=='libero_ac_euler_v1'
        predictor.load_state_dict(checkpoint['predictor'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        emit('RESUMED_FROM_EPOCH1',checkpoint=str(source),optimizer_state_entries=len(optimizer.state))

    def one(i,update,step):
        r=torch.load(CACHE_ROOT/'wam'/f'{i:04d}.pt',map_location='cpu',weights_only=False)
        reps=torch.load(CACHE_ROOT/'latent'/f'{i:04d}.pt',map_location='cpu',weights_only=True).cuda()
        kw=r['kw'].copy()
        kw['precomputed_video_latents']=kw.pop('video_latents')
        assert kw.pop('action_infer_mode','idm')=='idm'
        kw.pop('compile_action_infer',None)
        kw={k:(v.to('cuda') if torch.is_tensor(v) else v) for k,v in kw.items()}
        state=torch.as_tensor(adapter.state_from_observation(r['obs']),device='cuda')
        scale,offset=r['scale'].cuda(),r['offset'].cuda()
        def energy(a):
            ac=adapter.convert_torch((a[:,:8]-offset)/scale,state)
            return ranker.differentiable_energy(current_rep=reps[0],target_reps=reps[1:],
                ac_actions=ac.actions,ac_states=ac.states).mean()
        kw.update(action_guidance_fn=energy,action_guidance_after_steps=[7,8,9],
            action_guidance_step_size=4.0,action_guidance_horizon=8,
            action_guidance_normalize_gradient=False,action_guidance_max_delta_rms=0.02,
            action_guidance_verify_descent=False,action_flow_trace=False)
        optimizer.zero_grad(set_to_none=True)
        start=time.monotonic()
        with torch.enable_grad(),torch.backends.cuda.sdp_kernel(enable_flash=False,enable_math=True,enable_mem_efficient=False),torch.autograd.graph.save_on_cpu(pin_memory=True):
            result=train_method(**kw)
            action=result['action']
            if action.ndim==2:action=action[None]
            loss=F.mse_loss(action[:,:8,:6],r['expert'].cuda()[None,:,:6])
            nodes=result['training_nodes'];assert len(nodes)==3
            if update:
                if step<=3 or step%50==0:
                    nodegrads=torch.autograd.grad(loss,nodes,retain_graph=True)
                    assert all(torch.isfinite(g).all() and g.norm()>0 for g in nodegrads)
                loss.backward()
        value=float(loss.detach())
        if update:
            grads=[p.grad for p in params if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert predictor.action_encoder.weight.grad is not None
            torch.nn.utils.clip_grad_norm_(params,1.0)
            optimizer.step()
        elapsed=time.monotonic()-start
        del result,action,loss,nodes,kw,reps
        gc.collect();torch.cuda.empty_cache()
        return value,elapsed

    def save(step):
        path=runroot/f'step_{step:04d}.pt'
        payload=dict(predictor=predictor.state_dict(),optimizer=optimizer.state_dict(),
                     step=step,epoch=EPOCH,model_source='EEF-fixed LIBERO90 step2000',
                     data_format='libero_ac_euler_v1',scope='LIBERO10 predictor-only action supervised',
                     torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
        tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path)
        emit('CHECKPOINT_SAVED',step=step,path=str(path))

    def evaluate(label, checkpoint_step):
        values=[]
        for j,i in enumerate(valids,1):
            value,elapsed=one(i,False,j);values.append(value)
            row=dict(split='val',eval_at=label,epoch=EPOCH,step=checkpoint_step,
                     val_index=j,window=i,loss=value,seconds=elapsed)
            with logpath.open('a') as f:f.write(json.dumps(row)+'\n')
            if j%10==0:emit('VAL_PROGRESS',eval_at=label,completed=j,mean_loss=float(np.mean(values)))
        emit('PAIRED_VAL_COMPLETED',eval_at=label,windows=len(values),
             mean_loss=float(np.mean(values)),median_loss=float(np.median(values)))
        return values

    target=2 if smoke else TRAIN_STEPS
    baseline_values=evaluate('step_0000',0) if PAIRED_EVAL and not smoke else None
    for local_step,i in enumerate(order[:target],1):
        step=(EPOCH-1)*450+local_step if not smoke else local_step
        value,elapsed=one(i,True,step)
        row=dict(split='train',epoch=EPOCH,step=step,window=i,loss=value,seconds=elapsed)
        with logpath.open('a') as f:f.write(json.dumps(row)+'\n')
        if step<=3 or step%10==0:emit('TRAIN_PROGRESS',**row)
        if not smoke and (local_step%100==0 or local_step==target):save(step)
    if not smoke:
        values=evaluate(f'step_{target:04d}',target)
        summary=dict(train_steps=target,learning_rate=LEARNING_RATE,val_windows=50,
                     val_mean_loss=float(np.mean(values)),val_median_loss=float(np.median(values)),
                     completed_at=time.strftime('%F %T'))
        if baseline_values is not None:
            summary.update(baseline_val_mean_loss=float(np.mean(baseline_values)),
                           baseline_val_median_loss=float(np.median(baseline_values)),
                           paired_mean_change=float(np.mean(values)-np.mean(baseline_values)))
        (runroot/'summary.json').write_text(json.dumps(summary,indent=2))
    emit('ONE_EPOCH_COMPLETED' if not smoke else 'SMOKE_COMPLETED',steps=target)

if __name__=='__main__':
    logging.basicConfig(level=logging.WARNING)
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','cache-wam','cache-encoder','smoke','train'],required=True)
    p.add_argument('--limit',type=int,default=500)
    args=p.parse_args()
    try: dict(prepare=prepare,**{'cache-wam':lambda:cache_wam(args.limit),'cache-encoder':lambda:cache_encoder(args.limit)},
              smoke=lambda:trainer(True),train=lambda:trainer(False))[args.phase]()
    except Exception:traceback.print_exc();sys.exit(1)
