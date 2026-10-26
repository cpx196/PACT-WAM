"""One paired online episode per variant; executable gripper processed exactly once."""
import os,sys,json,time,fcntl,traceback,argparse
from pathlib import Path
from collections import deque
import numpy as np
import torch
from PIL import Image
import imageio.v2 as imageio
sys.path.insert(0,'/data/pxchen/JEPA_WAM')
sys.path.insert(0,'/data/pxchen/PACT-WAM/scripts')
sys.path.insert(0,'/data/pxchen/PACT-WAM/artifacts/predictor2.1ac1009')
from experiments.robot.libero import run_libero_eval as ev
from prismatic.models import load_vla
from smoke_joint_predictor import JointPredictor,DATA
from vectorized_ac_attention import enable_vectorized
import train_jepawam_arb_pilot100 as t

OUT=t.ROOT/'jepawam_arb_online_one_task_20261013'
ASSETS=Path('/data/pxchen/PACT-WAM/checkpoints/jepa_wam_assets_20261009')
def save(name,obj):
    p=OUT/name;temp=p.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(p)
def emit(event,**kw):print(json.dumps(dict(event=event,**kw)),flush=True)

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--full144',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--alpha-manifest',default=None)
    args=parser.parse_args()
    if args.full144:OUT=t.ROOT/'jepawam_arb_pw_full144_epoch15_20261013'
    if args.alpha_manifest:
        assert not args.full144
        OUT=t.ROOT/'jepawam_arb_alpha_harm6_20261013'
    OUT.mkdir(parents=True,exist_ok=True)
    lock=open(OUT/'run.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'config.json').exists():
        if os.environ.get('SMOKE_RETRY')=='1' and (OUT/'failure.json').exists() and not (OUT/'results.json').exists():
            stamp=str(int(time.time()))
            (OUT/'failure.json').rename(OUT/f'failure_{stamp}.json')
            (OUT/'config.json').rename(OUT/f'config_{stamp}.json')
        elif not ((args.full144 or args.alpha_manifest) and args.resume):raise RuntimeError('Refusing duplicate run')
    if args.resume and (OUT/'failure.json').exists():
        (OUT/'failure.json').rename(OUT/f'failure_resume_{int(time.time())}.json')
    torch.set_num_threads(4)
    manifest=json.load(open(t.ROOT/'jepawam_native_full144_20261009/task_manifest144.json'))
    assert len(manifest)==144 and len({(r['task_id'],r['seed']) for r in manifest})==144
    entry=manifest[0]
    assert entry==dict(suite='libero_10',task_id=365,seed=7)
    if args.alpha_manifest:
        selected=json.load(open(args.alpha_manifest))
        assert len(selected)==6 and all(r in manifest for r in selected)
        manifest=selected;entry=selected[0]
    policy=next((ASSETS/'JEPA_WAM/checkpoints/libero').rglob('step-040000-epoch-37-loss=0.0262.pt'))
    cfg=ev.GenerateConfig(pretrained_checkpoint=str(policy),
        base_vlm=str(ASSETS/'JEPA_WAM/checkpoints/pretrained_vlm/prism-qwen25-vjepa21-vitl-384px+0_5b+stage-finetune+x7'),
        llm_checkpoint_path=str(ASSETS/'Qwen2.5-0.5B'),
        vjepa_checkpoint_path=str(ASSETS/'vjepa2/vjepa2_1_vitl_dist_vitG_384.pt'),
        task_suite_name='libero_10',max_episode_steps=700,num_steps_wait=5,num_open_loop_steps=20,seed=7)
    pckpt=t.ROOT/'predictor21_joint20_train10epoch_bs16_gpu3_20261011/step19690.pt'
    paths={k:t.ROOT/f'jepawam_arb_{k}_15epoch_clip5_20261013/epoch15_step59055.pt' for k in ['pw','no_pw']}
    if args.full144 or args.alpha_manifest:paths={k:v for k,v in paths.items() if k=='pw'}
    variants=['alpha_0','alpha_0.25','alpha_0.5','alpha_1'] if args.alpha_manifest else ['pw'] if args.full144 else ['native','no_pw','pw']
    config=dict(task=entry,manifest=manifest if args.full144 or args.alpha_manifest else [entry],initial_state_index=0,
        variants=variants,max_steps=700,
        settling_steps=5,execute_steps=20,policy=str(policy),predictor=str(pckpt),arb={k:str(v) for k,v in paths.items()},
        unnorm_key='libero_10_no_noops',env_resolution=384,main_view_predictor=True,real_future_input=False,
        compile=False,gripper='native binarize/invert once; unchanged by ARB',
        correction='first6 executable command dimensions; no extra RMS cap or clipping',
        current_encoding='separate batch1 encoder call, matching predictor cache',physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']))
    if args.alpha_manifest:
        config.update(alpha_values=[0,.25,.5,1],correction='a_raw + alpha * learned first6 correction; unchanged gripper; no extra cap',
            selection='6 confirmed harm episodes: two per moka-pot, bowl/drawer and mug/plate task family; diagnostic subset only')
    if (OUT/'config.json').exists():
        old=json.load(open(OUT/'config.json'))
        assert {k:v for k,v in old.items() if k!='physical_gpu'}=={k:v for k,v in config.items() if k!='physical_gpu'}
    else:save('config.json',config)
    results=json.load(open(OUT/'results.json')) if args.resume and (OUT/'results.json').exists() else []
    done_keys={(r['variant'],r['task']['task_id'],r['task']['seed']) for r in results}
    expected=24 if args.alpha_manifest else 144 if args.full144 else 3
    if len(done_keys)==expected:emit('ALREADY_COMPLETE');return
    tick=time.perf_counter()
    model=load_vla(policy,load_for_training=False,base_vlm=cfg.base_vlm,llm_checkpoint_path=cfg.llm_checkpoint_path,
        vjepa_checkpoint_path=cfg.vjepa_checkpoint_path,load_visual_token_cosine_head=True).cuda().eval().requires_grad_(False)
    raw=torch.load(policy,map_location='cpu',weights_only=False,mmap=True)['model']['visual_token_cosine_head']
    model.visual_token_cosine_head.load_state_dict(raw,strict=True);del raw
    predictor=JointPredictor().eval().requires_grad_(False)
    ck=torch.load(pckpt,map_location='cpu',weights_only=False,mmap=True);predictor.load_state_dict(ck['model'],strict=True);del ck
    predictor.cuda();enable_vectorized(predictor)
    stats=json.load(open(DATA/'train_only_stats.json'))
    norm={k:(torch.tensor(v['mean'],device='cuda'),torch.tensor(v['std'],device='cuda').clamp_min(1e-6)) for k,v in stats.items()}
    arbs={}
    for k,path in paths.items():
        ck=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        assert ck['epoch']==15 and ck['config']['no_pw']==(k=='no_pw')
        m=t.ARB(stats).cuda().eval().requires_grad_(False);m.load_state_dict(ck['model'],strict=True);arbs[k]=m;del ck
    cfg.unnorm_key=model._check_unnorm_key(model.norm_stats,config['unnorm_key'])
    torch.cuda.synchronize();load_s=time.perf_counter()-tick;emit('MODELS_LOADED',seconds=load_s)
    capture={};active={'mode':None}
    def hook(module,args,output):
        if active['mode']=='pw':
            h=output['llm_hidden'];assert h.shape[1]>=1153
            with torch.autocast('cuda',dtype=torch.bfloat16):capture['W']=model.visual_token_cosine_head.align_dimension(h[:,1:577]).float()
    handle=model.register_forward_hook(hook)
    def action(obs,instruction,mode,alpha=1.):
        active['mode']=mode
        stages={};torch.cuda.synchronize();start=time.perf_counter()
        images=[Image.fromarray(obs['full_image']),Image.fromarray(obs['wrist_image'])]
        a=model.predict_action(image=images,instruction=instruction,unnorm_key=cfg.unnorm_key,proprio=obs['state']).astype(np.float32)
        assert a.shape==(20,7)
        a=ev._process_action(a);assert bool(np.isin(a[:,-1],[-1,0,1]).all())
        torch.cuda.synchronize();stages['wam_s']=time.perf_counter()-start
        if mode=='native':out=a
        else:
            b={'action':torch.from_numpy(a).cuda()[None], 'state':torch.as_tensor(obs['state'],dtype=torch.float32,device='cuda')[None]}
            if mode=='pw':
                tick=time.perf_counter()
                image=model.vision_backbone.image_transform(images[0])[None].cuda()
                with torch.autocast('cuda',dtype=torch.bfloat16):z=model.vision_backbone(image)
                assert z.shape==(1,576,1024)
                torch.cuda.synchronize();stages['current_encode_s']=time.perf_counter()-tick;tick=time.perf_counter()
                am,asd=norm['action'];sm,ssd=norm['state']
                with torch.autocast('cuda',dtype=torch.bfloat16):b['P']=predictor(z,(b['action']-am)/asd,(b['state']-sm)/ssd).float()
                b['W']=capture.pop('W');torch.cuda.synchronize();stages['predictor_s']=time.perf_counter()-tick
            else:
                b['P']=torch.zeros((1,576,1024),device='cuda');b['W']=torch.zeros_like(b['P'])
            assert b['P'].shape==b['W'].shape==(1,576,1024)
            tick=time.perf_counter();t.NO_PW=(mode=='no_pw')
            with torch.autocast('cuda',dtype=torch.bfloat16):corrected,delta=arbs[mode](b)
            torch.cuda.synchronize();stages['arb_s']=time.perf_counter()-tick
            c=corrected[0].float().cpu().numpy()
            out=a.copy();out[:,:6]=c if alpha==1 else a[:,:6]+alpha*(c-a[:,:6])
            assert np.array_equal(out[:,-1],a[:,-1]);stages['normalized_delta_rms']=float(delta.square().mean().sqrt())
            stages['applied_normalized_delta_rms']=alpha*stages['normalized_delta_rms']
        assert np.isfinite(out).all()
        stages['total_s']=time.perf_counter()-start
        return out,stages
    # Cache replay sanity: use only a current demonstration observation, never future frames.
    import h5py
    row=json.load(open(t.CACHE/'index.json'))[0]
    cached=torch.load(t.CACHE/row['cache_file'],map_location='cpu',weights_only=False,mmap=True)
    base=torch.load(DATA/row['file'],map_location='cpu',weights_only=False,mmap=True)
    with h5py.File(base['source'],'r') as f:
        g=f['data'][base['demo']]['obs'];ti=int(base['starts'][0])
        obs={'full_image':np.ascontiguousarray(g['agentview_rgb'][ti][::-1,::-1]),
             'wrist_image':np.ascontiguousarray(g['eye_in_hand_rgb'][ti][::-1,::-1]),'state':base['state'][0].numpy()}
        instruction=json.loads(f['data'].attrs['problem_info'])['language_instruction']
    ev.set_seed_everywhere(int(cached['window_seed_base']))
    active['mode']='pw'
    images=[Image.fromarray(obs['full_image']),Image.fromarray(obs['wrist_image'])]
    raw_a=model.predict_action(image=images,instruction=instruction,unnorm_key=cfg.unnorm_key,proprio=obs['state']).astype(np.float32)
    executable=ev._process_action(raw_a)
    assert np.allclose(executable,cached['wam_actions'][0].numpy(),atol=1e-5),'Cached WAM command mismatch'
    with torch.autocast('cuda',dtype=torch.bfloat16):z=model.vision_backbone(model.vision_backbone.image_transform(images[0])[None].cuda())
    assert torch.allclose(z.float().cpu(),base['current_latent'][0:1].float(),atol=.001,rtol=.001),'Current encoder mismatch'
    assert torch.allclose(capture.pop('W').cpu(),cached['W'][0:1],atol=.001,rtol=.001),'W alignment mismatch'
    save('smoke.json',dict(status='passed',cache_commands_match=True,current_encoder_match=True,W_match=True))
    emit('CACHE_CONTRACT_PASSED');del cached,base
    suite=ev.benchmark.get_benchmark_dict()['libero_10']()
    runs=[('pw',r,1.) for r in manifest] if args.full144 else [(m,entry,1.) for m in ['native','no_pw','pw']]
    if args.alpha_manifest:runs=[(f'alpha_{alpha:g}',r,alpha) for r in manifest for alpha in [0.,.25,.5,1.]]
    for mode,entry,alpha in runs:
        if (mode,entry['task_id'],entry['seed']) in done_keys:continue
        task=suite.get_task(entry['task_id'])
        ev.set_seed_everywhere(entry['seed']);begin=time.perf_counter()
        env,instruction=ev.get_libero_env(task,resolution=384);init=suite.get_task_init_states(entry['task_id'])[0]
        env_s=time.perf_counter()-begin;episode_start=time.perf_counter();frames=[];timings=[];queue=deque();success=False
        action_mode=('native' if alpha==0 else 'pw') if args.alpha_manifest else mode
        emit('EPISODE_START',variant=mode,task=entry,instruction=instruction,alpha=alpha)
        try:
            env.reset();rawobs=env.set_init_state(init)
            for step in range(705):
                if step<5:
                    rawobs,_,done,_=env.step(ev.get_libero_dummy_action())
                else:
                    frames.append(ev.get_libero_image(rawobs))
                    if not queue:
                        with torch.inference_mode():a,stage=action(ev._prepare_observation(rawobs),instruction,action_mode,alpha)
                        timings.append(stage);queue.extend(a)
                        if len(timings)<=2 or len(timings)%10==0:emit('REPLAN',variant=mode,replan=len(timings),**stage)
                    rawobs,_,done,_=env.step(np.asarray(queue.popleft()).tolist())
                if done:success=True;break
        finally:env.close()
        episode_s=time.perf_counter()-episode_start;video_start=time.perf_counter()
        video=OUT/f"{mode}_task{entry['task_id']}_seed{entry['seed']}_success{int(success)}.mp4"
        if frames:imageio.mimwrite(video,frames,fps=30)
        record=dict(variant=mode,alpha=alpha,task=entry,instruction=instruction,success=success,control_steps=len(frames),
            replans=len(timings),episode_s=episode_s,env_setup_s=env_s,video_s=time.perf_counter()-video_start,
            total_s=time.perf_counter()-begin,policy_mean_s=float(np.mean([r['total_s'] for r in timings])),
            steady_policy_mean_s=float(np.mean([r['total_s'] for r in timings[1:]])) if len(timings)>1 else None,
            stages=timings,video=str(video),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        results.append(record);save('results.json',results);emit('EPISODE_DONE',**{k:v for k,v in record.items() if k!='stages'})
        done_keys.add((mode,entry['task_id'],entry['seed']))
        mean=float(np.mean([r['total_s'] for r in results]))
        progress=dict(status='running',completed=len(results),total=len(runs),successes=sum(r['success'] for r in results),
            mean_task_s=mean,estimated_remaining_s=(len(runs)-len(results))*mean,model_load_s=load_s)
        save('progress.json',progress);emit('PROGRESS',**progress)
    handle.remove();save('summary.json',dict(status='complete',model_load_s=load_s,results=results));emit('COMPLETE')
    save('progress.json',dict(status='complete',completed=len(results),total=len(runs),successes=sum(r['success'] for r in results)))

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True);save('failure.json',dict(traceback=traceback.format_exc()));raise
