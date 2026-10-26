"""Resumable144 evaluation using the tested two-node cosine guidance adapter."""
import argparse, hashlib
import smoke_online_jepawam_guidance as s

OUT = s.Path('/data/pxchen/PACT-WAM/artifacts/jepawam_guidance2_cosine_full144_nativeprecision_20261014')
MANIFEST = s.Path('/data/pxchen/PACT-WAM/artifacts/jepawam_native_full144_20261009/task_manifest144.json')

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true')
    parser.add_argument('--diagnostic-pairs',action='store_true');args=parser.parse_args()
    if args.diagnostic_pairs:
        OUT=s.Path('/data/pxchen/PACT-WAM/artifacts/jepawam_guidance_nativeprecision_pairs4_20261014')
    s.OUT=OUT;OUT.mkdir(parents=True,exist_ok=True)
    lock=open(OUT/'run.lock','a');s.fcntl.flock(lock,s.fcntl.LOCK_EX|s.fcntl.LOCK_NB)
    smoke=s.json.load(open('/data/pxchen/PACT-WAM/artifacts/jepawam_gradient_guidance2_cosine_smoke_20261013/smoke.json'))
    assert smoke['status']=='passed' and smoke['zero_scale_max_abs_error']==0
    manifest=s.json.load(open(MANIFEST))
    assert len(manifest)==144 and len({(r['task_id'],r['seed']) for r in manifest})==144
    if args.diagnostic_pairs:
        selected={(410,7),(457,7),(1945,7),(411,17)}
        manifest=[r for r in manifest if (r['task_id'],r['seed']) in selected]
        assert len(manifest)==4
    variants=['native','guidance'] if args.diagnostic_pairs else ['guidance']
    runs=[(variant,entry) for entry in manifest for variant in variants]
    previous=s.json.load(open('/data/pxchen/PACT-WAM/artifacts/jepawam_gradient_guidance2_cosine_smoke_20261013/config.json'))
    config={k:v for k,v in previous.items() if k not in ['task_id','seed','note','physical_gpu']}
    config.update(manifest=manifest,manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(s.Path('/data/pxchen/PACT-WAM/scripts/jepawam_gradient_guidance.py').read_bytes()).hexdigest(),
        physical_gpu=s.os.environ['CUDA_VISIBLE_DEVICES'],
        variants=variants,execute_dtype='retain native NumPy output precision; no float32 cast',
        note='Two-node cosine guidance; native-precision executor. Diagnostic mode runs4 paired episodes, selected from historical harm/rescue; not an unbiased accuracy sample.')
    if (OUT/'config.json').exists():
        assert args.resume,'Existing output: use --resume'
        old=s.json.load(open(OUT/'config.json'))
        assert {k:v for k,v in old.items() if k!='physical_gpu'}=={k:v for k,v in config.items() if k!='physical_gpu'}
    else:s.save('config.json',config)
    if (OUT/'failure.json').exists():
        (OUT/'failure.json').rename(OUT/f'failure_previous_{int(s.time.time())}.json')
    results=s.json.load(open(OUT/'results.json')) if (OUT/'results.json').exists() else []
    done={(r['variant'],r['task']['task_id'],r['task']['seed']) for r in results}
    assert len(done)==len(results)
    if len(done)==len(runs):s.emit('ALREADY_COMPLETE');return
    s.torch.set_num_threads(4)
    begin=s.time.perf_counter()
    policy=s.Path(config['policy'])
    cfg=s.ev.GenerateConfig(pretrained_checkpoint=str(policy),
        base_vlm=str(s.ASSETS/'JEPA_WAM/checkpoints/pretrained_vlm/prism-qwen25-vjepa21-vitl-384px+0_5b+stage-finetune+x7'),
        llm_checkpoint_path=str(s.ASSETS/'Qwen2.5-0.5B'),
        vjepa_checkpoint_path=str(s.ASSETS/'vjepa2/vjepa2_1_vitl_dist_vitG_384.pt'),
        task_suite_name='libero_10',max_episode_steps=700,num_steps_wait=5,num_open_loop_steps=20,seed=7)
    model=s.load_vla(policy,load_for_training=False,base_vlm=cfg.base_vlm,llm_checkpoint_path=cfg.llm_checkpoint_path,
        vjepa_checkpoint_path=cfg.vjepa_checkpoint_path,load_visual_token_cosine_head=True).cuda().eval().requires_grad_(False)
    weights=s.torch.load(policy,map_location='cpu',weights_only=False,mmap=True)['model']['visual_token_cosine_head']
    model.visual_token_cosine_head.load_state_dict(weights,strict=True);del weights
    predictor=s.JointPredictor().eval().requires_grad_(False)
    ck=s.torch.load(s.PCKPT,map_location='cpu',weights_only=False,mmap=True)
    predictor.load_state_dict(ck['model'],strict=True);del ck
    predictor.cuda();s.enable_vectorized(predictor)
    key=model._check_unnorm_key(model.norm_stats,'libero_10_no_noops')
    adapter=s.JEPAGuidance(model,predictor,s.json.load(open(s.DATA/'train_only_stats.json')),key).install()
    s.torch.cuda.synchronize();s.emit('MODELS_LOADED',seconds=s.time.perf_counter()-begin)
    suite=s.ev.benchmark.get_benchmark_dict()['libero_10']()
    for variant,entry in runs:
        if (variant,entry['task_id'],entry['seed']) in done:continue
        s.ev.set_seed_everywhere(entry['seed']);tick=s.time.perf_counter()
        task=suite.get_task(entry['task_id']);env,instruction=s.ev.get_libero_env(task,resolution=384)
        frames=[];queue=s.deque();timings=[];nodes=[];success=False
        s.emit('EPISODE_START',variant=variant,task=entry,instruction=instruction)
        try:
            env.reset();raw=env.set_init_state(suite.get_task_init_states(entry['task_id'])[0])
            for step in range(705):
                if step<5:raw,_,finished,_=env.step(s.ev.get_libero_dummy_action())
                else:
                    frames.append(s.ev.get_libero_image(raw))
                    if not queue:
                        obs=s.ev._prepare_observation(raw)
                        images=[s.Image.fromarray(obs['full_image']),s.Image.fromarray(obs['wrist_image'])]
                        start=s.time.perf_counter()
                        if variant=='guidance':adapter.prepare(images[0],obs['state'])
                        else:adapter.context=None;adapter.records=[]
                        a=model.predict_action(image=images,instruction=instruction,unnorm_key=key,proprio=obs['state'])
                        assert a.dtype==s.np.float64, a.dtype
                        a=s.ev._process_action(a)
                        assert a.shape==(20,7) and s.np.isfinite(a).all()
                        assert len(adapter.records)==(2 if variant=='guidance' else 0)
                        s.torch.cuda.synchronize();timings.append(s.time.perf_counter()-start)
                        nodes.extend([dict(replan=len(timings),**r) for r in adapter.records])
                        queue.extend(a)
                        if len(timings)<=2 or len(timings)%10==0:
                            s.emit('REPLAN',variant=variant,task=entry,replan=len(timings),policy_s=timings[-1],nodes=adapter.records)
                    raw,_,finished,_=env.step(s.np.asarray(queue.popleft()).tolist())
                if finished:success=True;break
        finally:env.close()
        video=OUT/f'{variant}_task{entry["task_id"]}_seed{entry["seed"]}_success{int(success)}.mp4'
        s.imageio.mimwrite(video,frames,fps=30)
        record=dict(variant=variant,task=entry,instruction=instruction,success=success,
            control_steps=len(frames),replans=len(timings),total_s=s.time.perf_counter()-tick,
            policy_mean_s=float(s.np.mean(timings)),guidance_nodes=nodes,video=str(video))
        results.append(record);s.save('results.json',results);done.add((variant,entry['task_id'],entry['seed']))
        mean=float(s.np.mean([r['total_s'] for r in results]))
        progress=dict(status='running',completed=len(results),total=len(runs),successes=sum(r['success'] for r in results),
            mean_task_s=mean,estimated_remaining_s=(len(runs)-len(results))*mean)
        s.save('progress.json',progress);s.emit('EPISODE_DONE',**{k:v for k,v in record.items() if k!='guidance_nodes'})
        s.emit('PROGRESS',**progress)
    adapter.close()
    s.save('summary.json',dict(status='complete',config=config,results=results,
        peak_allocated_mib=s.torch.cuda.max_memory_allocated()/2**20,elapsed_s=s.time.perf_counter()-begin))
    s.save('progress.json',dict(status='complete',completed=len(runs),total=len(runs),successes=sum(r['success'] for r in results)))
    s.emit('COMPLETE')

if __name__=='__main__':
    try:main()
    except Exception:
        s.OUT=OUT;OUT.mkdir(parents=True,exist_ok=True);s.save('failure.json',dict(traceback=s.traceback.format_exc()));raise
