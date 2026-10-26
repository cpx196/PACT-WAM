"""One fixed-image instruction test and three native task410/seed7 replays."""
import smoke_online_jepawam_guidance as s

OUT=s.Path('/data/pxchen/PACT-WAM/artifacts/jepawam_W_instruction_native410_diagnostic_20261014')

def main():
    s.OUT=OUT;OUT.mkdir(parents=True,exist_ok=True)
    lock=open(OUT/'run.lock','a');s.fcntl.flock(lock,s.fcntl.LOCK_EX|s.fcntl.LOCK_NB)
    if (OUT/'config.json').exists():raise RuntimeError('Refusing duplicate diagnostic')
    s.torch.set_num_threads(4)
    policy=next((s.ASSETS/'JEPA_WAM/checkpoints/libero').rglob('step-040000-epoch-37-loss=0.0262.pt'))
    cfg=s.ev.GenerateConfig(pretrained_checkpoint=str(policy),
        base_vlm=str(s.ASSETS/'JEPA_WAM/checkpoints/pretrained_vlm/prism-qwen25-vjepa21-vitl-384px+0_5b+stage-finetune+x7'),
        llm_checkpoint_path=str(s.ASSETS/'Qwen2.5-0.5B'),
        vjepa_checkpoint_path=str(s.ASSETS/'vjepa2/vjepa2_1_vitl_dist_vitG_384.pt'),
        task_suite_name='libero_10',max_episode_steps=700,num_steps_wait=5,num_open_loop_steps=20,seed=7,
        unnorm_key='libero_10_no_noops')
    s.save('config.json',dict(task_id=410,seed=7,variants=['historical_loop','new_loop_float32','new_loop_float64'],
        physical_gpu=s.os.environ['CUDA_VISIBLE_DEVICES'],max_steps=700,execute_steps=20,
        model=str(policy),guidance=False,arb=False,compile=False))
    started=s.time.perf_counter()
    model=s.load_vla(policy,load_for_training=False,base_vlm=cfg.base_vlm,llm_checkpoint_path=cfg.llm_checkpoint_path,
        vjepa_checkpoint_path=cfg.vjepa_checkpoint_path,load_visual_token_cosine_head=True).cuda().eval().requires_grad_(False)
    weights=s.torch.load(policy,map_location='cpu',weights_only=False,mmap=True)['model']['visual_token_cosine_head']
    model.visual_token_cosine_head.load_state_dict(weights,strict=True);del weights
    suite=s.ev.benchmark.get_benchmark_dict()['libero_10']();task=suite.get_task(410)
    s.ev.set_seed_everywhere(7)
    env,instruction=s.ev.get_libero_env(task,resolution=384)
    try:
        env.reset();raw=env.set_init_state(suite.get_task_init_states(410)[0])
        for _ in range(5):raw,_,_,_=env.step(s.ev.get_libero_dummy_action())
        obs=s.ev._prepare_observation(raw)
    finally:env.close()
    captured={}
    def hook(module,args,out):
        with s.torch.no_grad(),s.torch.autocast('cuda',dtype=s.torch.bfloat16):
            captured['W']=model.visual_token_cosine_head.align_dimension(out['llm_hidden'][:,1:577]).float().clone()
            captured['action_memory']=out['llm_hidden'][:,-model.action_placeholder_tokens:].float().clone()
    handle=model.register_forward_hook(hook)
    texts=[instruction,instruction.replace('close it','open it'),
           'turn on the stove and put the moka pot on it',instruction]
    original_w=original_mem=original_a=None;checks=[]
    for text in texts:
        s.ev.set_seed_everywhere(7)
        a=model.predict_action(image=[s.Image.fromarray(obs['full_image']),s.Image.fromarray(obs['wrist_image'])],
            instruction=text,unnorm_key=cfg.unnorm_key,proprio=obs['state'])
        w,mem=captured['W'],captured['action_memory']
        if original_w is None:original_w=w;original_mem=mem;original_a=a
        def difference(x,reference):
            return dict(max_abs=float((x-reference).abs().max()),relative_l2=float((x-reference).norm()/reference.norm().clamp_min(1e-12)))
        pb=model.get_prompt_builder();pb.add_turn('human',f'What action should the robot take to {text.lower()}?');pb.add_turn('gpt','')
        rec=dict(instruction=text,text_tokens=int(model.llm_backbone.tokenizer(pb.get_prompt(),return_tensors='pt').input_ids.shape[1]),
            W_difference=difference(w,original_w),action_memory_difference=difference(mem,original_mem),
            raw_action_first6_rms=float(s.np.sqrt(s.np.mean((a[:,:6]-original_a[:,:6])**2))))
        checks.append(rec);s.emit('INSTRUCTION_CHECK',**rec)
    handle.remove();s.save('instruction_checks.json',checks)
    results=[]
    for variant in ['historical_loop','new_loop_float32','new_loop_float64']:
        s.ev.set_seed_everywhere(7);env,instruction=s.ev.get_libero_env(task,resolution=384)
        first_action=None;calls=[];tick=s.time.perf_counter()
        original_get_action=s.ev.get_action
        def trace(*args,**kwargs):
            actions=original_get_action(*args,**kwargs)
            calls.append(s.np.asarray(actions).copy());return actions
        try:
            if variant=='historical_loop':
                s.ev.get_action=trace
                success,frames=s.ev._run_episode(cfg,env,instruction,model,suite.get_task_init_states(410)[0])
                s.ev.get_action=original_get_action
            else:
                frames=[];queue=s.deque();success=False
                env.reset();raw=env.set_init_state(suite.get_task_init_states(410)[0])
                for step in range(705):
                    if step<5:raw,_,done,_=env.step(s.ev.get_libero_dummy_action())
                    else:
                        frames.append(s.ev.get_libero_image(raw))
                        if not queue:
                            obs=s.ev._prepare_observation(raw)
                            a=model.predict_action(image=[s.Image.fromarray(obs['full_image']),s.Image.fromarray(obs['wrist_image'])],
                                instruction=instruction,unnorm_key=cfg.unnorm_key,proprio=obs['state'])
                            calls.append(a.copy())
                            if variant=='new_loop_float32':a=a.astype(s.np.float32)
                            queue.extend(s.ev._process_action(a))
                        raw,_,done,_=env.step(s.np.asarray(queue.popleft()).tolist())
                    if done:success=True;break
        finally:
            s.ev.get_action=original_get_action;env.close()
        video=OUT/f'{variant}_task410_seed7_success{int(success)}.mp4'
        s.imageio.mimwrite(video,frames,fps=30)
        s.np.savez_compressed(OUT/f'{variant}_actions.npz',**{f'replan{i+1}':a for i,a in enumerate(calls)})
        rec=dict(variant=variant,success=bool(success),control_steps=len(frames),replans=len(calls),
            raw_output_dtype=str(calls[0].dtype),elapsed_s=s.time.perf_counter()-tick,video=str(video))
        results.append(rec);s.save('results.json',results);s.emit('EPISODE_DONE',**rec)
    comparisons=[]
    reference=s.np.load(OUT/'historical_loop_actions.npz')
    for variant in ['new_loop_float32','new_loop_float64']:
        candidate=s.np.load(OUT/f'{variant}_actions.npz')
        keys=[k for k in reference.files if k in candidate.files]
        errors=[float(s.np.max(s.np.abs(reference[k]-candidate[k]))) for k in keys]
        comparisons.append(dict(variant=variant,common_replans=len(keys),first_raw_action_max_abs=errors[0],
            first_divergent_raw_replan=next((i+1 for i,e in enumerate(errors) if e>1e-8),None),
            max_raw_action_abs_difference=max(errors)))
    s.save('summary.json',dict(status='complete',instruction_checks=checks,results=results,comparisons=comparisons,
        elapsed_s=s.time.perf_counter()-started,peak_allocated_mib=s.torch.cuda.max_memory_allocated()/2**20))
    s.emit('COMPLETE',comparisons=comparisons)

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True);s.OUT=OUT;s.save('failure.json',dict(traceback=s.traceback.format_exc()));raise
