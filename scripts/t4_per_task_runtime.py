"""Independent t+4 monitor and synchronous 4/8 pilot. Native loader is reused unchanged."""
import hashlib, json, os, random, time, traceback
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import torch
from PIL import Image
from t4_pilot_core import Schedule, calibrate, selftest
ROOT=Path('/data/pxchen/PACT-WAM')
OUT=Path(os.environ['T4_PILOT_OUT'])
NS=None
MONITOR=None
ROWS=[]
WINDOWS=[]

def save(name,data):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
def status(state,**kw):
    d=dict(state=state,updated=time.strftime('%F %T'),**kw);save('progress.json',d);print('T4_STATUS '+json.dumps(d),flush=True)
def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()
def restore_rng(s):
    random.setstate(s[0]);np.random.set_state(s[1]);torch.set_rng_state(s[2]);torch.cuda.set_rng_state_all(s[3])
def rng_hash():
    s=rng_state();h=hashlib.sha256(repr(s[0]).encode()+s[1][1].tobytes()+repr(s[1][2:]).encode()+s[2].numpy().tobytes())
    for t in s[3]:h.update(t.numpy().tobytes())
    return h.hexdigest()
@contextmanager
def timed(values,key):
    torch.cuda.synchronize();t=time.perf_counter()
    yield
    torch.cuda.synchronize();values[key]=time.perf_counter()-t

def finite(x,shape):
    assert tuple(x.shape)==tuple(shape),(x.shape,shape)
    assert torch.isfinite(x).all(),'Nonfinite tensor'

class Monitor:
    def __init__(self,device):
        from fastwam_ar2_eval import ARRanker
        s=rng_state()
        self.m=ARRanker(checkpoint_path=str(ROOT/'artifacts/fastwam_predictor21_ar2_full10epoch_bs16_20261014/step19690.pt'),device=device,dtype=torch.float32)
        restore_rng(s)
    @torch.inference_mode()
    def start(self,frame,state,commands):
        before=rng_hash();m=self.m;times={}
        with torch.autocast('cuda',enabled=False):
            with timed(times,'current_encoder_s'):
                x=m.pixels(frame)[None].to(m.device,torch.float32);finite(x,(1,3,384,384))
                z=m.encoder(x);finite(z,(1,576,1024))
            with timed(times,'predictor_s'):
                a=torch.tensor(commands[None],device=m.device,dtype=torch.float32)
                s=torch.tensor(state[None],device=m.device,dtype=torch.float32)
                finite(a,(1,4,7));finite(s,(1,8))
                am,ast=m.stats['action'];sm,sst=m.stats['state']
                p=m.predictor(z,(a-am)/ast,(s-sm)/sst,0);finite(p,(1,576,1024))
        assert rng_hash()==before,'Monitor forward consumed policy RNG'
        return dict(frame=frame.copy(),x=x,p=p,z=z,timing=times)
    @torch.inference_mode()
    def finish(self,cache,frame):
        before=rng_hash();m=self.m;times=cache['timing']
        with torch.autocast('cuda',enabled=False):
            with timed(times,'pair_encoder_s'):
                x=m.pixels(frame)[None].to(m.device,torch.float32)
                pair=torch.stack([cache['x'],x],dim=1);finite(pair,(1,2,3,384,384))
                y=m.encoder.encode_pair(pair).reshape(1,576,1024);finite(y,(1,576,1024))
                e=float((cache['p']-y).abs().mean())
                baseline=float((cache['z']-y).abs().mean())
        assert np.isfinite(e) and np.isfinite(baseline)
        assert rng_hash()==before,'Feedback encoding consumed policy RNG'
        return dict(error=e,copy_single_current_latent_l1=baseline,pixel_change_l1=float(np.abs(frame.astype(np.float32)-cache['frame'].astype(np.float32)).mean()),timing=times)

def episode(env,initial_state,task_description,model,processor,cfg,episode_idx,*,action_horizon,input_w,input_h,model_device,ac_ranker=None,ac_adapter=None,action_convergence_logger=None):
    arm=os.environ['T4_ARM'];enabled=os.environ.get('T4_MONITOR','1')=='1'
    policy=Schedule(arm,float(os.environ['T4_TAU']) if arm=='adaptive48' else None)
    limit=int(cfg.EVALUATION.max_steps_override);tid=int(cfg.EVALUATION.task_id)
    assert not cfg.EVALUATION.visualize_future_video
    assert not cfg.EVALUATION.vjepa2_ac.enabled and not cfg.EVALUATION.vjepa2_ac.guidance.enabled
    assert not cfg.EVALUATION.trigger_probe.enabled and not cfg.EVALUATION.use_action_ensembler
    env.reset();NS['apply_libero_camera'](env,cfg.EVALUATION.vjepa2_ac.camera)
    obs=env.set_init_state(initial_state)
    for _ in range(int(cfg.EVALUATION.num_steps_wait)):
        obs,_,_,_=env.step(NS['get_libero_dummy_action']())
    n=0;pid=-1;calls=[];replay=[];all_actions=[];local=[];env_s=0.;started=time.perf_counter();done=False
    while n<limit and not done:
        pid+=1;start=n
        commands,imgs,frames,timing=NS['_predict_action_chunk'](obs=obs,previous_jepa_image=None,task_description=task_description,model=model,processor=processor,cfg=cfg,action_horizon=action_horizon,input_w=input_w,input_h=input_h,model_device=model_device,executed_horizon=8,replan_index=pid)
        assert frames is None and commands.shape[0]>=8 and commands.shape[1]==7
        assert np.isfinite(commands).all()
        calls.append(timing)
        primary=NS['_agentview_for_jepa'](imgs,processor).copy()
        state=NS['_extract_sim_state'](obs).copy();first=commands[:4].copy()
        cache=MONITOR.start(primary,state,first) if enabled else None
        window=dict(arm=arm,task_id=tid,seed=int(os.environ['T4_POLICY_SEED']),trial=0,initial_state_index=0,plan_id=pid,monitor_window_id=pid,control_start=start,wall_start_s=time.perf_counter()-started,state=state.tolist(),commands=first.tolist(),tau=policy.tau,policy_timing=timing)
        executed=[];early=False
        for j in range(min(8,limit-start)):
            # Record the precise API command; no extra gripper processing.
            command=commands[j].tolist();replay.append(NS['get_libero_image'](obs,image_rotation_degrees=180))
            tic=time.perf_counter();obs,_,done,_=env.step(command);env_s+=time.perf_counter()-tic
            all_actions.append(command);n+=1
            if j<4:executed.append(command)
            if done:break
            if j==3:
                assert np.array_equal(np.asarray(executed,dtype=first.dtype),first),'Predictor/env.step command mismatch'
                if enabled:
                    actual=NS['_agentview_for_jepa'](NS['get_libero_image'](obs,image_rotation_degrees=180),processor).copy()
                    window.update(MONITOR.finish(cache,actual));early=policy.early(window['error'])
                    window.update(control_end=n,decision='replan4' if early else 'continue8',censored=False,actual_commands=executed,wall_feedback_s=time.perf_counter()-started)
                    # Bounded lossless diagnostics. Uniform samples and first early replan.
                    if pid%16==0 or (early and not any(w.get('decision')=='replan4' for w in local)):
                        d=OUT/f"seed{os.environ['T4_POLICY_SEED']}"/arm/f'task{tid}'/'diagnostics';d.mkdir(parents=True,exist_ok=True)
                        Image.fromarray(primary).save(d/f'plan{pid}_start.png');Image.fromarray(actual).save(d/f'plan{pid}_end.png')
                        if pid==0:
                            Image.fromarray(np.asarray(obs['agentview_image'])).save(d/'raw_end.png')
                            save(str((d/'pixel_path.json').relative_to(OUT)),dict(native_rotation=180,jepa_additional_rotation=0,crop_shape=list(primary.shape),tensor_shape=[1,3,384,384],target='encode_pair'))
                if early:break
        if enabled:
            if 'error' not in window:
                window.update(control_end=n,censored=True,decision='terminal_or_short_window',actual_commands=executed)
            window['plan_executed_steps']=n-start
            local.append(window);WINDOWS.append(window)
            with (OUT/'windows.jsonl').open('a') as f:f.write(json.dumps(window,allow_nan=False)+'\n');f.flush()
            if pid%10==0:status('RUNNING',arm=arm,task_id=tid,control_step=n,completed=len(ROWS),total=int(os.environ.get('T4_TOTAL',18)),windows=len(local))
    wall=time.perf_counter()-started
    timing=dict(episode_wall_s=wall,replan_count=len(calls),policy_replan_total_s=sum(c['total_s'] for c in calls),env_step_total_s=env_s,low_level_action_steps=n,per_replan=calls)
    episode.last=dict(success=bool(done),control_steps=n,wam_calls=len(calls),wam_calls_per100=100*len(calls)/n,wall_s=wall,wam_s=timing['policy_replan_total_s'],monitor_s=sum(sum(w.get('timing',{}).values()) for w in local),env_s=env_s,valid_decisions=sum(not w['censored'] for w in local),early=sum(w.get('decision')=='replan4' for w in local),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,actions=all_actions,rng=rng_hash(),windows=local)
    return bool(done),replay,[],None,timing

def driver(ns,**kw):
    global NS,MONITOR
    NS=ns;cfg=kw['cfg'];model=kw['model'];processor=kw['processor'];device=kw['model_device']
    bench=ns['benchmark'].get_benchmark_dict()['libero_10'](task_order_index=0)
    MONITOR=Monitor(device)
    ns['run_single_episode']=episode
    def run(tid,arm,limit=700,monitor=True,native=False):
        os.environ['T4_ARM']=arm;os.environ['T4_MONITOR']='1' if monitor else '0'
        ns['set_global_seed'](47,get_worker_init_fn=False);cfg.EVALUATION.task_id=tid;cfg.EVALUATION.max_steps_override=limit
        env,description=ns['get_libero_env'](bench.get_task(tid),ns['LIBERO_ENV_RESOLUTION'],47)
        status('START_EPISODE',task_id=tid,arm=arm,max_steps=limit,monitor=monitor,completed=len(ROWS),total=int(os.environ.get('T4_TOTAL',18)))
        try:
            args=dict(env=env,initial_state=bench.get_task_init_states(tid)[0],task_description=description,model=model,processor=processor,cfg=cfg,episode_idx=0,action_horizon=kw['action_horizon'],input_w=kw['input_w'],input_h=kw['input_h'],model_device=device)
            if native:
                captured=[];original=ns['_predict_action_chunk']
                def record(*a,**k):
                    result=original(*a,**k);captured.extend(result[0][:8].tolist());return result
                ns['_predict_action_chunk']=record
                try: NATIVE(**args)
                finally: ns['_predict_action_chunk']=original
                return dict(actions=captured[:limit],rng=rng_hash())
            outcome=episode(**args)
            io_start=time.perf_counter()
            if limit==700:
                ns['save_rollout_video'](OUT/arm/f'task{tid}'/'videos',outcome[1],f'task{tid}_trial0',success=outcome[0],task_description=description)
            episode.last['video_io_s']=time.perf_counter()-io_start
            result=dict(episode.last,task_id=tid,arm=arm,seed=47,initial_state_index=0,trial=0,task_description=description)
            return result
        finally:env.close()
    # Native control-loop equivalence and passive monitor RNG equivalence, 3 plans each.
    status('SMOKE')
    native=run(260,'fixed8',24,False,True)
    off=run(260,'fixed8',24,False)
    on=run(260,'fixed8',24,True)
    assert np.array_equal(native['actions'],off['actions']),'Independent loop differs from native'
    assert native['rng']==off['rng'],'Independent loop RNG differs from native'
    assert np.array_equal(off['actions'],on['actions']),'Shadow monitor changes actions'
    assert off['rng']==on['rng'],'Shadow monitor changes RNG'
    # Full-video / no-decode path equivalence on identical real observation and RNG.
    env,description=ns['get_libero_env'](bench.get_task(260),ns['LIBERO_ENV_RESOLUTION'],47)
    try:
        env.reset();obs=env.set_init_state(bench.get_task_init_states(260)[0])
        for _ in range(30):obs,_,_,_=env.step(ns['get_libero_dummy_action']())
        base=dict(obs=obs,previous_jepa_image=None,task_description=description,model=model,processor=processor,cfg=cfg,action_horizon=kw['action_horizon'],input_w=kw['input_w'],input_h=kw['input_h'],model_device=device,executed_horizon=8)
        s=rng_state();a=ns['_predict_action_chunk'](**base)[0]
        restore_rng(s);cfg.EVALUATION.visualize_future_video=True
        try:b=ns['_predict_action_chunk'](**base)[0]
        finally:cfg.EVALUATION.visualize_future_video=False
        delta=float(np.abs(a[:8]-b[:8]).max());assert delta<=1e-6,('Decode path mismatch',delta)
    finally:env.close()
    # Boundary checks use genuine monitor calls with mock terminal timing, never formal fabricated errors.
    test_boundaries(ns,kw,bench,model,processor,cfg)
    latency=run(260,'fixed8',160,True)
    assert latency['valid_decisions']>=20,'Fewer than 20 smoke monitoring cycles'
    save('smoke_latency.json',{k:v for k,v in latency.items() if k!='actions'})
    save('smoke_summary.json',dict(passed=True,cpu=selftest(),native_equivalence=True,shadow_actions_equal=True,shadow_rng_equal=True,fullvideo_no_decode_max_abs=delta,action_atol=1e-6,phase=0,current_shape=[1,576,1024],target_shape=[1,576,1024],target_api='encode_pair',boundary_checks=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
    # Remove smoke observations from the formal window stream without discarding evidence.
    (OUT/'windows.jsonl').rename(OUT/'smoke_windows.jsonl');WINDOWS.clear()
    calibration=[]
    for tid in (260,321):
        r=run(tid,'calibration');calibration.append(r);store_result(r)
    values=[[w['error'] for w in r['windows'] if not w['censored']] for r in calibration]
    tau=calibrate(values);os.environ['T4_TAU']=str(tau)
    source=OUT/'windows.jsonl'
    save('threshold.json',dict(tau=tau,quantile=.75,weighting='each trajectory total weight 1/2; inverse weighted ECDF',tasks=[260,321],seed=47,counts=list(map(len,values)),calibration_manifest_sha256=hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest(),calibration_windows_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),frozen=True))
    for arm in ('fixed8','adaptive48','random48'):
        for tid in (151,242,559,590):store_result(run(tid,arm))
    valid=[w for w in WINDOWS if w['arm']=='adaptive48' and not w['censored']]
    q=sum(w['decision']=='replan4' for w in valid)/len(valid)
    if len(valid)>=64 and .05<=q<=.90 and time.strftime('%H:%M')<'20:30':
        for tid in (151,242,559,590):store_result(run(tid,'fixed4'))
    from analyze_fastwam_t4_replan_pilot import analyze
    analyze(OUT)
    status('COMPLETED',completed=len(ROWS),total=len(ROWS),tau=tau)
    return dict(successes=0,duration=0)

def store_result(r):
    compact={k:v for k,v in r.items() if k not in ('actions','windows')}
    compact['actions_sha256']=hashlib.sha256(np.asarray(r['actions'],np.float32).tobytes()).hexdigest()
    ROWS.append(compact);save(f"{r['arm']}/task{r['task_id']}/result.json",compact);save('partial_results.json',ROWS)
    from analyze_fastwam_t4_replan_pilot import analyze
    analyze(OUT,partial=True)
    status('EPISODE_COMPLETED',arm=r['arm'],task_id=r['task_id'],success=r['success'],completed=len(ROWS),total=int(os.environ.get('T4_TOTAL',18)))

def test_boundaries(ns,kw,bench,model,processor,cfg):
    # Exercise truncation and terminal windows in the real controller, using real model outputs.
    env,description=ns['get_libero_env'](bench.get_task(260),ns['LIBERO_ENV_RESOLUTION'],47)
    class DoneAfter:
        def __init__(self,wrapped,steps):self.wrapped=wrapped;self.steps=steps;self.n=0
        def __getattr__(self,name):return getattr(self.wrapped,name)
        def step(self,a):
            result=self.wrapped.step(a);self.n+=1
            return result[0],result[1],self.n>=30+self.steps,result[3]
    try:
        for steps,forced in ((3,False),(3,True),(4,True)):
            ns['set_global_seed'](47,get_worker_init_fn=False);cfg.EVALUATION.max_steps_override=steps if not forced else 9
            os.environ['T4_ARM']='fixed8';os.environ['T4_MONITOR']='1'
            episode(env=DoneAfter(env,steps) if forced else env,initial_state=bench.get_task_init_states(260)[0],task_description=description,model=model,processor=processor,cfg=cfg,episode_idx=0,action_horizon=kw['action_horizon'],input_w=kw['input_w'],input_h=kw['input_h'],model_device=kw['model_device'])
            assert episode.last['windows'][-1]['censored'] and episode.last['valid_decisions']==0
    finally:env.close()

def main():
    global NATIVE
    source=ROOT/'experiments/libero/eval_libero_single.py';code=source.read_text()
    assert hashlib.sha256(code.encode()).hexdigest()=='8a24a14bdfdc484a925ca8ad35276a87754a6cf8232c88b0c6f86ea40d2c29bc','Native source changed: re-audit required'
    needle='if __name__ == "__main__":\n    eval_single_process()';assert code.count(needle)==1
    code=code.replace(needle,'if __name__ == "__main__":\n    install_driver(globals())\n    eval_single_process()')
    def install(ns):
        global NATIVE
        NATIVE=ns['run_single_episode']
        ns['_evaluation_task_keys']=lambda cfg: [('libero_10',tid) for tid in (151,242,260,321,559,590)]
        ns['_run_task_to_file']=lambda **kw: (None,driver(ns,**kw))
    exec(compile(code,str(source),'exec'),dict(__name__='__main__',__file__=str(source),install_driver=install))
if __name__=='__main__':
    try:main()
    except BaseException:
        save('failure.json',dict(traceback=traceback.format_exc()));status('FAILED');raise
