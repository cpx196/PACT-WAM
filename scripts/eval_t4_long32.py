"""Per-task offline calibration then 18 seed7 max32 trials, historical baseline contexts pinned."""
import hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from t4_per_task_runtime import Monitor,rng_hash
from t4_long32_core import Schedule,weighted_tau,fit_random_probability,selftest
ROOT=Path('/data/pxchen/PACT-WAM')
OUT=Path(os.environ['T4_PILOT_OUT'])
REFERENCE=ROOT/'evaluate_results/t4_seed7_baseline_reference_20261021'
TASKS=(151,242,260,321,559,590)
CAL_SEEDS=(57,67)
NS=None
MONITOR=None
CAL_ROWS=[]
TEST_ROWS=[]
STAGE='PREFLIGHT'

def save(name,value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)
def status(state,**kw):
    row=dict(state=state,stage=STAGE,updated=time.strftime('%F %T'),calibration_completed=len(CAL_ROWS),calibration_total=12,evaluation_completed=len(TEST_ROWS),evaluation_total=18,**kw)
    save('progress.json',row);print('LONG32 '+json.dumps(row,ensure_ascii=False),flush=True)
def append(name,row):
    with (OUT/name).open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def compact(row):return {k:v for k,v in row.items() if k not in ('actions','windows','replay')}

def preencode(cfg,*,device,dtype):
    from experiments.libero.text_context_cache import cache_identity,load_cache,record_audit
    manifest=json.loads((REFERENCE/'manifest.json').read_text())
    expected=os.environ['LONG32_BASELINE_CACHE_SHA256']
    assert manifest['text_cache_sha256']==expected
    path=Path(manifest['text_cache_path']);assert str(path)==os.environ['LONG32_BASELINE_CACHE_PATH']
    identity=manifest['encoding_provenance']
    _,text,_,token=NS['_resolve_configs'](model_id=str(cfg.model.model_id),tokenizer_model_id=str(cfg.model.tokenizer_model_id),redirect_common_files=bool(cfg.model.get('redirect_common_files',True)))
    # Resolver fills existing local paths here; offline flags prohibit downloads.
    text.download_if_necessary();token.download_if_necessary()
    actual=cache_identity(text.path,token.path,int(cfg.model.get('tokenizer_max_len',cfg.data.train.get('context_len',128))),dtype)
    for key,value in actual.items():
        if key!='encoding_batch_size':assert identity[key]==value,('Historical cache asset mismatch',key)
    assert identity['encoding_batch_size']=='historical_mixed_1_and_5'
    cache,checksum=load_cache(path,identity,NS['_evaluation_prompts'](cfg),expected)
    record_audit(OUT,cache,identity,'explicit_historical_baseline_compatibility',path,checksum)
    return cache

def initial_env(cfg,bench,tid,seed):
    NS['set_global_seed'](seed,get_worker_init_fn=False);cfg.seed=seed;cfg.EVALUATION.task_id=tid
    env,description=NS['get_libero_env'](bench.get_task(tid),NS['LIBERO_ENV_RESOLUTION'],seed)
    try:
        env.reset();NS['apply_libero_camera'](env,cfg.EVALUATION.vjepa2_ac.camera)
        obs=env.set_init_state(bench.get_task_init_states(tid)[0])
        for _ in range(int(cfg.EVALUATION.num_steps_wait)):obs,_,_,_=env.step(NS['get_libero_dummy_action']())
        return env,description,obs
    except BaseException:env.close();raise

def predict(obs,description,kw,pid,horizon=32):
    return NS['_predict_action_chunk'](obs=obs,previous_jepa_image=None,task_description=description,model=kw['model'],processor=kw['processor'],cfg=kw['cfg'],action_horizon=kw['action_horizon'],input_w=kw['input_w'],input_h=kw['input_h'],model_device=kw['model_device'],executed_horizon=horizon,replan_index=pid)

def run_episode(env,description,obs,kw,*,tid,seed,arm,limit=700,tau=None,probability=None,monitor=False,capture=False,log_name='windows.jsonl',video=True):
    policy=Schedule(arm,tau,probability,random_seed=701000+tid)
    n=0;pid=0;done=False;actions=[];windows=[];replay=[];calls=[];env_s=0.;started=time.perf_counter()
    while n<limit and not done:
        plan_start=n
        commands,imgs,frames,timing=predict(obs,description,kw,pid)
        assert frames is None and commands.shape==(32,7) and np.isfinite(commands).all()
        calls.append(timing)
        for offset in range(0,32,4):
            if n>=limit or done:break
            start=n;state=NS['_extract_sim_state'](obs).copy()
            four=commands[offset:offset+4].copy()
            primary=NS['_agentview_for_jepa'](NS['get_libero_image'](obs,image_rotation_degrees=180),kw['processor']).copy()
            remaining=min(4,limit-n)
            cache=MONITOR.start(primary,state,four) if monitor and remaining==4 else None
            actual=[]
            for j in range(remaining):
                command=four[j].tolist()
                if video:replay.append(NS['get_libero_image'](obs,image_rotation_degrees=180))
                tic=time.perf_counter();obs,_,done,_=env.step(command);env_s+=time.perf_counter()-tic
                actions.append(command);actual.append(command);n+=1
                if done:break
            complete=len(actual)==4
            assert np.array_equal(np.asarray(actual,dtype=four.dtype),four[:len(actual)]),'Predicted/executed commands differ'
            age=n-plan_start
            row=dict(stage=STAGE,arm=arm,task_id=tid,seed=seed,trial=0,initial_state_index=0,plan_id=pid,plan_offset=offset,control_start=start,control_end=n,plan_age=age,state=state.tolist(),commands=four.tolist(),actual_commands=actual,tau=tau,probability=probability,censored=not complete,terminal=bool(done),wall_s=time.perf_counter()-started)
            if cache is not None and complete:
                future=NS['_agentview_for_jepa'](NS['get_libero_image'](obs,image_rotation_degrees=180),kw['processor']).copy()
                row.update(MONITOR.finish(cache,future))
                if capture:
                    folder=OUT/'calibration_raw'/f'task{tid}'/f'seed{seed}'/f'window{len(windows):04d}';folder.mkdir(parents=True,exist_ok=False)
                    Image.fromarray(primary).save(folder/'current.png');Image.fromarray(future).save(folder/'future4.png')
                    np.savez_compressed(folder/'inputs.npz',state=state,commands=four)
                    row['raw_dir']=str(folder.relative_to(OUT))
                    (folder/'window.json').write_text(json.dumps(row,ensure_ascii=False,indent=2))
            if done:decision='terminal'
            elif n>=limit:decision='step_limit'
            else:decision=policy.decision(age,row.get('error'))
            row['decision']=decision
            row['eligible_decision']=complete and not done and n<limit and age<32
            append(log_name,row);windows.append(row)
            if STAGE=='CALIBRATION' or len(windows)%4==0:
                status('RUNNING',task_id=tid,seed=seed,arm=arm,control_step=n,wam_calls=len(calls),monitor_windows=sum('error' in w for w in windows),last_error=row.get('error'),last_decision=decision)
            if decision in ('error_replan','random_replan','max32','terminal','step_limit'):break
        pid+=1
    wall=time.perf_counter()-started
    array=np.asarray(actions,dtype=np.float32)
    return dict(task_id=tid,seed=seed,arm=arm,trial=0,initial_state_index=0,success=bool(done),control_steps=n,wam_calls=len(calls),wam_calls_per100=100*len(calls)/n,wall_s=wall,wam_s=sum(c['total_s'] for c in calls),monitor_s=sum(sum(w.get('timing',{}).values()) for w in windows),env_s=env_s,valid_error_windows=sum('error' in w for w in windows),eligible_decisions=sum(w['eligible_decision'] for w in windows),early=sum(w['decision'] in ('error_replan','random_replan') for w in windows),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,actions_sha256=hashlib.sha256(array.tobytes()).hexdigest(),rng=rng_hash(),tau=tau,probability=probability,task_description=description,actions=array,windows=windows,replay=replay)

def summarize(partial=True):
    reference=json.loads((REFERENCE/'baseline_summary.json').read_text())
    groups={}
    for arm in ('fixed32','adaptive32','random32'):
        rows=[r for r in TEST_ROWS if r['arm']==arm]
        if rows:groups[arm]=dict(n=len(rows),successes=sum(r['success'] for r in rows),wam_calls_per100=100*sum(r['wam_calls'] for r in rows)/sum(r['control_steps'] for r in rows),wall_s=sum(r['wall_s'] for r in rows),monitor_s=sum(r['monitor_s'] for r in rows))
    save('summary.json',dict(partial=partial,baseline_reference=str(REFERENCE),baseline=reference,calibration_rollouts=CAL_ROWS,test_rollouts=TEST_ROWS,groups=groups,calibration_seeds=list(CAL_SEEDS),test_seed=7,guidance=False,warning='Selected six diagnostic tasks, one seed/init per evaluation arm. Random budget is matched on offline replay expectation only; report observed online rate gap.'))
    lines=['# 六任务长计划与每4步止损评测',f'\n校准 {len(CAL_ROWS)}/12；正式评测 {len(TEST_ROWS)}/18。保留原 seed7 无guidance fixed8 基准。','\n|Task|原fixed8|fixed32|adaptive32|random32|','|---|---|---|---|---|']
    def cell(r):return ('成功' if r['success'] else '失败')+f"；{r['control_steps']}步/{r['wam_calls']}次WAM" if r else '待完成'
    for tid in TASKS:
        old=next(r for r in reference['rollouts'] if r['task_id']==tid)
        lines.append('|'+str(tid)+'|'+cell(old)+'|'+'|'.join(cell(next((r for r in TEST_ROWS if r['task_id']==tid and r['arm']==arm),None)) for arm in ('fixed32','adaptive32','random32'))+'|')
    lines+=['\n每任务使用独立seed57/67的fixed32 shadow轨迹，逐4步缓存原始输入后离线重算误差；按轨迹等权75%分位冻结阈值。随机概率只按校准误差流的提前重规划代理预算拟合，不使用seed7测试结果。在线路径会改变，实际调用预算未必相等，需报告实际差距。','\n每个4步窗口从最新真实观测/状态出发，输入原计划接下来4条动作，用现有phase0 Predictor预测下一段；没有修改动作或加入guidance。固定32及随机组评测不运行误差监控，监控开销单列；不以墙钟公平性替代WAM调用预算核对。','\n阈值未冻结之前不启动任何正式seed7评测。原baseline输入缓存及初始动作一致性必须先全部通过。']
    (OUT/'report_zh.md').write_text('\n'.join(lines))

def driver(ns,**kw):
    global NS,MONITOR,STAGE
    NS=ns;cfg=kw['cfg']
    assert kw['action_horizon']==32 and int(cfg.EVALUATION.replan_steps)==32
    assert int(cfg.EVALUATION.num_steps_wait)==30
    assert not cfg.EVALUATION.visualize_future_video and not cfg.EVALUATION.vjepa2_ac.enabled and not cfg.EVALUATION.vjepa2_ac.guidance.enabled
    assert not cfg.EVALUATION.trigger_probe.enabled and not cfg.EVALUATION.use_action_ensembler
    save('runtime_config.json',ns['OmegaConf'].to_container(cfg,resolve=True))
    bench=ns['benchmark'].get_benchmark_dict()['libero_10'](task_order_index=0)
    baseline_windows=[json.loads(line) for line in (REFERENCE/'baseline_windows.jsonl').read_text().splitlines()]
    preflight=[]
    status('CHECKING_HISTORICAL_BASELINE')
    for tid in TASKS:
        env,description,obs=initial_env(cfg,bench,tid,7)
        try:
            ref=next(w for w in baseline_windows if w['task_id']==tid and w['control_start']==0)
            state=ns['_extract_sim_state'](obs)
            a,_,_,_=predict(obs,description,kw,0)
            b,_,_,_=predict(obs,description,kw,0,horizon=8)
            row=dict(task_id=tid,state_equal=bool(np.array_equal(state,np.asarray(ref['state'],np.float32))),historical_first4_equal=bool(np.array_equal(a[:4],np.asarray(ref['commands'],np.float32))),historical_first4_max_abs=float(np.abs(a[:4]-np.asarray(ref['commands'],np.float32)).max()),full_actions_repeat_and_horizon_equal=bool(np.array_equal(a,b)))
            preflight.append(row);save('baseline_repro_preflight.json',dict(passed=False,tasks=preflight))
            assert row['state_equal'] and row['historical_first4_equal'] and row['full_actions_repeat_and_horizon_equal'],row
            status('BASELINE_TASK_VERIFIED',task_id=tid)
        finally:env.close()
    save('baseline_repro_preflight.json',dict(passed=True,tasks=preflight))
    MONITOR=Monitor(kw['model_device'])
    def run(tid,seed,arm,*,limit=700,tau=None,probability=None,monitor=False,capture=False,smoke=False):
        env,description,obs=initial_env(cfg,bench,tid,seed)
        status('SMOKE' if smoke else 'START_EPISODE',task_id=tid,seed=seed,arm=arm)
        try:
            row=run_episode(env,description,obs,kw,tid=tid,seed=seed,arm=arm,limit=limit,tau=tau,probability=probability,monitor=monitor,capture=capture,log_name='smoke_windows.jsonl' if smoke else 'windows.jsonl',video=not smoke)
            tic=time.perf_counter()
            if not smoke:
                folder=OUT/('calibration' if STAGE=='CALIBRATION' else 'evaluation')/f'task{tid}'/f'seed{seed}'/arm/'videos';folder.mkdir(parents=True,exist_ok=True)
                ns['save_rollout_video'](folder,row['replay'],f'task{tid}_trial0',success=row['success'],task_description=description)
            row['video_io_s']=time.perf_counter()-tic
            return row
        finally:env.close()
    STAGE='SMOKE'
    # Native fixed32 vs independent controller, including a partial final plan.
    ns['set_global_seed'](57,get_worker_init_fn=False);cfg.seed=57;cfg.EVALUATION.task_id=151
    env,description=ns['get_libero_env'](bench.get_task(151),ns['LIBERO_ENV_RESOLUTION'],57)
    original=ns['_predict_action_chunk'];native_actions=[]
    def recorded(*args,**kwargs):
        result=original(*args,**kwargs);native_actions.extend(result[0][:32].tolist());return result
    ns['_predict_action_chunk']=recorded;cfg.EVALUATION.max_steps_override=40
    try:
        ns['run_single_episode'](env=env,initial_state=bench.get_task_init_states(151)[0],task_description=description,model=kw['model'],processor=kw['processor'],cfg=cfg,episode_idx=0,action_horizon=kw['action_horizon'],input_w=kw['input_w'],input_h=kw['input_h'],model_device=kw['model_device'])
        native_rng=rng_hash()
    finally:ns['_predict_action_chunk']=original;env.close()
    off=run(151,57,'fixed32',limit=40,smoke=True)
    on=run(151,57,'fixed32',limit=40,monitor=True,smoke=True)
    adaptive=run(151,57,'adaptive32',limit=40,tau=1e9,monitor=True,smoke=True)
    random0=run(151,57,'random32',limit=40,probability=0.,smoke=True)
    assert np.array_equal(off['actions'],np.asarray(native_actions[:40],np.float32)) and off['rng']==native_rng
    for row in (on,adaptive,random0):
        assert np.array_equal(off['actions'],row['actions']) and off['rng']==row['rng']
        assert row['wam_calls']==2
    assert on['valid_error_windows']==10 and adaptive['valid_error_windows']==10
    adaptive4=run(151,57,'adaptive32',limit=12,tau=-1.,monitor=True,smoke=True)
    random4=run(151,57,'random32',limit=12,probability=1.,smoke=True)
    assert adaptive4['wam_calls']==random4['wam_calls']==3
    assert np.array_equal(adaptive4['actions'],random4['actions']) and adaptive4['rng']==random4['rng']
    save('smoke_summary.json',dict(passed=True,pure_core=selftest(),all_six_historical_first4_equal=True,native_fixed32_actions_rng_equal=True,passive_monitor_actions_rng_equal=True,adaptive_no_trigger_equals_fixed32=True,random_p0_equals_fixed32=True,adaptive_all_trigger_equals_random_p1=True,max32_and_partial_final_plan=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
    STAGE='CALIBRATION';summarize()
    for tid in TASKS:
        for seed in CAL_SEEDS:
            row=run(tid,seed,'fixed32',monitor=True,capture=True)
            CAL_ROWS.append(compact(row));save(f'calibration/task{tid}/seed{seed}/result.json',compact(row));summarize()
            status('EPISODE_COMPLETED',task_id=tid,seed=seed,arm='fixed32',success=row['success'])
    STAGE='OFFLINE_CALIBRATION';status('OFFLINE_REPLAY')
    by_task={};max_difference=0.;replayed=0
    for tid in TASKS:
        trajectories=[];hashes={};counts=[]
        for seed in CAL_SEEDS:
            values=[];h=hashlib.sha256()
            for meta_path in sorted((OUT/'calibration_raw'/f'task{tid}'/f'seed{seed}').glob('window*/window.json')):
                folder=meta_path.parent;meta=json.loads(meta_path.read_text());inputs=np.load(folder/'inputs.npz',allow_pickle=False)
                before=np.asarray(Image.open(folder/'current.png').convert('RGB'));after=np.asarray(Image.open(folder/'future4.png').convert('RGB'))
                error=MONITOR.finish(MONITOR.start(before,inputs['state'],inputs['commands']),after)['error']
                difference=abs(error-meta['error']);max_difference=max(max_difference,difference)
                assert difference<=1e-6,('Offline replay mismatch',tid,seed,meta['control_start'],difference)
                for name in ('window.json','current.png','future4.png','inputs.npz'):h.update((folder/name).read_bytes())
                values.append(error);replayed+=1
                append('offline_errors.jsonl',dict(task_id=tid,seed=seed,control_start=meta['control_start'],plan_age=meta['plan_age'],error=error,difference=difference))
            trajectories.append(values);counts.append(len(values));hashes[str(seed)]=h.hexdigest()
        tau=weighted_tau(trajectories)
        by_task[str(tid)]=dict(tau=tau,quantile=.75,trajectory_equal_weights=True,calibration_seeds=list(CAL_SEEDS),counts=counts,raw_sha256=hashes,random_budget=fit_random_probability(trajectories,tau))
        save('calibration_partial.json',dict(by_task=by_task,replayed_windows=replayed,maximum_online_offline_difference=max_difference))
        status('TASK_CALIBRATED',task_id=tid,tau=tau,random_probability=by_task[str(tid)]['random_budget']['probability'])
    save('thresholds.json',dict(frozen=True,state='FROZEN_BEFORE_ALL_SEED7_EVALUATION',freeze_time=time.strftime('%F %T'),by_task=by_task,replayed_windows=replayed,maximum_online_offline_difference=max_difference,calibration_seeds=list(CAL_SEEDS),test_seed=7,baseline_cache_sha256=os.environ['LONG32_BASELINE_CACHE_SHA256']))
    STAGE='EVALUATION';status('PARAMETERS_FROZEN');summarize()
    for tid in TASKS:
        parameters=by_task[str(tid)]
        for arm in ('fixed32','adaptive32','random32'):
            tau=parameters['tau'] if arm=='adaptive32' else None
            p=parameters['random_budget']['probability'] if arm=='random32' else None
            row=run(tid,7,arm,tau=tau,probability=p,monitor=arm=='adaptive32')
            TEST_ROWS.append(compact(row));save(f'evaluation/task{tid}/seed7/{arm}/result.json',compact(row));summarize()
            status('EPISODE_COMPLETED',task_id=tid,seed=7,arm=arm,success=row['success'])
    summarize(False);STAGE='COMPLETED';status('COMPLETED')
    return dict(successes=sum(row['success'] for row in TEST_ROWS),duration=sum(row['wall_s'] for row in CAL_ROWS+TEST_ROWS))

def main():
    global NS
    native=ROOT/'experiments/libero/eval_libero_single.py'
    assert sha(native)==os.environ['LONG32_NATIVE_SHA256'],'Native source changed after snapshot'
    code=native.read_text();needle='if __name__ == "__main__":\n    eval_single_process()';assert code.count(needle)==1
    def install(ns):
        global NS
        NS=ns
        ns['_evaluation_task_keys']=lambda cfg:[('libero_10',t) for t in TASKS]
        ns['_preencode_prompt_contexts']=preencode
        ns['_run_task_to_file']=lambda **kw:(None,driver(ns,**kw))
        original=ns['_load_model_checkpoint']
        def checked(model,checkpoint):
            load=model.mot.load_state_dict
            def check(*a,**k):
                result=load(*a,**k);save('checkpoint_keys.json',dict(missing=result.missing_keys,unexpected=result.unexpected_keys))
                assert not result.missing_keys and not result.unexpected_keys,result
                return result
            model.mot.load_state_dict=check
            try:return original(model,checkpoint)
            finally:model.mot.load_state_dict=load
        ns['_load_model_checkpoint']=checked
    code=code.replace(needle,'if __name__ == "__main__":\n    install_long32(globals())\n    eval_single_process()')
    exec(compile(code,str(native),'exec'),dict(__name__='__main__',__file__=str(native),install_long32=install))
if __name__=='__main__':
    try:main()
    except BaseException:save('failure.json',dict(traceback=traceback.format_exc()));status('FAILED');raise
