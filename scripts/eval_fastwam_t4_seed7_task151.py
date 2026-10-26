"""Seed7/task151 diagnostic: B8/A48/R48, original frozen threshold, no guidance."""
import hashlib,json,os,time,traceback
import numpy as np
import t4_seed7_runtime as runtime
from pathlib import Path
OUT=runtime.OUT
SOURCE=runtime.ROOT/'evaluate_results/fastwam_t4_replan_pilot_20261020'
ROWS=[]

def summarize(partial):
    rows={r['arm']:r for r in ROWS}
    s=dict(partial=partial,seed=7,task_id=151,rollouts=ROWS,interpretation='Seen historical failure case; mechanism diagnostic, not an independent generalization test.')
    if len(rows)==3:
        a,r,b=[rows[k] for k in ('adaptive48','random48','fixed8')]
        s.update(baseline_failure_reproduced=not b['success'],adaptive_rescue=a['success'] and not b['success'],adaptive_harm=b['success'] and not a['success'],random_rescue=r['success'] and not b['success'],random_harm=b['success'] and not r['success'],call_rate_gap=abs(a['wam_calls_per100']/r['wam_calls_per100']-1),stage_time_gap=abs((a['wam_s']+a['monitor_s'])/(r['wam_s']+r['monitor_s'])-1))
        s['budget_matched']=max(s['call_rate_gap'],s['stage_time_gap'])<=.15
    runtime.save('summary.json',s)
    lines=['# seed7 / task151 复现诊断',f"\n状态：{'进行中' if partial else '完成'}。阈值沿用seed47校准结果，不重新校准。",'\n|组|成功|控制步|WAM次数|WAM/100步|提前/有效决策|墙钟秒|','|---|---|---|---|---|---|---|']
    for r in ROWS:lines.append(f"|{r['arm']}|{r['success']}|{r['control_steps']}|{r['wam_calls']}|{r['wam_calls_per100']:.2f}|{r['early']}/{r['valid_decisions']}|{r['wall_s']:.1f}|")
    lines+=['\n只用于复现历史关注现象和检查触发时机；单task单seed不能证明泛化或可靠因果机制。终局成功标志不是触发输入。若预算差异超过15%，不能隔离时机收益。局部掉落是否复现需检查录像。']
    runtime.save('partial_results.json',ROWS)
    (OUT/'report_zh.md').write_text('\n'.join(lines))

def driver(ns,**kw):
    runtime.ROWS=ROWS
    runtime.NS=ns;cfg=kw['cfg'];model=kw['model'];processor=kw['processor']
    assert int(cfg.seed)==7 and not cfg.EVALUATION.visualize_future_video
    assert not cfg.EVALUATION.vjepa2_ac.enabled and not cfg.EVALUATION.vjepa2_ac.guidance.enabled
    runtime.save('runtime_config.json',ns['OmegaConf'].to_container(cfg,resolve=True))
    tau=json.loads((OUT/'threshold.json').read_text())['tau']
    assert tau==0.7292823195457458
    os.environ['T4_TAU']=str(tau)
    bench=ns['benchmark'].get_benchmark_dict()['libero_10'](task_order_index=0)
    runtime.MONITOR=runtime.Monitor(kw['model_device'])
    def run(arm,limit,monitor=True,native=False):
        os.environ['T4_ARM']=arm;os.environ['T4_MONITOR']='1' if monitor else '0'
        ns['set_global_seed'](7,get_worker_init_fn=False)
        cfg.EVALUATION.task_id=151;cfg.EVALUATION.max_steps_override=limit
        env,description=ns['get_libero_env'](bench.get_task(151),ns['LIBERO_ENV_RESOLUTION'],7)
        runtime.status('SMOKE' if limit!=700 else 'START_EPISODE',arm=arm,seed=7,task_id=151,completed=len(ROWS),total=3)
        args=dict(env=env,initial_state=bench.get_task_init_states(151)[0],task_description=description,model=model,processor=processor,cfg=cfg,episode_idx=0,action_horizon=kw['action_horizon'],input_w=kw['input_w'],input_h=kw['input_h'],model_device=kw['model_device'])
        try:
            if native:
                captured=[];original=ns['_predict_action_chunk']
                def record(*a,**k):
                    result=original(*a,**k);captured.extend(result[0][:8].tolist());return result
                ns['_predict_action_chunk']=record
                try:runtime.NATIVE(**args)
                finally:ns['_predict_action_chunk']=original
                return dict(actions=captured[:limit],rng=runtime.rng_hash())
            outcome=runtime.episode(**args)
            r=dict(runtime.episode.last,seed=7,task_id=151,arm=arm,trial=0,initial_state_index=0,task_description=description)
            io=time.perf_counter()
            if limit==700:
                d=OUT/arm/'task151'/'videos';d.mkdir(parents=True,exist_ok=True)
                ns['save_rollout_video'](d,outcome[1],'task151_trial0',success=outcome[0],task_description=description)
            r['video_io_s']=time.perf_counter()-io
            return r
        finally:env.close()
    # Recheck execution and RNG equivalence for this seed before collecting effects.
    native=run('fixed8',24,False,True)
    off=run('fixed8',24,False)
    on=run('fixed8',24,True)
    assert np.array_equal(native['actions'],off['actions']) and native['rng']==off['rng']
    assert np.array_equal(off['actions'],on['actions']) and off['rng']==on['rng']
    runtime.save('smoke_summary.json',dict(passed=True,seed=7,task_id=151,native_actions_equal=True,native_rng_equal=True,shadow_actions_equal=True,shadow_rng_equal=True,phase=0,parent_interface_smoke=str(SOURCE/'smoke_summary.json')))
    (OUT/'windows.jsonl').rename(OUT/'smoke_windows.jsonl');runtime.WINDOWS.clear()
    for arm in ('fixed8','adaptive48','random48'):
        r=run(arm,700)
        r['actions_sha256']=hashlib.sha256(np.asarray(r.pop('actions'),np.float32).tobytes()).hexdigest();r.pop('windows')
        ROWS.append(r);runtime.save(f'{arm}/task151/result.json',r);summarize(partial=len(ROWS)<3)
        runtime.status('EPISODE_COMPLETED',arm=arm,success=r['success'],seed=7,task_id=151,completed=len(ROWS),total=3)
    runtime.status('COMPLETED',seed=7,task_id=151,completed=3,total=3)
    return dict(successes=int(ROWS[0]['success']),duration=sum(r['wall_s'] for r in ROWS))

def main():
    runtime.driver=driver
    runtime.main()
if __name__=='__main__':
    try:main()
    except BaseException:
        runtime.save('failure.json',dict(traceback=traceback.format_exc()));runtime.status('FAILED');raise
