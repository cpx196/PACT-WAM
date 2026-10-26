"""Seed7/task151: one random15 rollout, original frozen threshold, no guidance."""
import hashlib,json,os,time,traceback
import numpy as np
import t4_seed7_r15_runtime as runtime
from pathlib import Path
OUT=runtime.OUT
SOURCE=runtime.ROOT/'evaluate_results/fastwam_t4_replan_pilot_20261020'
ROWS=[]

def summarize(partial):
    previous=runtime.ROOT/'evaluate_results/fastwam_t4_seed7_task151_20261020'
    reference=json.loads((previous/'summary.json').read_text())['rollouts']
    combined=reference+ROWS
    s=dict(partial=partial,seed=7,task_id=151,random_early_probability=.15,independent_schedule_rng_seed=470048,rollouts=ROWS,comparison_rollouts=combined,recalibrated=False)
    if ROWS:
        r=ROWS[0];a=next(x for x in reference if x['arm']=='adaptive48')
        s.update(actual_early_rate=r['early']/r['valid_decisions'],call_rate_gap_vs_adaptive=abs(r['wam_calls_per100']/a['wam_calls_per100']-1),stage_time_gap_vs_adaptive=abs((r['wam_s']+r['monitor_s'])/(a['wam_s']+a['monitor_s'])-1))
        s['budget_matched_vs_adaptive']=max(s['call_rate_gap_vs_adaptive'],s['stage_time_gap_vs_adaptive'])<=.15
    runtime.save('summary.json',s);runtime.save('partial_results.json',ROWS)
    lines=['# seed7 / task151 随机15%补测','\n预先固定每次四步决策以q=0.15概率提前replan；不是严格配额，也不是额外15%的WAM预算。不使用误差阈值进行决策。','\n|组|成功|控制步|WAM次数|WAM/100步|提前/有效决策|墙钟秒|','|---|---|---|---|---|---|---|']
    for r in combined:lines.append(f"|{r['arm']}|{r['success']}|{r['control_steps']}|{r['wam_calls']}|{r['wam_calls_per100']:.2f}|{r['early']}/{r['valid_decisions']}|{r['wall_s']:.1f}|")
    lines+=['\n沿用policy/env seed7、init0、trial0，独立调度RNG seed470048，与原随机25%组相同。不得根据本次成败重选调度seed。监控仍计算真实误差，仅作shadow记录。单次随机轨迹不能证明统计优劣；需按实际调用密度和GPU时间判读。']
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
        runtime.status('SMOKE' if limit!=700 else 'START_EPISODE',arm=arm,seed=7,task_id=151,completed=len(ROWS),total=1)
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
    r=run('random15',700)
    r['random_early_probability']=.15
    r['actions_sha256']=hashlib.sha256(np.asarray(r.pop('actions'),np.float32).tobytes()).hexdigest();r.pop('windows')
    ROWS.append(r);runtime.save('random15/task151/result.json',r);summarize(partial=False)
    runtime.status('COMPLETED',seed=7,task_id=151,completed=1,total=1)
    return dict(successes=int(ROWS[0]['success']),duration=sum(r['wall_s'] for r in ROWS))

def main():
    runtime.driver=driver
    runtime.main()
if __name__=='__main__':
    try:main()
    except BaseException:
        runtime.save('failure.json',dict(traceback=traceback.format_exc()));runtime.status('FAILED');raise
