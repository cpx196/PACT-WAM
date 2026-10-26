"""Seed7 remaining five tasks: B8/A48/R48, original frozen threshold, no guidance."""
import hashlib,json,os,time,traceback
import numpy as np
import t4_seed7_remaining5_runtime as runtime
from pathlib import Path
OUT=runtime.OUT
SOURCE=runtime.ROOT/'evaluate_results/fastwam_t4_replan_pilot_20261020'
ROWS=[]

TASKS=(242,260,321,559,590)
PREVIOUS=runtime.ROOT/'evaluate_results/fastwam_t4_seed7_task151_20261020'

def summarize(partial):
    previous=json.loads((PREVIOUS/'summary.json').read_text())
    assert not previous['partial'] and len(previous['rollouts'])==3
    allrows=previous['rollouts']+ROWS
    s=dict(partial=partial,seed=7,task_ids=list(TASKS),completed=len(ROWS),total=15,rollouts=ROWS,combined_six_task_rollouts=allrows,previous_task151=str(PREVIOUS),groups={},paired_results=[],note='All six are previously selected cases. Tasks260/321 were used for seed47 threshold calibration, so this extension is diagnostic, not a fresh held-out evaluation. Frozen tau unchanged.')
    for arm in ('fixed8','adaptive48','random48'):
        rs=[r for r in allrows if r['arm']==arm]
        if rs:
            steps=sum(r['control_steps'] for r in rs)
            s['groups'][arm]=dict(n=len(rs),successes=sum(r['success'] for r in rs),wam_calls_per100=100*sum(r['wam_calls'] for r in rs)/steps,wall_s=sum(r['wall_s'] for r in rs),wam_s=sum(r['wam_s'] for r in rs),monitor_s=sum(r['monitor_s'] for r in rs))
    for tid in (151,*TASKS):
        keyed={r['arm']:r for r in allrows if r['task_id']==tid}
        if len(keyed)==3:
            b,a,r=[keyed[k] for k in ('fixed8','adaptive48','random48')]
            s['paired_results'].append(dict(task_id=tid,baseline=b['success'],adaptive=a['success'],random=r['success'],adaptive_rescue=a['success'] and not b['success'],adaptive_harm=b['success'] and not a['success'],random_rescue=r['success'] and not b['success'],random_harm=b['success'] and not r['success']))
    if all(s['groups'].get(k,{}).get('n')==6 for k in ('fixed8','adaptive48','random48')):
        a,r=[s['groups'][k] for k in ('adaptive48','random48')]
        s['call_rate_gap']=abs(a['wam_calls_per100']/r['wam_calls_per100']-1)
        s['stage_time_gap']=abs((a['wam_s']+a['monitor_s'])/(r['wam_s']+r['monitor_s'])-1)
        s['budget_matched']=max(s['call_rate_gap'],s['stage_time_gap'])<=.15
    runtime.save('summary.json',s);runtime.save('partial_results.json',ROWS)
    lines=['# seed7 六任务三组对照',f"\n补跑进度：{len(ROWS)}/15。task151沿用前次已完成的三条。",'\n|Task|B8|A48|R48|','|---|---|---|---|']
    for tid in (151,*TASKS):
        cells=[]
        for arm in ('fixed8','adaptive48','random48'):
            r=next((r for r in allrows if r['task_id']==tid and r['arm']==arm),None)
            cells.append(('成功' if r['success'] else '失败')+f"；{r['control_steps']}步 / {r['wam_calls']}次WAM" if r else '待完成')
        lines.append('|'+str(tid)+'|'+'|'.join(cells)+'|')
    lines+=['\n阈值固定0.7292823195457458，不重校准；seed7，trial0/init0，无guidance。260/321此前用于seed47阈值校准，本次补跑用于完整诊断，不声称独立泛化证据。随机组成功不能单独证明触发机制优劣，必须同时核对预算和视频事件。']
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
    def run(tid,arm,limit,monitor=True,native=False):
        os.environ['T4_ARM']=arm;os.environ['T4_MONITOR']='1' if monitor else '0'
        ns['set_global_seed'](7,get_worker_init_fn=False)
        cfg.EVALUATION.task_id=tid;cfg.EVALUATION.max_steps_override=limit
        env,description=ns['get_libero_env'](bench.get_task(tid),ns['LIBERO_ENV_RESOLUTION'],7)
        runtime.status('SMOKE' if limit!=700 else 'START_EPISODE',arm=arm,seed=7,task_id=tid,completed=len(ROWS),total=15)
        args=dict(env=env,initial_state=bench.get_task_init_states(tid)[0],task_description=description,model=model,processor=processor,cfg=cfg,episode_idx=0,action_horizon=kw['action_horizon'],input_w=kw['input_w'],input_h=kw['input_h'],model_device=kw['model_device'])
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
            r=dict(runtime.episode.last,seed=7,task_id=tid,arm=arm,trial=0,initial_state_index=0,task_description=description)
            io=time.perf_counter()
            if limit==700:
                d=OUT/arm/f'task{tid}'/'videos';d.mkdir(parents=True,exist_ok=True)
                ns['save_rollout_video'](d,outcome[1],f'task{tid}_trial0',success=outcome[0],task_description=description)
            r['video_io_s']=time.perf_counter()-io
            return r
        finally:env.close()
    # Recheck execution and RNG equivalence for this seed before collecting effects.
    native=run(242,'fixed8',24,False,True)
    off=run(242,'fixed8',24,False)
    on=run(242,'fixed8',24,True)
    assert np.array_equal(native['actions'],off['actions']) and native['rng']==off['rng']
    assert np.array_equal(off['actions'],on['actions']) and off['rng']==on['rng']
    runtime.save('smoke_summary.json',dict(passed=True,seed=7,task_id=242,native_actions_equal=True,native_rng_equal=True,shadow_actions_equal=True,shadow_rng_equal=True,phase=0,parent_interface_smoke=str(SOURCE/'smoke_summary.json')))
    (OUT/'windows.jsonl').rename(OUT/'smoke_windows.jsonl');runtime.WINDOWS.clear()
    summarize(partial=True)
    for arm in ('fixed8','adaptive48','random48'):
        for tid in TASKS:
            r=run(tid,arm,700)
            r['actions_sha256']=hashlib.sha256(np.asarray(r.pop('actions'),np.float32).tobytes()).hexdigest();r.pop('windows')
            ROWS.append(r);runtime.save(f'{arm}/task{tid}/result.json',r);summarize(partial=len(ROWS)<15)
            runtime.status('EPISODE_COMPLETED',arm=arm,success=r['success'],seed=7,task_id=tid,completed=len(ROWS),total=15)
    runtime.status('COMPLETED',seed=7,task_ids=list(TASKS),completed=15,total=15)
    return dict(successes=int(ROWS[0]['success']),duration=sum(r['wall_s'] for r in ROWS))

def main():
    runtime.driver=driver
    runtime.main()
if __name__=='__main__':
    try:main()
    except BaseException:
        runtime.save('failure.json',dict(traceback=traceback.format_exc()));runtime.status('FAILED');raise
