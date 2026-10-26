"""Seed7 recalibrated A48 additional calibration-source tasks, frozen threshold, no guidance."""
import hashlib,json,os,time,traceback
import numpy as np
import t4_seed7_recal_calib2_runtime as runtime
from pathlib import Path
OUT=runtime.OUT
SOURCE=runtime.ROOT/'evaluate_results/fastwam_t4_replan_pilot_20261020'
ROWS=[]

TASKS=(260,321)
PREVIOUS=runtime.ROOT/'evaluate_results/fastwam_t4_seed7_task151_20261020'

def summarize(partial):
    reference_path=runtime.ROOT/'evaluate_results/fastwam_t4_seed7_remaining5_20261020/summary.json'
    reference=json.loads(reference_path.read_text())['combined_six_task_rollouts']
    threshold=json.loads((OUT/'threshold.json').read_text())
    already=json.loads((runtime.ROOT/'evaluate_results/fastwam_t4_seed7_recalibrated_a48_20261020/summary.json').read_text())['rollouts']
    combined=already+ROWS
    s=dict(partial=partial,seed=7,test_tasks=list(TASKS),calibration_tasks=[260,321],tau=threshold['tau'],old_tau=.7292823195457458,rollouts=ROWS,combined_recalibrated_rollouts=combined,paired_results=[],reference=str(reference_path),note='Recalibrated from176 seed7 B8 shadow windows, trajectory-equally-weighted75th percentile. All six shown;260/321 are calibration-source cases, not independent tests. Previously inspected test cases; diagnostic rerun, not fresh held-out evidence.')
    lines=['# seed7重校准后的A48验证',f"\n阈值{threshold['tau']:.12f}；原四条已完成，追加校准任务260/321进度{len(ROWS)}/2。",'\n|Task|B8|旧阈值A48|新阈值A48|随机25%|','|---|---|---|---|---|']
    for tid in (151,242,260,321,559,590):
        prior={r['arm']:r for r in reference if r['task_id']==tid}
        current=next((r for r in combined if r['task_id']==tid),None)
        def cell(r):
            return ('成功' if r['success'] else '失败')+f"；{r['control_steps']}步/{r['wam_calls']}次WAM" if r else '待完成'
        lines.append('|'+str(tid)+'|'+'|'.join(cell(r) for r in (prior['fixed8'],prior['adaptive48'],current,prior['random48']))+'|')
        if current:
            old=prior['adaptive48']
            s['paired_results'].append(dict(task_id=tid,old_success=old['success'],new_success=current['success'],old_calls=old['wam_calls'],new_calls=current['wam_calls'],same_action_hash=old['actions_sha256']==current['actions_sha256'],old_early=old['early'],new_early=current['early']))
    runtime.save('summary.json',s);runtime.save('partial_results.json',ROWS)
    lines+=['\n新阈值已在启动新A48前冻结，旧结果全部保留。没有按测试成败挑选阈值；但这些测试任务此前已经查看过，结果用于校准敏感性诊断。']
    (OUT/'report_zh.md').write_text('\n'.join(lines))

def driver(ns,**kw):
    runtime.ROWS=ROWS
    runtime.NS=ns;cfg=kw['cfg'];model=kw['model'];processor=kw['processor']
    assert int(cfg.seed)==7 and not cfg.EVALUATION.visualize_future_video
    assert not cfg.EVALUATION.vjepa2_ac.enabled and not cfg.EVALUATION.vjepa2_ac.guidance.enabled
    runtime.save('runtime_config.json',ns['OmegaConf'].to_container(cfg,resolve=True))
    tau=json.loads((OUT/'threshold.json').read_text())['tau']
    assert tau==0.729600727558136
    os.environ['T4_TAU']=str(tau)
    bench=ns['benchmark'].get_benchmark_dict()['libero_10'](task_order_index=0)
    runtime.MONITOR=runtime.Monitor(kw['model_device'])
    def run(tid,arm,limit,monitor=True,native=False):
        os.environ['T4_ARM']=arm;os.environ['T4_MONITOR']='1' if monitor else '0'
        ns['set_global_seed'](7,get_worker_init_fn=False)
        cfg.EVALUATION.task_id=tid;cfg.EVALUATION.max_steps_override=limit
        env,description=ns['get_libero_env'](bench.get_task(tid),ns['LIBERO_ENV_RESOLUTION'],7)
        runtime.status('SMOKE' if limit!=700 else 'START_EPISODE',arm=arm,seed=7,task_id=tid,completed=len(ROWS),total=2)
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
    native=run(260,'fixed8',24,False,True)
    off=run(260,'fixed8',24,False)
    on=run(260,'fixed8',24,True)
    assert np.array_equal(native['actions'],off['actions']) and native['rng']==off['rng']
    assert np.array_equal(off['actions'],on['actions']) and off['rng']==on['rng']
    runtime.save('smoke_summary.json',dict(passed=True,seed=7,task_id=260,native_actions_equal=True,native_rng_equal=True,shadow_actions_equal=True,shadow_rng_equal=True,phase=0,parent_interface_smoke=str(SOURCE/'smoke_summary.json')))
    (OUT/'windows.jsonl').rename(OUT/'smoke_windows.jsonl');runtime.WINDOWS.clear()
    summarize(partial=True)
    for arm in ('adaptive48',):
        for tid in TASKS:
            r=run(tid,arm,700)
            r['tau']=tau
            r['actions_sha256']=hashlib.sha256(np.asarray(r.pop('actions'),np.float32).tobytes()).hexdigest();r.pop('windows')
            ROWS.append(r);runtime.save(f'{arm}/task{tid}/result.json',r);summarize(partial=len(ROWS)<2)
            runtime.status('EPISODE_COMPLETED',arm=arm,success=r['success'],seed=7,task_id=tid,completed=len(ROWS),total=2)
    runtime.status('COMPLETED',seed=7,task_ids=list(TASKS),completed=2,total=2)
    return dict(successes=int(ROWS[0]['success']),duration=sum(r['wall_s'] for r in ROWS))

def main():
    runtime.driver=driver
    runtime.main()
if __name__=='__main__':
    try:main()
    except BaseException:
        runtime.save('failure.json',dict(traceback=traceback.format_exc()));runtime.status('FAILED');raise
