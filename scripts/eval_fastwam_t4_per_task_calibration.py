"""Per-task independent offline calibration (seeds57/67), then seed7 A48."""
import hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
import t4_per_task_runtime as runtime
from t4_pilot_core import calibrate
OUT=runtime.OUT
TASKS=(151,242,260,321,559,590)
CAL_SEEDS=(57,67)
CAL_ROWS=[]
TEST_ROWS=[]
ALL_ROWS=[]

class RecordingMonitor(runtime.Monitor):
    capture=False
    saved_bytes=0
    @runtime.torch.inference_mode()
    def start(self,frame,state,commands):
        cache=super().start(frame,state,commands)
        if self.capture:cache.update(raw_state=state.copy(),raw_commands=commands.copy())
        return cache
    @runtime.torch.inference_mode()
    def finish(self,cache,frame):
        result=super().finish(cache,frame)
        if self.capture:
            folder=self.folder/f'window{self.index:04d}';folder.mkdir(parents=True,exist_ok=False)
            Image.fromarray(cache['frame']).save(folder/'current.png')
            Image.fromarray(frame).save(folder/'future4.png')
            np.savez_compressed(folder/'inputs.npz',state=cache['raw_state'],commands=cache['raw_commands'])
            meta=dict(task_id=self.task_id,seed=self.seed,window_id=self.index,control_start=8*self.index,control_end=8*self.index+4,error_online=result['error'],current='current.png',future='future4.png',inputs='inputs.npz',frame_api='native_corrected_primary_view',phase=0)
            (folder/'window.json').write_text(json.dumps(meta,indent=2))
            self.saved_bytes+=sum(p.stat().st_size for p in folder.iterdir())
            assert self.saved_bytes<=2*1024**3,'Calibration raw cache exceeds2GiB cap'
            self.index+=1
        return result

def summarize(partial):
    old=runtime.ROOT/'evaluate_results/fastwam_t4_seed7_remaining5_20261020/summary.json'
    previous=json.loads(old.read_text())['combined_six_task_rollouts']
    thresholds=json.loads((OUT/'thresholds.json').read_text()) if (OUT/'thresholds.json').exists() else None
    s=dict(partial=partial,calibration_seeds=list(CAL_SEEDS),test_seed=7,task_ids=list(TASKS),calibration_rollouts=CAL_ROWS,test_rollouts=TEST_ROWS,thresholds=thresholds,reference=str(old),note='Task-specific calibration from independent policy/env seeds57/67; same explicit init0. Seed7 outcomes are excluded from threshold fitting. These task identities were already selected and inspected, so this remains a mechanism diagnostic.')
    runtime.save('summary.json',s)
    lines=['# 每任务独立离线校准后seed7重测',f'\n校准轨迹{len(CAL_ROWS)}/12，seed7 A48测试{len(TEST_ROWS)}/6。','\n|Task|任务阈值|B8 seed7|旧全局阈值A48|每任务阈值A48|随机25% seed7|','|---|---|---|---|---|---|']
    def cell(r):return ('成功' if r['success'] else '失败')+f"；{r['control_steps']}步/{r['wam_calls']}次WAM" if r else '待完成'
    for tid in TASKS:
        prior={r['arm']:r for r in previous if r['task_id']==tid}
        current=next((r for r in TEST_ROWS if r['task_id']==tid),None)
        tau=f"{thresholds['by_task'][str(tid)]['tau']:.8f}" if thresholds else '未冻结'
        lines.append('|'+str(tid)+'|'+tau+'|'+'|'.join(cell(r) for r in (prior['fixed8'],prior['adaptive48'],current,prior['random48']))+'|')
    lines+=['\n所有阈值在seed7测试前冻结，使用相同75%分位规则、各校准轨迹权重1/2。校准先保存无损图像/精确命令/状态，再独立离线重算误差；离线与采集时shadow误差需在1e-6以内一致。阈值不按成败调整。不加guidance，replan仍只在4与8之间选择。']
    (OUT/'report_zh.md').write_text('\n'.join(lines))

def driver(ns,**kw):
    runtime.NS=ns;runtime.ROWS=ALL_ROWS
    cfg=kw['cfg'];model=kw['model'];processor=kw['processor']
    assert not cfg.EVALUATION.visualize_future_video and not cfg.EVALUATION.vjepa2_ac.enabled and not cfg.EVALUATION.vjepa2_ac.guidance.enabled
    runtime.save('runtime_config.json',ns['OmegaConf'].to_container(cfg,resolve=True))
    bench=ns['benchmark'].get_benchmark_dict()['libero_10'](task_order_index=0)
    monitor=RecordingMonitor(kw['model_device']);runtime.MONITOR=monitor
    os.environ['T4_TOTAL']='18'
    def run(tid,seed,arm,limit=700,monitor_enabled=True,native=False):
        os.environ.update(T4_POLICY_SEED=str(seed),T4_ARM=arm,T4_MONITOR='1' if monitor_enabled else '0')
        ns['set_global_seed'](seed,get_worker_init_fn=False);cfg.seed=seed
        cfg.EVALUATION.task_id=tid;cfg.EVALUATION.max_steps_override=limit
        env,description=ns['get_libero_env'](bench.get_task(tid),ns['LIBERO_ENV_RESOLUTION'],seed)
        runtime.status('SMOKE' if limit!=700 else 'START_EPISODE',seed=seed,task_id=tid,arm=arm,completed=len(ALL_ROWS),total=18)
        args=dict(env=env,initial_state=bench.get_task_init_states(tid)[0],task_description=description,model=model,processor=processor,cfg=cfg,episode_idx=0,action_horizon=kw['action_horizon'],input_w=kw['input_w'],input_h=kw['input_h'],model_device=kw['model_device'])
        try:
            if native:
                commands=[];original=ns['_predict_action_chunk']
                def record(*a,**k):
                    result=original(*a,**k);commands.extend(result[0][:8].tolist());return result
                ns['_predict_action_chunk']=record
                try:runtime.NATIVE(**args)
                finally:ns['_predict_action_chunk']=original
                return dict(actions=commands[:limit],rng=runtime.rng_hash())
            monitor.seed=seed;monitor.task_id=tid;monitor.index=0
            monitor.folder=OUT/'calibration_raw'/f'task{tid}'/f'seed{seed}'
            result=runtime.episode(**args)
            r=dict(runtime.episode.last,task_id=tid,seed=seed,arm=arm,initial_state_index=0,trial=0,task_description=description)
            start=time.perf_counter()
            if limit==700:
                path=OUT/f'seed{seed}'/arm/f'task{tid}'/'videos';path.mkdir(parents=True,exist_ok=True)
                ns['save_rollout_video'](path,result[1],f'task{tid}_trial0',success=result[0],task_description=description)
            r['video_io_s']=time.perf_counter()-start
            return r
        finally:env.close()
    # Check this runtime copy with the actual calibration seed before collecting data.
    native=run(151,57,'fixed8',24,False,True)
    off=run(151,57,'fixed8',24,False)
    on=run(151,57,'fixed8',24,True)
    assert np.array_equal(native['actions'],off['actions']) and native['rng']==off['rng']
    assert np.array_equal(off['actions'],on['actions']) and off['rng']==on['rng']
    runtime.save('smoke_summary.json',dict(passed=True,seed=57,native_actions_equal=True,native_rng_equal=True,shadow_actions_equal=True,shadow_rng_equal=True))
    (OUT/'windows.jsonl').rename(OUT/'smoke_windows.jsonl');runtime.WINDOWS.clear()
    def store(r,dest,collection):
        r['actions_sha256']=hashlib.sha256(np.asarray(r.pop('actions'),np.float32).tobytes()).hexdigest();r.pop('windows')
        collection.append(r);ALL_ROWS.append(r);runtime.save(dest,r);summarize(True)
        runtime.status('EPISODE_COMPLETED',seed=r['seed'],task_id=r['task_id'],arm=r['arm'],success=r['success'],completed=len(ALL_ROWS),total=18)
    monitor.capture=True
    for tid in TASKS:
        for seed in CAL_SEEDS:
            r=run(tid,seed,'fixed8')
            store(r,f'calibration/task{tid}/seed{seed}/result.json',CAL_ROWS)
    monitor.capture=False
    runtime.status('OFFLINE_REPLAY',completed=12,total=18)
    by_task={};maximum_difference=0.;replayed=0;start=time.perf_counter()
    for tid in TASKS:
        errors=[];counts=[];hashes={}
        for seed in CAL_SEEDS:
            values=[];h=hashlib.sha256()
            for f in sorted((OUT/'calibration_raw'/f'task{tid}'/f'seed{seed}').glob('window*/window.json')):
                meta=json.loads(f.read_text());folder=f.parent
                inputs=np.load(folder/'inputs.npz',allow_pickle=False)
                before=np.asarray(Image.open(folder/'current.png').convert('RGB'))
                after=np.asarray(Image.open(folder/'future4.png').convert('RGB'))
                cache=monitor.start(before,inputs['state'],inputs['commands'])
                error=monitor.finish(cache,after)['error']
                delta=abs(error-meta['error_online']);maximum_difference=max(maximum_difference,delta)
                assert delta<=1e-6,('Offline replay mismatch',tid,seed,meta['window_id'],delta)
                for name in ('window.json','current.png','future4.png','inputs.npz'):h.update((folder/name).read_bytes())
                values.append(error);replayed+=1
                with (OUT/'offline_errors.jsonl').open('a') as output:output.write(json.dumps(dict(task_id=tid,seed=seed,window_id=meta['window_id'],error=error))+'\n')
            errors.append(values);counts.append(len(values));hashes[str(seed)]=h.hexdigest()
        by_task[str(tid)]=dict(tau=calibrate(errors),counts=counts,calibration_seeds=list(CAL_SEEDS),raw_sha256=hashes)
        runtime.status('OFFLINE_TASK_CALIBRATED',task_id=tid,tau=by_task[str(tid)]['tau'],counts=counts,replayed_windows=replayed)
    frozen=dict(state='FROZEN_BEFORE_TEST',frozen=True,freeze_time=time.strftime('%F %T'),quantile=.75,weighting='two trajectories each total weight1/2, inverse weighted ECDF',test_seed=7,calibration_seeds=list(CAL_SEEDS),by_task=by_task,replayed_windows=replayed,maximum_online_offline_error_difference=maximum_difference,offline_replay_wall_s=time.perf_counter()-start,raw_cache_bytes=monitor.saved_bytes)
    runtime.save('thresholds.json',frozen);summarize(True)
    for tid in TASKS:
        tau=by_task[str(tid)]['tau'];os.environ['T4_TAU']=str(tau)
        r=run(tid,7,'adaptive48');r['tau']=tau
        store(r,f'test_seed7/task{tid}/result.json',TEST_ROWS)
    summarize(False);runtime.status('COMPLETED',completed=18,total=18,calibration_rollouts=12,test_rollouts=6)
    return dict(successes=sum(r['success'] for r in TEST_ROWS),duration=sum(r['wall_s'] for r in ALL_ROWS))

def main():runtime.driver=driver;runtime.main()
if __name__=='__main__':
    try:main()
    except BaseException:
        runtime.save('failure.json',dict(traceback=traceback.format_exc()));runtime.status('FAILED');raise
