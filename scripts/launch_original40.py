"""Detached one-shot launcher; wait for an idle GPU, retain native defaults."""
import fcntl, hashlib, json, os, shutil, subprocess, time, traceback
from pathlib import Path
ROOT=Path('/data/pxchen/PACT-WAM')
OUT=ROOT/'evaluate_results/libero10long_seeds7_17_27_37_replan8_40_sameT5_20261021'
PYTHON='/data/pxchen/envs/jepa-wam/bin/python'

def save(name,value):
    p=OUT/name; q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(value,indent=2));q.replace(p)

def main():
    assert not (OUT/'active_process.json').exists(), 'Do not restart active worker'
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (OUT/'launcher.pid').write_text(str(os.getpid()))
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    files=['scripts/eval_original40.py','scripts/launch_original40.py','experiments/libero/eval_libero_single.py','experiments/libero/text_context_cache.py','configs/sim_libero.yaml']
    hashes={}
    for name in files:
        p=ROOT/name;hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copyfile(p,snapshot/p.name)
    vanilla=Path('/data/pxchen/libero-suite/LIBERO/libero/libero')
    config=OUT/'vanilla_config';config.mkdir(exist_ok=True)
    import yaml
    (config/'config.yaml').write_text(yaml.safe_dump(dict(benchmark_root=str(vanilla),bddl_files=str(vanilla/'bddl_files'),init_states=str(vanilla/'init_files'),assets=str(vanilla/'assets'),datasets='/data/pxchen/libero-suite/LIBERO/datasets')))
    env=os.environ.copy()
    env.update(TZ='Asia/Shanghai',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',LIBERO_CONFIG_PATH=str(config),DIFFSYNTH_MODEL_BASE_PATH='/data/pxchen/FastWAM/checkpoints',DIFFSYNTH_SKIP_DOWNLOAD='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',ORIGINAL40_OUT=str(OUT),ORIGINAL40_NATIVE_SHA256=hashes['experiments/libero/eval_libero_single.py'],PYTHONPATH=':'.join(['/data/pxchen/libero-suite/LIBERO',str(ROOT),str(ROOT/'src'),str(ROOT/'artifacts/jepawam_deploy_smoke_20261009/python_deps')]))
    env['LD_LIBRARY_PATH']='/data/pxchen/envs/jepa-wam/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    for key in list(env):
        if key.startswith('LIBERO_WORKER_'):env.pop(key)
    worker=snapshot/'eval_original40.py'
    with (OUT/'metadata_check.log').open('w') as log:
        subprocess.run([PYTHON,str(worker),'--metadata-only'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    cmd=[PYTHON,str(worker),'task=libero_optional_idm_2cam224_1e-4','ckpt='+str(ROOT/'checkpoints/fastwam/libero_optional_idm_2cam224.pt'),'seed=7','EVALUATION.task_suite_name=libero_10','EVALUATION.task_id=0','EVALUATION.num_trials=1','EVALUATION.replan_steps=8','EVALUATION.output_dir='+str(OUT),'EVALUATION.dataset_stats_path='+str(ROOT/'checkpoints/fastwam/libero_optional_idm_2cam224_dataset_stats.json'),'EVALUATION.offload_text_encoder=true']
    save('command.json',cmd)
    save('manifest.json',dict(name=OUT.name,seeds=[7,17,27,37],order="seed-major",initial_state_index=0,trials_per_task_per_seed=1,total_tasks=10,total_episodes=40,replan_steps=8,guidance=False,benchmark='vanilla LIBERO10 Long; not LIBERO-Plus',source_sha256=hashes,default_overrides='Only dataset/checkpoint/output, seeds7/17/27/37, replan8, trials1, and T5 offloading to persist contexts; retain default compile=true, IDM mode, shifts=1, max steps=700.'))
    save('progress.json',dict(state='WAITING_GPU',updated=time.strftime('%F %T')))
    ready_since={}
    while True:
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
        busy={r.split(',')[0].strip() for r in apps}
        now=time.monotonic();available=[]
        for row in rows:
            idx,uuid,free,util=[s.strip() for s in row.split(',')]
            if uuid not in busy and int(free)>=18432 and int(util)<5:available.append(idx)
        ready_since={i:ready_since.get(i,now) for i in available}
        ready=[i for i in available if now-ready_since[i]>=30]
        save('gpu_wait.json',dict(updated=time.strftime('%F %T'),available=available,required_free_mib=18432))
        if ready:gpu=ready[0];break
        time.sleep(15)
    env.update(CUDA_VISIBLE_DEVICES=gpu,MUJOCO_EGL_DEVICE_ID=gpu)
    save('progress.json',dict(state='LOADING',gpu=int(gpu),updated=time.strftime('%F %T')))
    with (OUT/'run.log').open('w') as log:
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        save('active_process.json',dict(worker_pid=child.pid,launcher_pid=os.getpid(),gpu=int(gpu),started=time.strftime('%F %T')))
        code=child.wait()
    if code:raise RuntimeError(f'Worker exited {code}')
    assert json.loads((OUT/'progress.json').read_text())['state']=='COMPLETED'
    save('launcher_completed.json',dict(returncode=0,completed=time.strftime('%F %T')))

if __name__=='__main__':
    try:main()
    except BaseException:
        if OUT.exists():save('launcher_failure.json',dict(traceback=traceback.format_exc()))
        raise
