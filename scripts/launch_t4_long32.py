"""One-shot nohup launcher: isolated output, pinned sources/cache, wait for free GPU6."""
import fcntl,hashlib,json,os,shutil,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/data/pxchen/PACT-WAM')
OUT=ROOT/'evaluate_results/fastwam_t4_long32_per_task_20261021'
REFERENCE=ROOT/'evaluate_results/t4_seed7_baseline_reference_20261021'
PYTHON='/data/pxchen/envs/jepa-wam/bin/python'
OWNER=False
CHILD=None

def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(p)
def snapshot():
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=20)
    _,uuid,free=next([v.strip() for v in line.split(',')] for line in gpu.splitlines() if line.split(',')[0].strip()=='6')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True,timeout=20)
    live=[];orphans=[]
    for line in apps.splitlines():
        u,pid,name=[v.strip() for v in line.split(',',2)]
        if u!=uuid:continue
        if name=='[Not Found]' and not Path('/proc',pid).exists():orphans.append(line)
        else:live.append(line)
    return dict(gpu=6,free_mib=int(free),live=live,orphan_contexts=orphans,ready=not live and int(free)>=18432)
def main():
    global OWNER,CHILD
    OUT.mkdir(parents=True,exist_ok=False);OWNER=True
    lock=(OUT/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (OUT/'launcher.pid').write_text(str(os.getpid()))
    source=OUT/'source_snapshot';source.mkdir()
    files=['scripts/launch_t4_long32.py','scripts/eval_t4_long32.py','scripts/t4_long32_core.py','scripts/test_long32_controller.py','scripts/t4_per_task_runtime.py','scripts/t4_pilot_core.py','scripts/fastwam_ar2_eval.py','experiments/libero/eval_libero_single.py','experiments/libero/text_context_cache.py','configs/sim_libero.yaml']
    hashes={}
    for name in files:
        p=ROOT/name;hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copyfile(p,source/p.name)
    assert hashes['experiments/libero/eval_libero_single.py']=='4fadd3bcaf56375245892ab78be04aa0cd8d151764f18f802e9554c9808eefcf'
    ref=json.loads((REFERENCE/'manifest.json').read_text());assert ref['text_cache_sha256']=='6102e417e48e5e6d516f18d56d4556fcc09e8562ee7c7f6b6a7d0225e918522a'
    assert hashlib.sha256(Path(ref['text_cache_path']).read_bytes()).hexdigest()==ref['text_cache_sha256']
    for name,digest in ref['source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,('Original baseline changed',name)
    assert shutil.disk_usage(OUT).free>10*1024**3
    manifest=dict(created=time.strftime('%F %T'),timezone='Asia/Shanghai',task_ids=[151,242,260,321,559,590],calibration_seeds=[57,67],test_seed=7,calibration_episodes=12,evaluation_episodes=18,evaluation_arms=['fixed32','adaptive32','random32'],baseline_retained=str(REFERENCE),baseline_rerun=False,trial=0,initial_state_index=0,max_control_steps=700,settling_steps=30,plan_horizon=32,check_interval=4,guidance=False,ranking=False,decode_video=False,gpu=6,threshold_rule='per-task equal-trajectory-weight 75th percentile, offline raw replay; >=16 windows per seed and >=64 total',random_probability_rule='per-task exact finite-horizon expected calls matched to offline error-stream replay calls; independent RNG 701000+task_id',random_budget_warning='Offline proxy budget match only, report actual evaluation call-rate gap; do not retune from seed7 results.',predictor_rule='phase0 t+4 from latest REAL observation/state, original plan next4 commands, repeated through max32',all_parameters_frozen_before_seed7_evaluation=True,source_sha256=hashes,baseline_cache_sha256=ref['text_cache_sha256'],baseline_cache_identity=ref['encoding_provenance'],reproducibility_gate='All six initial states and historical first4 commands exactly equal; repeated full32 action/horizon equivalence; native controller and monitor RNG/action smoke',failure_policy='Stop and preserve diagnostics on reproducibility, finite-value, calibration sample or offline replay mismatch; no retry with adjusted thresholds')
    save('manifest.json',manifest)
    original=json.loads((ROOT/'evaluate_results/fastwam_t4_seed7_task151_20261020/command.json').read_text())
    args=[]
    for a in original[2:]:
        key=a.split('=',1)[0]
        if key=='seed':a='seed=57'
        elif key=='EVALUATION.output_dir':a=key+'='+str(OUT)
        elif key=='EVALUATION.replan_steps':a=key+'=32'
        args.append(a)
    cmd=[PYTHON,str(source/'eval_t4_long32.py'),*args,'EVALUATION.text_encoder_batch_size=1']
    save('command.json',cmd)
    env=os.environ.copy();env.update(TZ='Asia/Shanghai',CUDA_VISIBLE_DEVICES='6',MUJOCO_EGL_DEVICE_ID='6',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',LIBERO_CONFIG_PATH='/data/pxchen/FastWAM/.libero_plus_config',DIFFSYNTH_MODEL_BASE_PATH='/data/pxchen/FastWAM/checkpoints',DIFFSYNTH_SKIP_DOWNLOAD='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',T4_PILOT_OUT=str(OUT),LONG32_NATIVE_SHA256=hashes['experiments/libero/eval_libero_single.py'],LONG32_BASELINE_CACHE_SHA256=ref['text_cache_sha256'],LONG32_BASELINE_CACHE_PATH=ref['text_cache_path'],PYTHONPATH=':'.join(map(str,[source,ROOT,ROOT/'src',Path('/data/pxchen/LIBERO-plus'),ROOT/'artifacts/jepawam_deploy_smoke_20261009/python_deps'])))
    env.pop('LIBERO_WORKER_MODE',None)
    # Run CPU controller tests from the same immutable source snapshot before GPU allocation.
    result=subprocess.run([PYTHON,str(source/'test_long32_controller.py')],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (OUT/'cpu_test.log').write_text(result.stdout);assert result.returncode==0,result.stdout[-2000:]
    save('cpu_tests_passed.json',dict(passed=True,log='cpu_test.log'))
    save('progress.json',dict(state='WAITING_GPU',updated=time.strftime('%F %T'),gpu=6))
    while True:
        snap=snapshot();save('gpu_snapshot.json',snap)
        if snap['ready']:
            time.sleep(2)
            if snapshot()['ready']:break
        time.sleep(10)
    save('progress.json',dict(state='LOADING',updated=time.strftime('%F %T'),gpu=6))
    with (OUT/'run.log').open('w') as log:
        CHILD=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        save('active_process.json',dict(pid=CHILD.pid,launcher_pid=os.getpid(),gpu=6,started=time.strftime('%F %T')))
        code=CHILD.wait()
    assert code==0,('Worker failed',code)
    assert json.loads((OUT/'progress.json').read_text())['state']=='COMPLETED'
    save('launcher_completed.json',dict(returncode=code,completed=time.strftime('%F %T')))
if __name__=='__main__':
    try:main()
    except BaseException:
        if OWNER:
            if CHILD is not None and CHILD.poll() is None:CHILD.terminate()
            save('launcher_failure.json',dict(traceback=traceback.format_exc()))
            # Preserve the worker's more specific progress/failure when present.
            if not (OUT/'failure.json').exists():save('progress.json',dict(state='FAILED',updated=time.strftime('%F %T')))
        raise
