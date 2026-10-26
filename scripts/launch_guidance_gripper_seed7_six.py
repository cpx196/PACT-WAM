"""One-shot nohup launcher: isolated output, pinned sources/cache, wait for free GPU6."""
import fcntl,hashlib,json,os,shutil,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/data/pxchen/PACT-WAM')
OUT=ROOT/'evaluate_results/fastwam_guidance789_ar2_seed7_six_gripperfix_20261021'
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
    files=['scripts/launch_guidance_gripper_seed7_six.py','scripts/eval_guidance_gripper_seed7_six.py','scripts/test_guidance_gripper_consistency.py','scripts/fastwam_ar2_eval.py','experiments/libero/eval_libero_single.py','experiments/libero/text_context_cache.py','src/fastwam/models/wan22/fastwam_idm.py','src/fastwam/models/wan22/schedulers/scheduler_continuous.py','configs/sim_libero.yaml']
    hashes={}
    for name in files:
        p=ROOT/name;hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copyfile(p,source/p.name)
    assert hashes['experiments/libero/eval_libero_single.py']=='25d1940d14035d9416ac26e42fb01d7350f61a8a330a09c48fc12c49b9d93d1b'
    ref=json.loads((REFERENCE/'manifest.json').read_text());assert ref['text_cache_sha256']=='6102e417e48e5e6d516f18d56d4556fcc09e8562ee7c7f6b6a7d0225e918522a'
    assert hashlib.sha256(Path(ref['text_cache_path']).read_bytes()).hexdigest()==ref['text_cache_sha256']
    for name,digest in ref['source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,('Original baseline changed',name)
    assert shutil.disk_usage(OUT).free>10*1024**3
    manifest=dict(created=time.strftime('%F %T'),timezone='Asia/Shanghai',task_ids=[151,242,260,321,559,590],seed=7,trial=0,initial_state_index=0,total_formal_rollouts=6,reuse_old_guidance_episodes=False,baseline_retained=str(REFERENCE),baseline_rerun=False,max_control_steps=700,settling_steps=30,plan_horizon=32,execution_horizon=8,guidance_nodes=[7,8,9],raw_gradient_scale=4,max_delta_rms=.02,gripper_fix=True,ar_gradient_unchanged=True,denoising_unchanged=True,gpu=6,source_sha256=hashes,baseline_cache_sha256=ref['text_cache_sha256'],baseline_cache_identity=ref['encoding_provenance'],gates=['CPU gripper regression','all6 baseline first4 exact','initial state exact','decode on/off full action exact','all6 guided repeat exact','formal initial observations/actions match preflight','every guidance forward exact executable gripper','every guidance backward zero gripper gradient'],failure_policy='Stop and preserve diagnostic files on any gate failure; do not change tuning or relax action equality.')
    save('manifest.json',manifest)
    original=json.loads((ROOT/'evaluate_results/fastwam_guidance789_ar2_epoch10_selected6_20261020/formal_command.json').read_text())
    args=[]
    for a in original[2:]:
        key=a.split('=',1)[0]
        if key.startswith('MULTIRUN.'):continue
        if key=='EVALUATION.output_dir':a=key+'='+str(OUT)
        args.append(a)
    cmd=[PYTHON,str(source/'eval_guidance_gripper_seed7_six.py'),*args,'EVALUATION.text_encoder_batch_size=1','EVALUATION.binarize_gripper=true','EVALUATION.task_suite_name=libero_10','EVALUATION.task_id=151','EVALUATION.use_action_ensembler=false']
    save('command.json',cmd)
    env=os.environ.copy();env.update(TZ='Asia/Shanghai',CUDA_VISIBLE_DEVICES='6',MUJOCO_EGL_DEVICE_ID='6',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',LIBERO_CONFIG_PATH='/data/pxchen/FastWAM/.libero_plus_config',DIFFSYNTH_MODEL_BASE_PATH='/data/pxchen/FastWAM/checkpoints',DIFFSYNTH_SKIP_DOWNLOAD='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',GUIDANCE_SIX_OUT=str(OUT),GUIDANCE_SIX_NATIVE_SHA256=hashes['experiments/libero/eval_libero_single.py'],GUIDANCE_SIX_CACHE_SHA256=ref['text_cache_sha256'],PYTHONPATH=':'.join(map(str,[source,ROOT,ROOT/'src',Path('/data/pxchen/LIBERO-plus'),ROOT/'artifacts/jepawam_deploy_smoke_20261009/python_deps'])))
    env['LD_LIBRARY_PATH']='/data/pxchen/envs/jepa-wam/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    env.pop('LIBERO_WORKER_MODE',None)
    # Run CPU controller tests from the same immutable source snapshot before GPU allocation.
    result=subprocess.run([PYTHON,str(source/'test_guidance_gripper_consistency.py')],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
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
