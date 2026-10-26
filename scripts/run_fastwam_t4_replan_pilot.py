"""One-GPU, one-resident-process launcher. Refuse duplicate outputs or occupied GPU."""
import fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/data/pxchen/PACT-WAM')
OUT=ROOT/'evaluate_results/fastwam_t4_replan_pilot_20261020'
PYTHON='/data/pxchen/envs/jepa-wam/bin/python'
def save(name,data):
    p=OUT/name;q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(data,ensure_ascii=False,indent=2));q.replace(p)
def gpu_ready():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    selected=next(line for line in rows.splitlines() if line.split(',')[0].strip()=='6')
    _,uuid,free=[x.strip() for x in selected.split(',')]
    assert int(free)>=18432,('GPU6 insufficient memory',selected)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    for line in apps.splitlines():
        u,pid,name=[x.strip() for x in line.split(',',2)]
        if u==uuid:
            assert name=='[Not Found]' and not Path('/proc',pid).exists(),('GPU6 occupied',line)
    return dict(gpu=6,free_mib=int(free),process_snapshot=apps)
def main():
    # mkdir is exclusive; neither historical output nor another run may be overwritten.
    OUT.mkdir(parents=True,exist_ok=False)
    lock=(OUT/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (OUT/'launcher.pid').write_text(str(os.getpid()))
    snap=gpu_ready();save('gpu_start.json',snap)
    assets={}
    for name in ('checkpoints/fastwam/libero_optional_idm_2cam224.pt','checkpoints/jepa_wam_assets_20261009/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt','artifacts/fastwam_predictor21_ar2_full10epoch_bs16_20261014/step19690.pt','artifacts/fastwam_predictor21_ar2_stride20_20261014/train_only_stats.json'):
        p=ROOT/name;st=p.stat();assets[name]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    hashes={}
    for name in ('experiments/libero/eval_libero_single.py','scripts/fastwam_ar2_eval.py','scripts/t4_pilot_core.py','scripts/eval_fastwam_t4_replan_pilot.py','scripts/run_fastwam_t4_replan_pilot.py','scripts/analyze_fastwam_t4_replan_pilot.py','artifacts/fastwam_predictor21_ar2_stride20_20261014/train_only_stats.json'):
        hashes[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    save('manifest.json',dict(created=time.strftime('%F %T'),timezone='Asia/Shanghai',gpu=6,seed=47,calibration_tasks=[260,321],test_tasks=[151,242,559,590],initial_state_index=0,trial=0,phase=0,arms=['calibration','fixed8','adaptive48','random48'],max_control_steps=700,settling=30,quantile=.75,random_early_probability=.25,assets=assets,sha256=hashes,action_tolerance=1e-6,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),note='Existing repository changes preserved. No guidance/ranking/AR2/decode/compile. Paired seed47, same init0; failure-selected tasks.'))
    ref=ROOT/'evaluate_results/fastwam_noguidance_task151_ood35_fullvideo_20261019/formal_command.json'
    old=json.loads(ref.read_text());args=[a for a in old[2:] if not a.startswith('MULTIRUN.')]
    replacements={'EVALUATION.output_dir':str(OUT),'EVALUATION.visualize_future_video':'false','seed':'47'}
    args=[a.split('=',1)[0]+'='+replacements[a.split('=',1)[0]] if a.split('=',1)[0] in replacements else a for a in args]
    args.extend(['EVALUATION.task_suite_name=libero_10','EVALUATION.task_id=260','EVALUATION.use_action_ensembler=false'])
    cmd=[PYTHON,'scripts/eval_fastwam_t4_replan_pilot.py',*args];save('command.json',cmd)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='6',MUJOCO_EGL_DEVICE_ID='6',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',LIBERO_CONFIG_PATH='/data/pxchen/FastWAM/.libero_plus_config',DIFFSYNTH_MODEL_BASE_PATH='/data/pxchen/FastWAM/checkpoints',DIFFSYNTH_SKIP_DOWNLOAD='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',T4_PILOT_OUT=str(OUT),PYTHONPATH=':'.join(map(str,[ROOT,ROOT/'src',Path('/data/pxchen/LIBERO-plus'),ROOT/'artifacts/jepawam_deploy_smoke_20261009/python_deps'])))
    env.pop('LIBERO_WORKER_MODE',None)
    gpu_ready()
    with (OUT/'run.log').open('w') as log:
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        save('active_process.json',dict(pid=child.pid,launcher_pid=os.getpid(),gpu=6,started=time.strftime('%F %T')))
        rc=child.wait()
    assert rc==0,('Pilot worker failed',rc)
if __name__=='__main__':
    try:main()
    except BaseException:
        if OUT.exists() and not isinstance(sys.exc_info()[1],FileExistsError):save('launcher_failure.json',dict(traceback=traceback.format_exc()))
        raise
