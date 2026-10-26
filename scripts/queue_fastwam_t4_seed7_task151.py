"""One-shot queue: wait for parent pilot exit and usable GPU6, then run three episodes."""
import fcntl,hashlib,json,os,shutil,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/data/pxchen/PACT-WAM')
PARENT=ROOT/'evaluate_results/fastwam_t4_replan_pilot_20261020'
OUT=ROOT/'evaluate_results/fastwam_t4_seed7_task151_20261020'
PYTHON='/data/pxchen/envs/jepa-wam/bin/python'
CHILD=None
OWNER=False

def save(name,data):
    p=OUT/name;q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(data,ensure_ascii=False,indent=2));q.replace(p)
def status(state,**kw):
    d=dict(state=state,updated=time.strftime('%F %T'),**kw);save('progress.json',d);print(json.dumps(d,ensure_ascii=False),flush=True)
def parent_active():
    try:
        fd=(PARENT/'launcher.lock').open('a')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return True
    else:fd.close()
    p=json.loads((PARENT/'active_process.json').read_text())
    for pid in (p['pid'],p['launcher_pid']):
        f=Path('/proc',str(pid),'cmdline')
        if f.exists():
            cmd=f.read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if 't4_replan_pilot.py' in cmd:return True
    return False

def gpu_snapshot():
    g=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=20)
    _,uuid,free=next([x.strip() for x in l.split(',')] for l in g.splitlines() if l.split(',')[0].strip()=='6')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True,timeout=20)
    live=[];dead=[]
    for line in apps.splitlines():
        u,pid,name=[x.strip() for x in line.split(',',2)]
        if u!=uuid:continue
        # NVML process lists can miss a live owner: free memory is checked as well.
        if name=='[Not Found]' and not Path('/proc',pid).exists():dead.append(line)
        else:live.append(line)
    return dict(gpu=6,free_mib=int(free),live_processes=live,orphan_contexts=dead,ready=not live and int(free)>=18432)

def main():
    global CHILD,OWNER
    # Exclusive mkdir means a second launcher cannot modify an existing run.
    OUT.mkdir(parents=True,exist_ok=False);OWNER=True
    lock=(OUT/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (OUT/'launcher.pid').write_text(str(os.getpid()))
    threshold=json.loads((PARENT/'threshold.json').read_text());assert threshold['frozen'] and threshold['tau']==0.7292823195457458
    shutil.copyfile(PARENT/'threshold.json',OUT/'threshold.json')
    assert json.loads((PARENT/'smoke_summary.json').read_text())['passed']
    source=OUT/'source_snapshot';source.mkdir()
    hashes={}
    for name in ('queue_fastwam_t4_seed7_task151.py','eval_fastwam_t4_seed7_task151.py','t4_seed7_runtime.py','t4_pilot_core.py','fastwam_ar2_eval.py'):
        p=ROOT/'scripts'/name;hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copyfile(p,source/name)
    native=ROOT/'experiments/libero/eval_libero_single.py'
    assert hashlib.sha256(native.read_bytes()).hexdigest()=='8a24a14bdfdc484a925ca8ad35276a87754a6cf8232c88b0c6f86ea40d2c29bc'
    save('manifest.json',dict(created=time.strftime('%F %T'),timezone='Asia/Shanghai',seed=7,task_ids=[151],arms=['fixed8','adaptive48','random48'],episodes=3,trial=0,initial_state_index=0,max_control_steps=700,settling=30,gpu=6,guidance=False,ranking=False,decode_video=False,phase=0,threshold=threshold['tau'],threshold_source=str(PARENT/'threshold.json'),threshold_source_sha256=hashlib.sha256((PARENT/'threshold.json').read_bytes()).hexdigest(),recalibrate=False,independent_schedule_rng_seed=470048,sha256=hashes,parent=str(PARENT),purpose='Reproduce seen seed7 failure and inspect event-local timing; not fresh generalization evidence.'))
    old=json.loads((PARENT/'command.json').read_text())
    args=[]
    for a in old[2:]:
        key=a.split('=',1)[0]
        if key=='EVALUATION.output_dir':a=key+'='+str(OUT)
        elif key=='seed':a='seed=7'
        elif key=='EVALUATION.task_id':a=key+'=151'
        args.append(a)
    cmd=[PYTHON,'scripts/eval_fastwam_t4_seed7_task151.py',*args];save('command.json',cmd)
    status('WAITING_PARENT_AND_GPU',gpu=6,seed=7,task_id=151,total=3)
    while True:
        try:
            busy=parent_active();snap=gpu_snapshot();save('queue_snapshot.json',dict(parent_active=busy,**snap,updated=time.strftime('%F %T')))
            if not busy and snap['ready']:
                # Require completion of the preceding verified pilot; preserve failures.
                assert json.loads((PARENT/'progress.json').read_text())['state']=='COMPLETED','Parent stopped without completing; inspect its failure before running'
                assert not (PARENT/'failure.json').exists() and not (PARENT/'launcher_failure.json').exists()
                time.sleep(2)
                if not parent_active() and gpu_snapshot()['ready']:break
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as e:
            save('gpu_query_retry.json',dict(error=repr(e),updated=time.strftime('%F %T')))
        time.sleep(10)
    env=os.environ.copy();env.update(TZ='Asia/Shanghai',CUDA_VISIBLE_DEVICES='6',MUJOCO_EGL_DEVICE_ID='6',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',LIBERO_CONFIG_PATH='/data/pxchen/FastWAM/.libero_plus_config',DIFFSYNTH_MODEL_BASE_PATH='/data/pxchen/FastWAM/checkpoints',DIFFSYNTH_SKIP_DOWNLOAD='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',T4_PILOT_OUT=str(OUT),PYTHONPATH=':'.join(map(str,[ROOT,ROOT/'src',Path('/data/pxchen/LIBERO-plus'),ROOT/'artifacts/jepawam_deploy_smoke_20261009/python_deps'])))
    env.pop('LIBERO_WORKER_MODE',None)
    status('LOADING',gpu=6,seed=7,task_id=151,total=3)
    with (OUT/'run.log').open('w') as log:
        CHILD=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        save('active_process.json',dict(pid=CHILD.pid,launcher_pid=os.getpid(),gpu=6,seed=7,started=time.strftime('%F %T')))
        rc=CHILD.wait()
    assert rc==0,('seed7 worker failed',rc)
    assert json.loads((OUT/'progress.json').read_text())['state']=='COMPLETED'
if __name__=='__main__':
    try:main()
    except BaseException:
        if OWNER:
            if CHILD is not None and CHILD.poll() is None:CHILD.terminate()
            save('launcher_failure.json',dict(traceback=traceback.format_exc()));status('FAILED')
        raise
