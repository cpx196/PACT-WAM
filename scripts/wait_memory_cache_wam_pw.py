"""Unique nohup waiter for one-GPU P/W caching; never kills other processes."""
import os,sys,time,json,fcntl,subprocess
from pathlib import Path
ROOT=Path('/data/pxchen/PACT-WAM/artifacts/jepawam_pw_joint20_stride20_full_20261012')
ROOT.mkdir(exist_ok=True)
lock=open(ROOT/'waiter.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(ROOT/'waiter.pid').write_text(str(os.getpid()))
def report(**kw):
    kw['time']=time.time();p=ROOT/'wait_status.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(kw,indent=2));q.replace(p)
last_gpu=None;stable=0;attempt=0
while not (ROOT/'summary.json').exists():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    cards=[tuple(map(int,line.split(','))) for line in output.splitlines() if line.strip()]
    gpu,free=max(cards,key=lambda x:x[1])
    if free>=7168:
        stable=stable+1 if last_gpu==gpu else 1;last_gpu=gpu
    else:stable=0;last_gpu=None
    report(status='waiting_memory',best_gpu=gpu,free_mib=free,required_free_mib=7168,stable_checks=stable)
    if stable<3:time.sleep(15);continue
    # Also avoids duplicates if a separately launched worker already holds its lock.
    probe=open(ROOT/'run.lock','a')
    try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:probe.close();time.sleep(30);continue
    probe.close();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
    report(status='starting',gpu=gpu,free_mib=free)
    attempt+=1
    if (ROOT/'failure.json').exists():
        (ROOT/'failure.json').rename(ROOT/f'failure_before_attempt_{int(time.time())}.json')
    print(json.dumps(dict(event='START_WORKER',gpu=gpu,attempt=attempt,time=time.time())),flush=True)
    with open(ROOT/'run.log','a') as log:
        child=subprocess.Popen(['/data/pxchen/envs/jepa-wam/bin/python','-u','/data/pxchen/PACT-WAM/scripts/cache_wam_pw_full.py'],env=env,stdout=log,stderr=log,stdin=subprocess.DEVNULL)
        report(status='running',gpu=gpu,pid=child.pid)
        code=child.wait()
    if code==0 and (ROOT/'summary.json').exists():report(status='complete');break
    failure=json.load(open(ROOT/'failure.json')) if (ROOT/'failure.json').exists() else {}
    if 'OutOfMemoryError' not in failure.get('traceback',''):
        report(status='failed',exit_code=code,failure=failure);sys.exit(code or 1)
    print(json.dumps(dict(event='OOM_RETRY_WAIT',attempt=attempt)),flush=True)
    last_gpu=None;stable=0;time.sleep(30)
