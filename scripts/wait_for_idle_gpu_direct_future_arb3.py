"""Unique one-hour bounded nohup waiter for authorized three-epoch continuation."""
import argparse
import csv
import fcntl
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/data/pxchen/PACT-WAM')
PYTHON='/data/pxchen/envs/jepa-wam/bin/python'
SCRIPT=ROOT/'scripts/continue_direct_future_arb3.py'
ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
lock=(args.out/'waiter.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (args.out/'metrics.jsonl').exists() and not (args.out/'summary.json').exists()
(args.out/'waiter.pid').write_text(str(os.getpid()))
try:
    subprocess.run([PYTHON,str(SCRIPT),'--out',str(args.out),'--cpu-preflight'],check=True)
    (args.out/'status.txt').write_text('WAITING_FOR_IDLE_GPU\n')
    previous=set(); deadline=time.monotonic()+3600
    while time.monotonic()<deadline:
        busy={r[0].strip() for r in csv.reader(subprocess.check_output([
            'nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()) if r}
        rows=csv.reader(subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines())
        idle={r[0].strip() for r in rows if r[1].strip() not in busy and int(r[2])<128 and int(r[3])<5}
        stable=idle & previous
        if stable:
            gpu=sorted(stable,key=int)[0]
            (args.out/'gpu_assignment.txt').write_text(gpu+'\n')
            (args.out/'status.txt').write_text('RUNNING\n')
            print(time.strftime('%F %T'),'START_GPU',gpu,flush=True)
            env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='4')
            with (args.out/'run.log').open('a') as log:
                code=subprocess.run([PYTHON,str(SCRIPT),'--out',str(args.out)],cwd=ROOT,
                    env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            (args.out/'status.txt').write_text('COMPLETED\n' if code==0 else f'FAILED_EXIT_{code}\n')
            raise SystemExit(code)
        previous=idle; time.sleep(15)
    (args.out/'status.txt').write_text('WAIT_TIMEOUT_1H\n')
except Exception:
    (args.out/'status.txt').write_text('LAUNCHER_FAILED\n'); raise
