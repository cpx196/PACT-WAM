"""One bounded waiter; no preemption or duplicate pilot launches."""
import csv
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/data/pxchen/PACT-WAM')
p=argparse.ArgumentParser()
p.add_argument('--out',type=Path,default=ROOT/'artifacts/libero10_large_idm_arb_action_only_pilot100_20261007')
p.add_argument('--full-epoch',action='store_true')
args=p.parse_args()
OUT=args.out
OUT.mkdir(parents=True,exist_ok=True)
lock=(OUT/'waiter.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OUT/'metrics.jsonl').exists() and not (OUT/'summary.json').exists()
(OUT/'waiter.pid').write_text(str(os.getpid()))
subprocess.run(['/data/pxchen/envs/jepa-wam/bin/python',str(ROOT/'scripts/pilot_large_idm_arb_action_only100.py'),
                '--out',str(OUT),'--cpu-preflight'],check=True)
deadline=time.monotonic()+12*3600
previous=set()
(OUT/'status.txt').write_text('WAITING_FOR_IDLE_GPU\n')
while time.monotonic()<deadline:
    try:
        busy={r[0].strip() for r in csv.reader(subprocess.check_output([
            'nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()) if r}
        rows=csv.reader(subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines())
        idle={r[0].strip() for r in rows if r[1].strip() not in busy and int(r[2])<128 and int(r[3])<5}
        stable=idle & previous
        if stable:
            gpu=sorted(stable,key=int)[0]
            (OUT/'gpu_assignment.txt').write_text(gpu+'\n')
            (OUT/'status.txt').write_text('RUNNING\n')
            print(time.strftime('%F %T'),'START_GPU',gpu,flush=True)
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
            env['PYTHONPATH']=str(ROOT);env['OMP_NUM_THREADS']='4'
            with (OUT/'run.log').open('a') as log:
                result=subprocess.run(['/data/pxchen/envs/jepa-wam/bin/python',
                    str(ROOT/'scripts/pilot_large_idm_arb_action_only100.py'),'--out',str(OUT)]+(['--full-epoch'] if args.full_epoch else []),
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            (OUT/'status.txt').write_text('COMPLETED\n' if result.returncode==0 else f'FAILED_EXIT_{result.returncode}\n')
            raise SystemExit(result.returncode)
        previous=idle
    except (subprocess.SubprocessError,ValueError) as e:
        print(time.strftime('%F %T'),str(e),flush=True);previous=set()
    time.sleep(30)
(OUT/'status.txt').write_text('WAIT_TIMEOUT_12H\n')
