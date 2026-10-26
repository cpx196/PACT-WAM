"""Bounded, unique nohup launcher; never preempt another GPU process."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/data/pxchen/PACT-WAM')
PYTHON = '/data/pxchen/envs/jepa-wam/bin/python'
SCRIPT = ROOT/'scripts/train_fulltoken_mlp80m_pilot5_20261008.py'


def idle():
    busy = {r[0].strip() for r in csv.reader(subprocess.check_output(
        ['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'], text=True).splitlines()) if r}
    rows = csv.reader(subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'], text=True).splitlines())
    return {r[0].strip() for r in rows if r[1].strip() not in busy and int(r[2]) < 128 and int(r[3]) < 5}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out/'launcher.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (args.out/'config.json').exists() and not (args.out/'summary.json').exists()
    (args.out/'launcher.pid').write_text(str(os.getpid()))
    def status(text):
        (args.out/'status.txt').write_text(text+'\n')
        print(time.strftime('%F %T'), text, flush=True)
    status('CPU_PREFLIGHT')
    subprocess.run([PYTHON,str(SCRIPT),'--out',str(args.out),'--cpu-preflight'], check=True)
    previous = idle(); time.sleep(8); stable = previous & idle()
    if not stable:
        status('NO_STABLE_IDLE_GPU'); return
    gpu = sorted(stable, key=int)[0]
    (args.out/'gpu_assignment.txt').write_text(gpu+'\n')
    status('RUNNING_GPU_'+gpu)
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4')
    with (args.out/'run.log').open('a') as log:
        worker = subprocess.Popen([PYTHON,str(SCRIPT),'--out',str(args.out),'--epochs','5','--batch','32','--max-seconds','1800'],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        (args.out/'worker.pid').write_text(str(worker.pid))
        code = worker.wait()
    status('COMPLETED' if code == 0 and (args.out/'summary.json').exists() else f'FAILED_EXIT_{code}')


if __name__ == '__main__':
    main()
