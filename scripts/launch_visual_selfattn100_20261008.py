"""Launch only on two verified idle GPUs; never preempt any process."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/data/pxchen/PACT-WAM')
OUT=ROOT/'artifacts/libero10_visual_selfattn_pilot100_20261008'
assert not OUT.exists(), 'Existing test: inspect rather than duplicate.'

def idle():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True).splitlines()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader'],text=True)
    occupied={line.split(',')[0].strip() for line in processes.splitlines()}
    return {int(a):b for line in lines for a,b,c,d in [list(map(str.strip,line.split(',')))]
        if int(c)<200 and int(d)==0 and b not in occupied}

first=idle()
time.sleep(8)
second=idle()
assert all(i in first and i in second for i in (0,4)), 'Selected GPU no longer idle.'
OUT.mkdir()
launches=[]
for gpu,variant in [(0,'original'),(4,'visual_selfattn')]:
    dest=OUT/variant
    dest.mkdir()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4')
    with (dest/'run.log').open('ab') as log:
        process=subprocess.Popen(['nohup',sys.executable,str(ROOT/'scripts/pilot_visual_selfattn100_20261008.py'),
            '--out',str(dest),'--variant',variant,'--max-seconds','480'],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    launches.append(dict(gpu=gpu,variant=variant,pid=process.pid,output=str(dest)))
(OUT/'launch.json').write_text(json.dumps(dict(time=time.strftime('%F %T'),jobs=launches),indent=2))
print(json.dumps(launches),flush=True)
