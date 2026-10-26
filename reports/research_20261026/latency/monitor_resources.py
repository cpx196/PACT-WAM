import json,subprocess,time,os
from pathlib import Path
p=Path('/data/pxchen/PACT-WAM/evaluate_results/ar2_compiled_latency_20261026')
pid=794343
with (p/'gpu_resource_samples.jsonl').open('a') as f:
 while Path(f'/proc/{pid}').exists():
  r=dict(time=time.strftime('%F %T'),gpu=subprocess.getoutput('nvidia-smi -i 7 --query-gpu=memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits'),processes=subprocess.getoutput('nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader'))
  f.write(json.dumps(r)+'\n');f.flush();time.sleep(5)
