"""One-shot batch dependency: generate the CPU report after the GPU worker exits."""
import fcntl,json,subprocess,time
from pathlib import Path
OUT=Path('/data/pxchen/PACT-WAM/evaluate_results/ar2_async_timing_v6_20261027')
PY='/data/pxchen/envs/jepa-wam/bin/python'
SCRIPT=Path(__file__).parent/'analyze_timing.py'
def save(state,**fields):
 p=OUT/'postprocess_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(state=state,updated=time.strftime('%F %T'),**fields),indent=2));tmp.replace(p)
lock=(OUT/'postprocess.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
save('WAITING_FOR_GPU_BENCHMARK')
while not (OUT/'launcher_completed.json').exists():
 if (OUT/'launcher_failure.json').exists():save('GPU_BENCHMARK_FAILED');raise SystemExit(1)
 time.sleep(5)
save('ANALYZING')
with (OUT/'postprocess.log').open('w') as log:
 result=subprocess.run([PY,str(SCRIPT),str(OUT)],stdout=log,stderr=subprocess.STDOUT)
save('COMPLETED' if result.returncode==0 else 'ANALYSIS_FAILED',returncode=result.returncode)
raise SystemExit(result.returncode)
