import pathlib,time,json,subprocess,urllib.request,os
OUT=pathlib.Path('/data/pxchen/PACT-WAM/evaluate_results/ar2_gpu4_counter_20261028');(OUT/'monitor.pid').write_text(str(os.getpid()))
with (OUT/'gpu4_background_monitor.jsonl').open('a') as f:
 deadline=time.monotonic()+5400
 while time.monotonic()<deadline:
  row=dict(time=time.strftime('%F %T'),unix=time.time())
  try:
   row['gpu']=subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,clocks.sm','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
   row['health']=urllib.request.urlopen('http://127.0.0.1:8097/health',timeout=2).status
  except Exception as e:row['error']=type(e).__name__
  f.write(json.dumps(row)+'\n');f.flush()
  p=OUT/'controller_status.json'
  if p.exists():
   try:
    if json.loads(p.read_text())['state'] in ['COMPLETED_VLLM_HEALTHY','FAILED']:break
   except Exception:pass
  time.sleep(5)
