"""Sample device contention and worker CPU activity without synchronizing CUDA."""
import json,os,subprocess,time
from pathlib import Path
def monitor_resources(stop,gpu,attempt):
 out=Path(os.environ.get('ASYNC_TIMING_OUT','/data/pxchen/PACT-WAM/evaluate_results/ar2_async_timing_v6_20261027'))
 ticks=os.sysconf('SC_CLK_TCK');last={};sample=0
 def command(args):return subprocess.check_output(args,text=True,timeout=8).strip()
 def proc_cpu(path,key,now):
  s=Path(path).read_text().rsplit(')',1)[1].split();value=int(s[11])+int(s[12]);prev=last.get(key);last[key]=(now,value)
  return dict(cpu_pct=None if prev is None else 100*(value-prev[1])/ticks/(now-prev[0]),threads=int(s[17]),resident_pages=int(s[21]))
 with (out/'competition_samples.jsonl').open('a') as f:
  while not stop.is_set():
   now=time.monotonic();row=dict(unix_s=time.time(),monotonic_s=now,gpu=int(gpu),attempt=attempt)
   try:
    row['gpu_state']=command(['nvidia-smi','-i',gpu,'--query-gpu=uuid,name,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory','--format=csv,noheader,nounits'])
    row['compute_processes']=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'])
    active=json.loads((out/'active_process.json').read_text());pid=active['pid'];row['worker_pid']=pid
    try:
     row['worker_cpu']=proc_cpu(f'/proc/{pid}/stat','worker',now)
     status=Path(f'/proc/{pid}/status').read_text().splitlines();row['main_context_switches']={s.split(':')[0]:int(s.split(':')[1]) for s in status if 'ctxt_switches:' in s}
     roles_path=out/'thread_roles.json'
     if roles_path.exists():
      roles=json.loads(roles_path.read_text());row['thread_cpu']={role:proc_cpu(f'/proc/{pid}/task/{tid}/stat',role,now) for role,tid in roles.items()}
    except (OSError,KeyError,ValueError) as e:row['worker_cpu_unavailable']=str(e)
    row['host_loadavg']=Path('/proc/loadavg').read_text().strip()
    total=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]));old=last.get('host_cpu');last['host_cpu']=(sum(total),total[3]+total[4])
    row['host_cpu_busy_pct']=None if old is None else 100*(1-((total[3]+total[4])-old[1])/(sum(total)-old[0]))
    if sample%5==0:
     try:row['process_gpu_utilization_pmon']=command(['nvidia-smi','pmon','-i',gpu,'-c','1','-s','um'])
     except subprocess.SubprocessError as e:row['pmon_unavailable']=str(e)
   except Exception as e:row['error']=str(e)
   row['collection_s']=time.monotonic()-now;f.write(json.dumps(row)+'\n');f.flush();sample+=1;stop.wait(max(0.05,1-row['collection_s']))
