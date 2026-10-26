"""Frozen cross-OOD sample; immediately claim a usable idle GPU, without touching other jobs."""
import argparse,collections,fcntl,hashlib,json,os,random,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/data/pxchen/PACT-WAM')
OUT=ROOT/'evaluate_results/fastwam_noguidance_task151_ood35_fullvideo_20261019'
REF=ROOT/'evaluate_results/fastwam_noguidance_long200_fullvideo_camera_fixed_20261015'
SOURCE=Path('/data/pxchen/LIBERO-plus/libero/libero/benchmark/task_classification.json')
BASE='LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket'
CATS=['Background Textures','Camera Viewpoints','Robot Initial States','Language Instructions','Light Conditions','Objects Layout','Sensor Noise']
# SSH shells do not inherit the evaluation environment. Pin known existing paths.
python_paths=[ROOT,ROOT/'src',Path('/data/pxchen/LIBERO-plus'),ROOT/'artifacts/jepawam_deploy_smoke_20261009/python_deps']
os.environ['PYTHONPATH']=':'.join(map(str,python_paths))
for p in reversed(python_paths):sys.path.insert(0,str(p))
os.environ['LIBERO_CONFIG_PATH']='/data/pxchen/FastWAM/.libero_plus_config'
os.environ['MUJOCO_GL']='egl';os.environ['PYOPENGL_PLATFORM']='egl'
os.environ['OMP_NUM_THREADS']='4';os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['DIFFSYNTH_MODEL_BASE_PATH']='/data/pxchen/FastWAM/checkpoints'
os.environ['DIFFSYNTH_SKIP_DOWNLOAD']='true'
os.environ['PYTHONUNBUFFERED']='1'
CHILD=None
def save(name,data):
 p=OUT/name;q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(data,ensure_ascii=False,indent=2));q.replace(p)
def status(state,**extra):
 data=dict(state=state,updated=time.strftime('%F %T'),**extra);save('status.json',data)
 print(json.dumps(data,ensure_ascii=False),flush=True)
def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'task_manifest.json').exists():
  rows=json.loads((OUT/'task_manifest.json').read_text())['tasks']
  assert len(rows)==35 and len({r['task_id'] for r in rows})==35
  return rows
 allrows=json.loads(SOURCE.read_text())['libero_10']
 pool=[r for r in allrows if r['name']==BASE or r['name'].startswith(BASE+'_')]
 rng=random.Random(20261019);selected=[]
 for cat in CATS:
  levels=collections.defaultdict(list)
  for r in pool:
   if r['category']==cat:levels[int(r['difficulty_level'])].append(r)
  for values in levels.values():values.sort(key=lambda r:r['id']);rng.shuffle(values)
  # Include every available difficulty, then spread remaining slots across levels.
  # Explicitly retain task151 as an anchor; no other result consulted for selection.
  chosen=[]
  if cat=='Background Textures':
   anchor=next(r for r in pool if r['id']-1==151);chosen.append(anchor)
  while len(chosen)<5:
   eligible=[k for k,v in levels.items() if any(t not in chosen for t in v)]
   level=min(eligible,key=lambda k:(sum(t['difficulty_level']==k for t in chosen),k))
   chosen.append(next(t for t in levels[level] if t not in chosen))
  selected.extend(chosen)
 # Resolve runtime IDs and check all task names before any GPU is used.
 from wand.api import library as _wand_library
 from libero.libero import benchmark
 bench=benchmark.get_benchmark_dict()['libero_10'](task_order_index=0)
 mapping={bench.get_task(i).name:i for i in range(bench.n_tasks)}
 rows=[]
 for t in selected:
  tid=mapping[t['name']];assert tid==t['id']-1
  assert len(bench.get_task_init_states(tid))>0
  rows.append(dict(suite='libero_10',task_id=tid,base_task=BASE,category=t['category'],difficulty_level=t['difficulty_level'],name=t['name'],instruction=bench.get_task(tid).language,seed=7,initial_state_index=0))
 rng.shuffle(rows);rows.sort(key=lambda r:r['task_id']!=151) # first completion is reproducibility anchor
 tasktext=''.join(f'libero_10,{r["task_id"]}\n' for r in rows)
 (OUT/'tasks35.txt').write_text(tasktext)
 summary=dict(episodes=35,seed=7,trials=1,sampling_seed=20261019,base_task=BASE,
  category_counts=dict(collections.Counter(r['category'] for r in rows)),
  difficulty_counts={c:dict(collections.Counter(r['difficulty_level'] for r in rows if r['category']==c)) for c in CATS},
  task_sha256=hashlib.sha256(tasktext.encode()).hexdigest(),selection='5 variants per OOD category; cover available difficulty levels; task151 forced as anchor, all other outcomes not consulted',
  reference_run=str(REF),estimate_after_gpu_start_minutes=[90,130],queue_time_unknown=True,
  settings=dict(guidance=False,jepa=False,trigger=False,replan_steps=8,denoise_steps=10,video_shift=1,action_shift=1,max_steps=700,num_steps_wait=30,compile=False,camera_override=False,rotation_degrees=180))
 save('task_manifest.json',dict(settings=summary,tasks=rows));save('sampling_summary.json',summary)
 return rows
def gpu_snapshot():
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True,timeout=20)
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
 occupied=set();orphaned=set()
 for line in apps.splitlines():
  if not line.strip():continue
  uuid,pid,name=[x.strip() for x in line.split(',',2)]
  if name=='[Not Found]' and not Path('/proc',pid).exists():orphaned.add(uuid)
  else:occupied.add(uuid)
 result=[]
 for line in rows.splitlines():
  idx,uuid,mem,total,util=[x.strip() for x in line.split(',')]
  result.append(dict(gpu=int(idx),memory_mib=int(mem),free_memory_mib=int(total)-int(mem),utilization=int(util),has_live_compute_process=uuid in occupied,has_dead_orphan_context=uuid in orphaned,
    fully_idle=uuid not in occupied and int(total)-int(mem)>=16384))
 return result
def verify_runtime_config():
 from omegaconf import OmegaConf
 cfg=OmegaConf.load(OUT/'formal/manager_config.yaml')
 assert all(cfg.EVALUATION.vjepa2_ac.camera.get(k) is None for k in ['position','quaternion','fovy'])
 assert not cfg.EVALUATION.vjepa2_ac.enabled and not cfg.EVALUATION.vjepa2_ac.guidance.enabled
 assert not cfg.EVALUATION.trigger_probe.enabled
 assert cfg.EVALUATION.replan_steps==8 and cfg.EVALUATION.num_inference_steps==10
 assert cfg.EVALUATION.video_sigma_shift==cfg.EVALUATION.action_sigma_shift==1
 assert cfg.seed==7 and cfg.EVALUATION.max_steps_override==700 and cfg.EVALUATION.num_trials==1
def run(rows):
 global CHILD
 assets=Path(os.environ['DIFFSYNTH_MODEL_BASE_PATH'])
 assert (assets/'Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth').stat().st_size>10_000_000_000
 assert (assets/'Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth').stat().st_size>2_000_000_000
 assert (assets/'Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json').is_file()
 lock=open(OUT/'launcher.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'formal').exists(),'Formal directory exists: refuse duplicate launch'
 (OUT/'launcher.pid').write_text(str(os.getpid()))
 status('WAITING_IDLE_GPU',episodes=len(rows),required_idle_seconds=0,poll_seconds=3,minimum_free_memory_mib=16384,allow_dead_orphan_context=True)
 stable={}
 while True:
  try:snapshot=gpu_snapshot()
  except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as exc:
   print('GPU_QUERY_RETRY '+repr(exc),flush=True);stable={};time.sleep(3);continue
  now=time.monotonic()
  ready=[str(r['gpu']) for r in snapshot if r['fully_idle']]
  stable={i:stable.get(i,now) for i in ready}
  save('gpu_wait_snapshot.json',dict(time=time.strftime('%F %T'),gpus=snapshot,stable_seconds={i:round(now-t) for i,t in stable.items()}))
  candidates=ready
  if candidates:
   gpu=candidates[0]
   if next(r for r in gpu_snapshot() if str(r['gpu'])==gpu)['fully_idle']:break
  time.sleep(3)
 cmd=json.loads((REF/'formal_command.json').read_text())
 replacements={'EVALUATION.output_dir':str(OUT/'formal'),'MULTIRUN.task_file':str(OUT/'tasks35.txt'),'MULTIRUN.num_gpus':'1'}
 cmd=[a.split('=',1)[0]+'='+replacements[a.split('=',1)[0]] if a.split('=',1)[0] in replacements else a for a in cmd]
 cmd[0]=sys.executable
 save('formal_command.json',cmd)
 dest=OUT/'formal';dest.mkdir()
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu;env['MUJOCO_EGL_DEVICE_ID']=gpu
 # Refuse if another process appeared between idle selection and launch.
 assert next(r for r in gpu_snapshot() if str(r['gpu'])==gpu)['fully_idle']
 with (dest/'run.log').open('w') as log:
  proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  CHILD=proc
  save('formal_process.json',dict(pid=proc.pid,gpus=[gpu],started=time.strftime('%F %T')))
  status('STARTING',gpu=gpu,pid=proc.pid)
  verified=False
  while proc.poll() is None:
   results=list((dest/'libero_10').glob('*_results.json'))
   save('progress.json',dict(completed=len(results),total=len(rows),gpu=gpu,pid=proc.pid,updated=time.strftime('%F %T')))
   if results and not verified:
    verify_runtime_config()
    indexes=list((dest/'libero_10/full_predictions').glob('*/index.json'))
    assert indexes,'Completed result missing full prediction media'
    x=json.loads(indexes[0].read_text());assert x['seed']==7 and not x['guidance']
    assert x['replans'] and all(r['prediction_frame_count']==9 for r in x['replans'])
    assert list((dest/'libero_10/videos').glob('*actual.mp4'))
    save('first_episode_media_check.json',dict(passed=True,task_id=x['task_id'],replans=len(x['replans']),camera_override=False))
    status('RUNNING_STABLE',gpu=gpu,pid=proc.pid,completed=len(results));verified=True
   time.sleep(15)
  assert proc.returncode==0,f'manager exited {proc.returncode}'
 verify_runtime_config()
 results=[json.loads(p.read_text()) for p in (dest/'libero_10').glob('*_results.json')]
 assert {r['task_id'] for r in results}=={r['task_id'] for r in rows}
 indexes=list((dest/'libero_10/full_predictions').glob('*/index.json'));assert len(indexes)==len(rows)
 groups={c:[r for r in results if next(t for t in rows if t['task_id']==r['task_id'])['category']==c] for c in CATS}
 save('summary.json',dict(episodes=len(results),successes=sum(r['successes'] for r in results),by_category={c:dict(total=len(rs),successes=sum(r['successes'] for r in rs)) for c,rs in groups.items()}))
 status('COMPLETED',gpu=gpu,episodes=len(results),successes=sum(r['successes'] for r in results))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
 try:
  rows=prepare()
  if (OUT/'failure.json').exists():(OUT/'failure.json').rename(OUT/('preflight_failure_'+time.strftime('%Y%m%d_%H%M%S')+'.json'))
  if args.prepare_only:print(json.dumps(json.loads((OUT/'sampling_summary.json').read_text()),indent=2),flush=True)
  else:run(rows)
 except BaseException:
  if CHILD is not None and CHILD.poll() is None:
   CHILD.terminate()
   try:CHILD.wait(timeout=30)
   except subprocess.TimeoutExpired:print('Own manager did not exit after SIGTERM; inspect its worker state before restarting.',flush=True)
  # A second launcher must not overwrite the live launcher's state.
  if not isinstance(sys.exc_info()[1],BlockingIOError):
   save('failure.json',dict(traceback=traceback.format_exc()));status('FAILED')
  raise
