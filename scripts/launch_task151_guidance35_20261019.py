"""GPU3+6 paired guidance35; smoke on residual-memory GPU6 before full run."""
import fcntl,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import launch_task151_ood35_wait_20261019 as shared
from omegaconf import OmegaConf
ROOT=shared.ROOT
OUT=ROOT/'evaluate_results/fastwam_guidance789_ar2_epoch10_task151_ood35_20261019'
CONTROL=ROOT/'evaluate_results/fastwam_noguidance_task151_ood35_fullvideo_20261019'
HISTORY=ROOT/'evaluate_results/fastwam_guidance789_ar2_epoch10_full144_20261015'
CHILD=None
def save(name,value):
 p=OUT/name;q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(value,ensure_ascii=False,indent=2));q.replace(p)
def status(s,**kw):
 d=dict(state=s,updated=time.strftime('%F %T'),**kw);save('status.json',d);print(json.dumps(d),flush=True)
def command(dest,smoke):
 old=OmegaConf.load(HISTORY/'seed_7/manager_config.yaml')
 def flat(value,prefix):
  if isinstance(value,dict):
   for k,v in value.items():yield from flat(v,prefix+'.'+k)
  else:yield prefix+'='+json.dumps(value,separators=(',',':'))
 cmd=json.loads((CONTROL/'formal_command.json').read_text())
 cmd=[a for a in cmd if not a.lstrip('+').startswith('EVALUATION.vjepa2_ac.')]
 cmd[0]=sys.executable;cmd[1]='scripts/fastwam_ar2_fullvideo_manager.py'
 replace={'EVALUATION.output_dir':str(dest),'MULTIRUN.task_file':str(OUT/('smoke_task.txt' if smoke else 'tasks35.txt')),'EVALUATION.max_steps_override':'48' if smoke else '700','MULTIRUN.num_gpus':'1' if smoke else '2'}
 cmd=[a.split('=',1)[0]+'='+replace[a.split('=',1)[0]] if a.split('=',1)[0] in replace else a for a in cmd]
 for s in flat(OmegaConf.to_container(old.EVALUATION.vjepa2_ac,resolve=True),'EVALUATION.vjepa2_ac'):
  cmd.append('++'+s if s.startswith('EVALUATION.vjepa2_ac.encoder_image_rotation_degrees=') else s)
 save(('smoke' if smoke else 'formal')+'_command.json',cmd);return cmd
def verify_config(dest,smoke):
 c=OmegaConf.load(dest/'manager_config.yaml');g=c.EVALUATION.vjepa2_ac.guidance
 assert c.EVALUATION.vjepa2_ac.enabled and g.enabled and not g.trigger.enabled
 assert g.step_size==4 and not g.normalize_gradient and g.max_delta_rms==.02
 assert list(g.after_flow_steps)==[7,8,9] and g.ac_steps==2
 assert all(c.EVALUATION.vjepa2_ac.camera.get(k) is None for k in ('position','quaternion','fovy'))
 assert c.EVALUATION.vjepa2_ac.encoder_image_rotation_degrees==0
 assert c.EVALUATION.video_sigma_shift==c.EVALUATION.action_sigma_shift==1
 assert c.EVALUATION.max_steps_override==(48 if smoke else 700) and c.seed==7
 assert c.EVALUATION.replan_steps==8 and c.EVALUATION.num_inference_steps==10
 assert not c.EVALUATION.compile_action_infer and not g.compile_predictor
def execute(gpus,smoke):
 global CHILD
 dest=OUT/('smoke' if smoke else 'formal');dest.mkdir()
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=','.join(gpus);env['MUJOCO_EGL_DEVICE_ID']=gpus[0]
 with (dest/'run.log').open('w') as log:
  CHILD=subprocess.Popen(command(dest,smoke),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  save(('smoke' if smoke else 'formal')+'_process.json',dict(pid=CHILD.pid,gpus=gpus,started=time.strftime('%F %T')))
  stable=False
  while CHILD.poll() is None:
   completed=len(list((dest/'libero_10').glob('*_results.json')))
   save('progress.json',dict(stage='smoke' if smoke else 'formal',completed=completed,total=1 if smoke else 35,gpus=gpus,pid=CHILD.pid,updated=time.strftime('%F %T')))
   if completed and not smoke and not stable:
    verify_config(dest,False);status('RUNNING_STABLE',gpus=gpus,completed=completed,total=35);stable=True
   time.sleep(10)
  assert CHILD.returncode==0,CHILD.returncode
 verify_config(dest,smoke)
 indexes=list((dest/'libero_10/full_predictions').glob('*/index.json'))
 assert len(indexes)==(1 if smoke else 35)
 assert all(len(r['replans'])>0 and all(t['prediction_frame_count']==9 for t in r['replans']) for r in map(lambda p:json.loads(p.read_text()),indexes))
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 lock=open(OUT/'launcher.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'smoke').exists() and not (OUT/'formal').exists(),'Refuse duplicate evaluation'
 rows=json.loads((CONTROL/'task_manifest.json').read_text());assert len(rows['tasks'])==35
 for name in ['task_manifest.json','tasks35.txt']:shutil.copy2(CONTROL/name,OUT/name)
 assert hashlib.sha256((OUT/'tasks35.txt').read_bytes()).digest()==hashlib.sha256((CONTROL/'tasks35.txt').read_bytes()).digest()
 (OUT/'smoke_task.txt').write_text('libero_10,151\n')
 cfg=json.loads((HISTORY/'run_config.json').read_text());cfg.update(seeds=[7],episodes=35,task_source=str(OUT/'tasks35.txt'),max_steps_override=700,gpus=[3,6],smoke_gpu=6,paired_control=str(CONTROL),full_prediction_video=True,common_weights_base=os.environ['DIFFSYNTH_MODEL_BASE_PATH'],downloads_disabled=True,estimate_after_launch_minutes=[60,90])
 save('run_config.json',cfg);(OUT/'launcher.pid').write_text(str(os.getpid()))
 status('SELECTING_IDLE_GPU',required=[3,6])
 while True:
  available=[r['gpu'] for r in shared.gpu_snapshot() if r['fully_idle']]
  if 6 in available:break
  status('WAITING_IDLE_GPU');time.sleep(3)
 save('gpu_assignment.json',dict(smoke=[6],formal=[3,6]));status('SMOKE_RUNNING',gpu=6);execute(['6'],True)
 subprocess.run([sys.executable,str(ROOT/'scripts/verify_calibrated_guidance144_smoke_20261007.py'),'--root',str(OUT)],cwd=ROOT,check=True)
 status('SMOKE_PASSED',gpu=6)
 # Recheck resources between smoke and formal. Never start on a card taken by a live job.
 while not {3,6}.issubset({r['gpu'] for r in shared.gpu_snapshot() if r['fully_idle']}):time.sleep(3)
 status('FORMAL_RUNNING',gpus=[3,6],total=35);execute(['3','6'],False)
 results=[json.loads(p.read_text()) for p in (OUT/'formal/libero_10').glob('*_results.json')]
 baseline={r['task_id']:r for r in map(lambda p:json.loads(p.read_text()),(CONTROL/'formal/libero_10').glob('*_results.json'))}
 assert {r['task_id'] for r in results}=={r['task_id'] for r in rows['tasks']}
 lookup={r['task_id']:r for r in rows['tasks']};rescue=[];harm=[];cats={}
 for r in results:
  tid=r['task_id'];b=baseline[tid]['successes']>0;s=r['successes']>0
  if s and not b:rescue.append(tid)
  if b and not s:harm.append(tid)
  c=cats.setdefault(lookup[tid]['category'],dict(total=0,control_successes=0,guidance_successes=0));c['total']+=1;c['control_successes']+=int(b);c['guidance_successes']+=int(s)
 save('summary.json',dict(episodes=35,successes=sum(r['successes'] for r in results),control_successes=26,rescue=sorted(rescue),harm=sorted(harm),by_category=cats))
 status('COMPLETED',gpus=[3,6],episodes=35,successes=sum(r['successes'] for r in results))
if __name__=='__main__':
 try:main()
 except BaseException:
  if CHILD is not None and CHILD.poll() is None:CHILD.terminate()
  save('failure.json',dict(traceback=traceback.format_exc()));status('FAILED');raise
