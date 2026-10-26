"""36 paired rollouts, fixed init-state0, three new seeds, existing eval paths."""
import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
import launch_task151_ood35_wait_20261019 as shared
from omegaconf import OmegaConf
ROOT=shared.ROOT
OUT=ROOT/'evaluate_results/fastwam_selected6_paired_seeds17_27_37_20261020'
GREF=ROOT/'evaluate_results/fastwam_guidance789_ar2_epoch10_task151_ood35_20261019'
CREF=ROOT/'evaluate_results/fastwam_noguidance_task151_ood35_fullvideo_20261019'
IDS=[151,242,260,321,559,590]
SEEDS=[17,27,37]
CHILD=None
def save(name,obj):
 p=OUT/name;q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(obj,ensure_ascii=False,indent=2));q.replace(p)
def status(state,**kw):
 d=dict(state=state,updated=time.strftime('%F %T'),**kw);save('status.json',d);print(json.dumps(d),flush=True)
def read_results():
 rows=[]
 for seed in SEEDS:
  for arm in ('guidance','control'):
   for p in (OUT/f'seed_{seed}'/arm/'libero_10').glob('*_results.json'):
    r=json.loads(p.read_text());assert r['task_id'] in IDS and r['total_episodes']==1
    rows.append(dict(seed=seed,arm=arm,task_id=r['task_id'],success=bool(r['successes']),duration_seconds=r['duration'],result_path=str(p)))
 return rows
def verify(dest,seed,arm):
 c=OmegaConf.load(dest/'manager_config.yaml');e=c.EVALUATION;ac=e.vjepa2_ac;g=ac.guidance
 assert c.seed==seed and e.num_trials==1 and e.replan_steps==8 and e.max_steps_override==700
 assert e.num_inference_steps==10 and e.num_steps_wait==30
 assert e.video_sigma_shift==e.action_sigma_shift==1
 assert e.action_infer_mode=='idm' and e.visualize_future_video and not e.compile_action_infer
 assert ac.camera.image_rotation_degrees==180 and all(ac.camera.get(k) is None for k in ('position','quaternion','fovy'))
 assert not g.trigger.enabled and ac.enabled==(arm=='guidance') and g.enabled==(arm=='guidance')
 if arm=='guidance':
  assert list(g.after_flow_steps)==[7,8,9] and g.step_size==4 and not g.normalize_gradient and g.max_delta_rms==.02 and g.ac_steps==2
  assert ac.encoder_image_rotation_degrees==0 and ac.dtype=='float32'
  assert str(ac.checkpoint_path).endswith('fastwam_predictor21_ar2_full10epoch_bs16_20261014/step19690.pt')
def main():
 global CHILD
 OUT.mkdir(parents=True,exist_ok=True)
 lock=(OUT/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not any((OUT/f'seed_{s}').exists() for s in SEEDS),'Refuse duplicate run'
 (OUT/'launcher.pid').write_text(str(os.getpid()))
 (OUT/'tasks6.txt').write_text(''.join(f'libero_10,{i}\n' for i in IDS))
 manifest=dict(tasks=IDS,seeds=SEEDS,arms=['guidance','control'],episodes=36,initial_state_index=0,trials_per_task_seed=1,seed7_excluded=True,gpu=6,seed_semantics='Vary global/policy/environment seed; same explicit task initial-state index0 in both arms. Do not change trial index.',guidance_reference=str(GREF),control_reference=str(CREF),full_prediction_video=True,estimated_hours=[3,4],queue_wait_additional=True)
 save('manifest.json',manifest)
 save('task_list.json',[dict(task_id=t,seed=s,arm=a,initial_state_index=0,trial=0) for s in SEEDS for a in ('guidance','control') for t in IDS])
 for seed in SEEDS:
  for arm in ('guidance','control'):
   dest=OUT/f'seed_{seed}'/arm
   status('WAITING_GPU',gpu=6,seed=seed,arm=arm)
   while True:
    snap=shared.gpu_snapshot();save('gpu_snapshot.json',snap)
    if next(r for r in snap if r['gpu']==6)['fully_idle']:break
    time.sleep(10)
   dest.mkdir(parents=True)
   ref=GREF if arm=='guidance' else CREF
   cmd=json.loads((ref/'formal_command.json').read_text())
   replacements={'EVALUATION.output_dir':str(dest),'MULTIRUN.task_file':str(OUT/'tasks6.txt'),'MULTIRUN.num_gpus':'1','seed':str(seed)}
   cmd=[a.split('=',1)[0]+'='+replacements[a.split('=',1)[0]] if a.split('=',1)[0] in replacements else a for a in cmd]
   cmd[0]=sys.executable;save(f'command_{seed}_{arm}.json',cmd)
   env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='6';env['MUJOCO_EGL_DEVICE_ID']='6'
   assert next(r for r in shared.gpu_snapshot() if r['gpu']==6)['fully_idle']
   with (dest/'run.log').open('w') as log:
    CHILD=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    save('active_process.json',dict(pid=CHILD.pid,gpu=6,seed=seed,arm=arm,started=time.strftime('%F %T')))
    status('RUNNING',gpu=6,seed=seed,arm=arm)
    checked=False
    while CHILD.poll() is None:
     rows=read_results();save('partial_results.json',rows)
     save('progress.json',dict(completed=len(rows),total=36,seed=seed,arm=arm,updated=time.strftime('%F %T')))
     if not checked and any(r['seed']==seed and r['arm']==arm for r in rows):
      verify(dest,seed,arm);status('RUNNING_STABLE',gpu=6,seed=seed,arm=arm,completed=len(rows),total=36);checked=True
     time.sleep(10)
    assert CHILD.returncode==0,dict(seed=seed,arm=arm,returncode=CHILD.returncode)
   verify(dest,seed,arm)
   rows=read_results();current=[r for r in rows if r['seed']==seed and r['arm']==arm]
   assert {r['task_id'] for r in current}==set(IDS)
   indexes=[json.loads(p.read_text()) for p in (dest/'libero_10/full_predictions').glob('*/index.json')]
   assert len(indexes)==6 and all(x['seed']==seed and x['replans'] and all(y['prediction_frame_count']==9 for y in x['replans']) for x in indexes)
   assert len(list((dest/'libero_10/videos').glob('*actual.mp4')))==6
   status('ARM_COMPLETED',seed=seed,arm=arm,completed=len(rows),total=36)
 rows=read_results();assert len(rows)==36
 keyed={(r['seed'],r['task_id'],r['arm']):r for r in rows};assert len(keyed)==36
 pairs=[]
 for s in SEEDS:
  for t in IDS:
   c=keyed[s,t,'control']['success'];g=keyed[s,t,'guidance']['success']
   pairs.append(dict(task_id=t,seed=s,control=c,guidance=g,rescue=g and not c,harm=c and not g))
 save('summary.json',dict(pairs=18,rollouts=36,control_successes=sum(r['control'] for r in pairs),guidance_successes=sum(r['guidance'] for r in pairs),rescue=[r for r in pairs if r['rescue']],harm=[r for r in pairs if r['harm']],by_task=[dict(task_id=t,n=3,control_successes=sum(r['control'] for r in pairs if r['task_id']==t),guidance_successes=sum(r['guidance'] for r in pairs if r['task_id']==t)) for t in IDS],paired_results=pairs,rollout_results=rows))
 save('progress.json',dict(completed=36,total=36,updated=time.strftime('%F %T')));status('COMPLETED',rollouts=36)
if __name__=='__main__':
 try:main()
 except BaseException:
  if CHILD is not None and CHILD.poll() is None:CHILD.terminate()
  OUT.mkdir(parents=True,exist_ok=True);save('failure.json',dict(traceback=traceback.format_exc()));status('FAILED');raise
