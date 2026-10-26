"""Five new matched rollouts plus the already-completed identical task151."""
import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
import launch_task151_ood35_wait_20261019 as shared
from omegaconf import OmegaConf
ROOT=shared.ROOT
OUT=ROOT/'evaluate_results/fastwam_guidance789_ar2_epoch10_selected6_20261020'
REF=ROOT/'evaluate_results/fastwam_guidance789_ar2_epoch10_task151_ood35_20261019'
BASE=ROOT/'evaluate_results/fastwam_noguidance_long200_fullvideo_camera_fixed_20261015'
IDS=[151,242,260,321,559,590]
NEW=IDS[1:]
def save(name,obj):
 p=OUT/name;q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(obj,indent=2,ensure_ascii=False));q.replace(p)
def status(state,**kw):
 d=dict(state=state,updated=time.strftime('%F %T'),**kw);save('status.json',d);print(json.dumps(d),flush=True)
def results(path):
 return {r['task_id']:(r,p) for p in path.glob('*_results.json') for r in [json.loads(p.read_text())]}
def verify(cfg):
 g=cfg.EVALUATION.vjepa2_ac.guidance
 assert cfg.seed==7 and cfg.EVALUATION.num_trials==1 and cfg.EVALUATION.replan_steps==8
 assert cfg.EVALUATION.max_steps_override==700 and cfg.EVALUATION.num_inference_steps==10
 assert cfg.EVALUATION.video_sigma_shift==cfg.EVALUATION.action_sigma_shift==1
 assert list(g.after_flow_steps)==[7,8,9] and g.step_size==4 and g.max_delta_rms==.02
 assert g.ac_steps==2 and not g.normalize_gradient and g.enabled and not g.trigger.enabled
 assert all(cfg.EVALUATION.vjepa2_ac.camera.get(k) is None for k in ('position','quaternion','fovy'))
 assert cfg.EVALUATION.vjepa2_ac.encoder_image_rotation_degrees==0
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 lock=(OUT/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'formal').exists(),'Refuse duplicate launch'
 verify(OmegaConf.load(REF/'formal/manager_config.yaml'))
 baseline=results(BASE/'formal/libero_10');reuse=results(REF/'formal/libero_10')[151]
 assert all(baseline[i][0]['successes']==0 for i in IDS)
 save('manifest.json',dict(seed=7,trial=0,initial_state_index=0,all_tasks=IDS,new_tasks=NEW,reused_task151_result=str(reuse[1]),control_run=str(BASE),guidance_reference=str(REF),gpu=6))
 (OUT/'tasks5.txt').write_text(''.join(f'libero_10,{i}\n' for i in NEW))
 (OUT/'launcher.pid').write_text(str(os.getpid()))
 gpu=next(r for r in shared.gpu_snapshot() if r['gpu']==6)
 save('gpu_assignment.json',gpu);assert gpu['fully_idle'],'GPU6 has a live job or insufficient memory'
 cmd=json.loads((REF/'formal_command.json').read_text())
 replace={'EVALUATION.output_dir':str(OUT/'formal'),'MULTIRUN.task_file':str(OUT/'tasks5.txt'),'MULTIRUN.num_gpus':'1'}
 cmd=[a.split('=',1)[0]+'='+replace[a.split('=',1)[0]] if a.split('=',1)[0] in replace else a for a in cmd]
 save('formal_command.json',cmd)
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='6';env['MUJOCO_EGL_DEVICE_ID']='6'
 dest=OUT/'formal';dest.mkdir()
 assert next(r for r in shared.gpu_snapshot() if r['gpu']==6)['fully_idle']
 with (dest/'run.log').open('w') as log:
  proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  save('process.json',dict(pid=proc.pid,gpu=6));status('STARTING',gpu=6,new_episodes=5,reused_episodes=1)
  checked=False
  while proc.poll() is None:
   rs=results(dest/'libero_10')
   save('progress.json',dict(completed_new=len(rs),total_new=5,reused=1,total=6,updated=time.strftime('%F %T')))
   if rs and not checked:
    verify(OmegaConf.load(dest/'manager_config.yaml'));status('RUNNING_STABLE',completed_new=len(rs),gpu=6);checked=True
   time.sleep(10)
  assert proc.returncode==0,proc.returncode
 verify(OmegaConf.load(dest/'manager_config.yaml'))
 rs=results(dest/'libero_10');assert set(rs)==set(NEW)
 indexes=[json.loads(p.read_text()) for p in (dest/'libero_10/full_predictions').glob('*/index.json')]
 assert len(indexes)==5 and all(x['replans'] and all(r['prediction_frame_count']==9 for r in x['replans']) for x in indexes)
 rs[151]=reuse
 rows=[dict(task_id=i,control_success=False,guidance_success=bool(rs[i][0]['successes']),reused=i==151,result_path=str(rs[i][1]),control_path=str(baseline[i][1])) for i in IDS]
 save('summary.json',dict(episodes=6,new_episodes=5,reused_episodes=1,successes=sum(r['guidance_success'] for r in rows),control_successes=0,rescue=[r['task_id'] for r in rows if r['guidance_success']],harm=[],tasks=rows))
 status('COMPLETED',episodes=6,successes=sum(r['guidance_success'] for r in rows))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(parents=True,exist_ok=True);save('failure.json',dict(traceback=traceback.format_exc()));status('FAILED');raise
