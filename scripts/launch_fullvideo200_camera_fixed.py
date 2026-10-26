"""Exact historical native cameras; success-gated restart, prefer two idle GPUs."""
import os,sys,time,json,subprocess,fcntl,shutil,hashlib,traceback
from pathlib import Path
from omegaconf import OmegaConf
ROOT=Path('/data/pxchen/PACT-WAM')
BAD=ROOT/'evaluate_results/fastwam_noguidance_long200_fullvideo_20261015'
OUT=ROOT/'evaluate_results/fastwam_noguidance_long200_fullvideo_camera_fixed_20261015'
HISTORY=ROOT/'evaluate_results/long800_paired_seed7_20260927/control_noguidance/manager_config.yaml'
OUT.mkdir(parents=True,exist_ok=True)
lock=open(OUT/'launcher.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OUT/'smoke').exists() and not (OUT/'formal').exists(),'Refuse duplicate run'
for name in ['tasks200.txt','task_manifest.json','sampling_summary.json']:
    shutil.copy2(BAD/name,OUT/name)
assert hashlib.sha256((OUT/'tasks200.txt').read_bytes()).hexdigest()=='495c6a87798ae62fee848aae344e85a3a14982f5b13aba239b000d0e95435543'
(OUT/'smoke_task.txt').write_text('libero_10,593\n')
(OUT/'launcher.pid').write_text(str(os.getpid()))
previous=OmegaConf.load(HISTORY)
def save(name,x):
    p=OUT/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def status(s):
    (OUT/'status.txt').write_text(s+'\n');print(time.strftime('%F %T')+' '+s,flush=True)
def available():
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True,timeout=20).splitlines()
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20).splitlines()
    free=[]
    for row in rows:
        idx,uuid,mem,util=[x.strip() for x in row.split(',')]
        apps=[x.split(',') for x in lines if x.split(',')[0].strip()==uuid]
        if not apps and int(mem)<128 and int(util)<5:free.append(idx)
        elif idx=='6' and int(mem)<5500 and int(util)<5 and apps and all(x[2].strip()=='[Not Found]' and not Path('/proc/'+x[1].strip()).exists() for x in apps):free.append(idx)
    return free
def wait(prefer_two=False):
    stable={};start=time.monotonic()
    while True:
        try:ids=available()
        except subprocess.TimeoutExpired:time.sleep(10);continue
        now=time.monotonic();stable={i:stable.get(i,now) for i in ids}
        ready=[i for i in ids if now-stable[i]>=60]
        if len(ready)>=2 and prefer_two:return ready[:2]
        if ready and (not prefer_two or now-start>=120):return ready[:1]
        time.sleep(15)
def launch(gpus,smoke):
    dest=OUT/('smoke' if smoke else 'formal');dest.mkdir()
    keys=['dataset_stats_path','replan_steps','num_inference_steps','compile_action_infer','visualize_future_video','offload_text_encoder','action_infer_mode','video_sigma_shift','action_sigma_shift','sigma_shift']
    fields=[f'EVALUATION.{k}='+json.dumps(previous.EVALUATION[k]) for k in keys]
    fields += ['EVALUATION.num_steps_wait=30','EVALUATION.trigger_probe.enabled=false','EVALUATION.vjepa2_ac.enabled=false',
        'EVALUATION.vjepa2_ac.guidance.enabled=false','EVALUATION.vjepa2_ac.guidance.trigger.enabled=false',
        'EVALUATION.vjepa2_ac.camera.preset=libero_default','EVALUATION.vjepa2_ac.camera.position=null',
        'EVALUATION.vjepa2_ac.camera.quaternion=null','EVALUATION.vjepa2_ac.camera.fovy=null',
        'EVALUATION.vjepa2_ac.camera.image_rotation_degrees=180','++EVALUATION.timing_enabled=true',
        'model.load_text_encoder=true','model.redirect_common_files=false','model.skip_dit_load_from_pretrain=true','seed=7']
    cmd=[sys.executable,'scripts/fastwam_fullvideo200_manager.py','task=libero_optional_idm_2cam224_1e-4',
        'ckpt='+str(ROOT/'checkpoints/fastwam/libero_optional_idm_2cam224.pt'),'EVALUATION.output_dir='+str(dest),
        'MULTIRUN.task_file='+str(OUT/('smoke_task.txt' if smoke else 'tasks200.txt')),
        'MULTIRUN.num_gpus='+str(len(gpus)),'EVALUATION.num_trials=1','EVALUATION.max_steps_override=700',*fields]
    save(('smoke' if smoke else 'formal')+'_command.json',cmd)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=','.join(gpus);env['MUJOCO_EGL_DEVICE_ID']=gpus[0]
    with (dest/'run.log').open('w') as log:
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        save(('smoke' if smoke else 'formal')+'_process.json',dict(pid=process.pid,gpus=gpus))
        rc=process.wait();assert rc==0,rc
def verify(smoke):
    dest=OUT/('smoke' if smoke else 'formal')
    cfg=OmegaConf.load(dest/'manager_config.yaml')
    assert all(cfg.EVALUATION.vjepa2_ac.camera.get(k) is None for k in ['position','quaternion','fovy'])
    results=[json.loads(p.read_text()) for p in (dest/'libero_10').glob('*_results.json')]
    if smoke:
        assert len(results)==1 and results[0]['task_id']==593
        assert results[0]['successes']==1,'Historical-success smoke failed; do not launch formal'
    else:assert len(results)==200
    indexes=list((dest/'libero_10/full_predictions').glob('*/index.json'))
    assert len(indexes)==len(results)
    import imageio.v2 as imageio
    for p in indexes:
        item=json.loads(p.read_text());assert all(x['prediction_frame_count']==9 for x in item['replans'])
        q=item['replans'][0];reader=imageio.get_reader(p.parent/q['file']);assert reader.count_frames()==9;reader.close()
    save(('smoke' if smoke else 'formal')+'_check.json',dict(passed=True,episodes=len(results),successes=sum(d['successes'] for d in results),camera_overrides=False,prediction_frames=9))
try:
    status('WAITING_SMOKE_IDLE_GPU');ids=wait();save('gpu_assignment.json',dict(smoke=ids))
    status('SMOKE_RUNNING GPU'+','.join(ids));launch(ids,True);verify(True)
    status('SMOKE_PASSED_WAITING_FORMAL_PREFER_TWO');ids=wait(True);save('gpu_assignment.json',dict(formal=ids))
    status('FULL200_RUNNING GPUs='+','.join(ids));launch(ids,False);verify(False);status('COMPLETED')
except BaseException:
    status('FAILED');save('failure.json',dict(traceback=traceback.format_exc()));raise
