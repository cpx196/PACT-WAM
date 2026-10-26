"""Unique nohup waiter: configurable GPU count, smoke-gated 200-task pool."""
import os,sys,time,json,subprocess,fcntl,argparse
from pathlib import Path
from omegaconf import OmegaConf
ROOT=Path('/data/pxchen/PACT-WAM')
OUT=ROOT/'evaluate_results/fastwam_noguidance_long200_fullvideo_20261015'
parser=argparse.ArgumentParser()
parser.add_argument('--num-gpus',type=int,choices=[1,2],default=2)
parser.add_argument('--allow-orphan-memory-gpu',default=None)
args=parser.parse_args()
OUT.mkdir(parents=True,exist_ok=True)
lock=open(OUT/'launcher.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OUT/'formal').exists() and not (OUT/'smoke').exists(),'Refuse duplicate run'
(OUT/'launcher.pid').write_text(str(os.getpid()))
def status(s):
    (OUT/'status.txt').write_text(s+'\n');print(time.strftime('%F %T')+' '+s,flush=True)
def idle():
    busy=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).split())
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
    free=[]
    for row in rows:
        idx,uuid,mem,util=[v.strip() for v in row.split(',')]
        if uuid not in busy and int(mem)<128 and int(util)<5:free.append(idx)
        elif idx==args.allow_orphan_memory_gpu and int(mem)<5500 and int(util)<5:
            # Only allow known dead CUDA contexts, never a live compute job.
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True).splitlines()
            matching=[a.split(',') for a in apps if a.split(',')[0].strip()==uuid]
            if matching and all(a[2].strip()=='[Not Found]' and not Path('/proc/'+a[1].strip()).exists() for a in matching):
                free.append(idx)
    return free
def wait_gpus():
    start={}
    while True:
        now=time.monotonic();free=idle();start={k:start.get(k,now) for k in free}
        ready=[k for k in free if now-start[k]>=60]
        if len(ready)>=args.num_gpus:return ready[:args.num_gpus]
        time.sleep(20)
def run(gpus,smoke=False):
    old=OmegaConf.load(ROOT/'evaluate_results/guidance789_rawgrad4x_long36_full144_libero10_calibrated_step2000_20261007/seed_7/manager_config.yaml')
    fields=['dataset_stats_path','replan_steps','num_inference_steps','compile_action_infer','visualize_future_video','offload_text_encoder','action_infer_mode','video_sigma_shift','action_sigma_shift','sigma_shift']
    overrides=[f'EVALUATION.{k}='+json.dumps(old.EVALUATION[k]) for k in fields]
    overrides+=['EVALUATION.num_steps_wait=30','EVALUATION.trigger_probe.enabled=false','EVALUATION.vjepa2_ac.enabled=false',
        'EVALUATION.vjepa2_ac.guidance.enabled=false','EVALUATION.vjepa2_ac.guidance.trigger.enabled=false',
        'EVALUATION.vjepa2_ac.candidate_generation_mode=separate','EVALUATION.vjepa2_ac.camera.preset=libero_default',
        'EVALUATION.vjepa2_ac.camera.image_rotation_degrees=180','++EVALUATION.timing_enabled=true',
        'model.load_text_encoder=true','model.redirect_common_files=false','model.skip_dit_load_from_pretrain=true','seed=7']
    dest=OUT/('smoke' if smoke else 'formal');dest.mkdir()
    cmd=[sys.executable,'scripts/fastwam_fullvideo200_manager.py','task=libero_optional_idm_2cam224_1e-4',
        'ckpt='+str(ROOT/'checkpoints/fastwam/libero_optional_idm_2cam224.pt'),
        'EVALUATION.output_dir='+str(dest),'MULTIRUN.task_file='+str(OUT/('smoke_task.txt' if smoke else 'tasks200.txt')),
        'MULTIRUN.num_gpus='+str(len(gpus)),'EVALUATION.num_trials=1',
        'EVALUATION.max_steps_override='+('48' if smoke else 'null'),*overrides]
    (dest/'command.json').write_text(json.dumps(cmd,indent=2));env=os.environ.copy()
    env['CUDA_VISIBLE_DEVICES']=','.join(gpus);env['MUJOCO_EGL_DEVICE_ID']=gpus[0]
    with (dest/'run.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
def verify_smoke():
    records=list((OUT/'smoke/libero_10/full_predictions').glob('*/index.json'));assert len(records)==1
    d=json.loads(records[0].read_text());assert not d['guidance'] and len(d['replans'])>=1
    import imageio
    for r in d['replans']:
        assert r['prediction_frame_count']==9
        reader=imageio.get_reader(str(records[0].parent/r['file']));n=reader.count_frames();reader.close();assert n==9,n
    report=dict(passed=True,replans=len(d['replans']),prediction_frames_each=9,untruncated_horizon_steps=32,
        video_bytes=d['video_bytes'],encoding_seconds=d['encoding_seconds'],guidance=False)
    (OUT/'smoke_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
try:
    subprocess.run([sys.executable,'scripts/prepare_fullvideo200.py'],cwd=ROOT,check=True)
    status(f'WAITING_{args.num_gpus}_AVAILABLE_GPUS; stable >=60s; orphan-memory exception='+str(args.allow_orphan_memory_gpu))
    pair=wait_gpus();(OUT/'gpu_assignment.json').write_text(json.dumps(dict(smoke_gpu=pair[0],initial_gpus=pair,num_gpus=args.num_gpus)))
    status('SMOKE_RUNNING GPU'+pair[0]);run(pair[:1],True);verify_smoke()
    status(f'WAITING_{args.num_gpus}_AVAILABLE_GPUS_FOR_FORMAL')
    pair=wait_gpus();(OUT/'gpu_assignment.json').write_text(json.dumps(dict(formal_gpus=pair,num_gpus=args.num_gpus)))
    status('FULL200_RUNNING GPUs='+','.join(pair));run(pair)
    results=list((OUT/'formal/libero_10').glob('gpu*_task*_results.json'));assert len(results)==200,len(results)
    status('COMPLETED')
except BaseException as e:
    status('FAILED '+repr(e));raise
