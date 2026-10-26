"""Paper-style offline task weights then six pinned-input guidance rollouts."""
import fcntl,hashlib,json,os,shutil,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/data/pxchen/PACT-WAM')
OUT=ROOT/'evaluate_results/feedback_weights_6task_seed7_sameT5_20261021'
PYTHON='/data/pxchen/envs/jepa-wam/bin/python'

def save(name,obj):
 p=OUT/name;q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2));q.replace(p)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def idle():
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).split()
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
 return [r[0] for line in rows for r in [[x.strip() for x in line.split(',')]] if r[1] not in apps and int(r[2])>=20000 and int(r[3])<5]

def main():
 OUT.mkdir(exist_ok=False)
 lock=(OUT/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (OUT/'launcher.pid').write_text(str(os.getpid()))
 src=OUT/'source_snapshot';src.mkdir()
 files=['scripts/test_weighted_energy.py','artifacts/predictor2.1ac1009/smoke_joint_predictor.py','artifacts/predictor2.1ac1009/vectorized_ac_attention.py','scripts/calibrate_weighted_six.py','scripts/weighted_ar2_ranker.py','scripts/eval_weighted_six.py','scripts/launch_weighted_six.py','scripts/fastwam_ar2_eval.py','scripts/test_guidance_gripper_consistency.py','experiments/libero/eval_libero_single.py','experiments/libero/text_context_cache.py','src/fastwam/models/wan22/fastwam_idm.py','src/fastwam/models/wan22/schedulers/scheduler_continuous.py','configs/sim_libero.yaml']
 hashes={}
 for name in files:
  p=ROOT/name;hashes[name]=sha(p);shutil.copyfile(p,src/p.name)
 assert hashes['experiments/libero/eval_libero_single.py']=='25d1940d14035d9416ac26e42fb01d7350f61a8a330a09c48fc12c49b9d93d1b'
 ref=ROOT/'evaluate_results/t4_seed7_baseline_reference_20261021'
 manifest=json.loads((ref/'manifest.json').read_text())
 assert sha(manifest['text_cache_path'])==manifest['text_cache_sha256']=='6102e417e48e5e6d516f18d56d4556fcc09e8562ee7c7f6b6a7d0225e918522a'
 for path,digest in manifest['source_sha256'].items():assert sha(path)==digest,path
 env=os.environ.copy()
 env.update(TZ='Asia/Shanghai',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',LIBERO_CONFIG_PATH='/data/pxchen/FastWAM/.libero_plus_config',DIFFSYNTH_MODEL_BASE_PATH='/data/pxchen/FastWAM/checkpoints',DIFFSYNTH_SKIP_DOWNLOAD='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',GUIDANCE_SIX_OUT=str(OUT),GUIDANCE_SIX_NATIVE_SHA256=hashes['experiments/libero/eval_libero_single.py'],GUIDANCE_SIX_CACHE_SHA256=manifest['text_cache_sha256'],ACTION_WEIGHTS_OUT=str(OUT/'offline'),ACTION_WEIGHTS_REGISTRY=str(OUT/'offline/weights_registry.json'),PYTHONPATH=':'.join(map(str,[src,ROOT,ROOT/'src',Path('/data/pxchen/LIBERO-plus'),ROOT/'artifacts/jepawam_deploy_smoke_20261009/python_deps'])))
 env['LD_LIBRARY_PATH']='/data/pxchen/envs/jepa-wam/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
 for key in list(env):
  if key.startswith('LIBERO_WORKER_'):env.pop(key)
 save('manifest.json',dict(case_set='6task_seed7_sameT5',task_ids=[151,242,260,321,559,590],seed=7,trial=0,initial_state_index=0,source_sha256=hashes,beta=1.,offline_states_per_base_task=200,actions_per_state=32,weights_shape=[2,576,1024],loss='Existing L1 retained; paper Eq14/15 per-element weighting only',online_feedback=False,expert_memory=False,gripper_fix=True,baseline_guidance_reference=str(ROOT/'evaluate_results/fastwam_guidance789_ar2_seed7_six_gripperfix_20261021'),t5_sha256=manifest['text_cache_sha256']))
 with (OUT/'cpu_test.log').open('w') as log:
  subprocess.run([PYTHON,str(src/'test_guidance_gripper_consistency.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 with (OUT/'weighted_energy_test.log').open('w') as log:
  subprocess.run([PYTHON,str(src/'test_weighted_energy.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 save('progress.json',dict(state='WAITING_GPU',updated=time.strftime('%F %T')))
 since={}
 while True:
  now=time.monotonic();available=idle();since={g:since.get(g,now) for g in available}
  ready=[g for g in available if now-since[g]>=60]
  save('gpu_wait.json',dict(updated=time.strftime('%F %T'),available=available,required_free_mib=20000,stable_seconds=60))
  if ready:gpu=ready[0];break
  time.sleep(15)
 env.update(CUDA_VISIBLE_DEVICES=gpu,MUJOCO_EGL_DEVICE_ID=gpu)
 for path,digest in hashes.items():assert sha(ROOT/path)==digest,('Source changed while queued',path)
 def run(stage,command,logfile):
  save('progress.json',dict(state=stage,gpu=int(gpu),updated=time.strftime('%F %T')))
  with (OUT/logfile).open('w') as log:
   child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   save('active_process.json',dict(stage=stage,worker_pid=child.pid,launcher_pid=os.getpid(),gpu=int(gpu),started=time.strftime('%F %T')))
   code=child.wait()
  if code:raise RuntimeError((stage,'worker exited',code))
 run('OFFLINE_CALIBRATION',[PYTHON,str(src/'calibrate_weighted_six.py')],'calibration.log')
 registry=json.loads(Path(env['ACTION_WEIGHTS_REGISTRY']).read_text());assert registry['complete'] and len(registry['tasks'])==3
 old=json.loads((ROOT/'evaluate_results/fastwam_guidance789_ar2_epoch10_selected6_20261020/formal_command.json').read_text())
 args=[]
 for arg in old[2:]:
  key=arg.split('=',1)[0]
  if key.startswith('MULTIRUN.'):continue
  if key=='EVALUATION.output_dir':arg=key+'='+str(OUT)
  args.append(arg)
 args+=['EVALUATION.text_encoder_batch_size=1','EVALUATION.binarize_gripper=true','EVALUATION.task_suite_name=libero_10','EVALUATION.task_id=151','EVALUATION.use_action_ensembler=false']
 cmd=[PYTHON,str(src/'eval_weighted_six.py'),*args];save('command.json',cmd)
 run('GUIDANCE_PREFLIGHT_AND_SIX_ROLLOUTS',cmd,'run.log')
 assert json.loads((OUT/'progress.json').read_text())['state']=='COMPLETED'
 save('launcher_completed.json',dict(returncode=0,completed=time.strftime('%F %T')))

if __name__=='__main__':
 try:main()
 except BaseException:
  if OUT.exists():save('launcher_failure.json',dict(traceback=traceback.format_exc()))
  raise
