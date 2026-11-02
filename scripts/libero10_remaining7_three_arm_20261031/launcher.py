import os,json,time,hashlib,subprocess,fcntl,traceback,ctypes,ctypes.util
from pathlib import Path
R=Path('/data/pxchen/PACT-WAM');OUT=R/'evaluate_results/libero10_remaining7_robotpose_three_arm_seed7_20261031';SRC=OUT/'source_snapshot';PY='/data/pxchen/envs/jepa-wam/bin/python'
def save(name,x):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def available():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
 busy=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).split())
 return [v for line in rows for v in [[s.strip() for s in line.split(',')]] if v[1] not in busy and int(v[2])>=22500 and int(v[3])<10]
def egl_index(physical):
 l=ctypes.CDLL(ctypes.util.find_library('EGL'));l.eglGetProcAddress.restype=ctypes.c_void_p
 q=ctypes.CFUNCTYPE(ctypes.c_uint,ctypes.c_int,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int))(l.eglGetProcAddress(b'eglQueryDevicesEXT'))
 a=ctypes.CFUNCTYPE(ctypes.c_uint,ctypes.c_void_p,ctypes.c_int,ctypes.POINTER(ctypes.c_ssize_t))(l.eglGetProcAddress(b'eglQueryDeviceAttribEXT'))
 ds=(ctypes.c_void_p*32)();n=ctypes.c_int();assert q(32,ds,ctypes.byref(n))
 cuda=ctypes.CDLL('libcuda.so.1');assert cuda.cuInit(0)==0
 target=subprocess.check_output(['nvidia-smi','--id='+physical,'--query-gpu=pci.bus_id','--format=csv,noheader'],text=True).strip().lower()[-10:]
 for i in range(n.value):
  v=ctypes.c_ssize_t(-1)
  if not a(ds[i],0x323A,ctypes.byref(v)):continue
  d=ctypes.c_int();assert cuda.cuDeviceGet(ctypes.byref(d),int(v.value))==0
  b=ctypes.create_string_buffer(64);assert cuda.cuDeviceGetPCIBusId(b,64,d)==0
  if b.value.decode().lower()[-10:]==target:return i
 raise RuntimeError('No matching NVIDIA EGL device')
def main():
 lock=(OUT/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 save('launcher_identity.json',dict(pid=os.getpid(),time=time.strftime('%F %T')))
 manifest=json.loads((OUT/'manifest.json').read_text());assert manifest['total_episodes']==813
 while True:
  free=available()
  if free:gpu=next((g for g in free if g[0]=='2'),free[0]);break
  save('progress.json',dict(state='WAITING_GPU',time=time.strftime('%F %T')));time.sleep(5)
 egl=egl_index(gpu[0]);save('device.json',dict(physical_gpu=int(gpu[0]),uuid=gpu[1],egl_device_index=egl))
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu[1],PHYSICAL_GPU=gpu[0],MUJOCO_EGL_DEVICE_ID=str(egl),TZ='Asia/Shanghai',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',LIBERO_CONFIG_PATH='/data/pxchen/FastWAM/.libero_plus_config',DIFFSYNTH_MODEL_BASE_PATH='/data/pxchen/FastWAM/checkpoints',DIFFSYNTH_SKIP_DOWNLOAD='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH=':'.join(map(str,[SRC,R,R/'src',Path('/data/pxchen/LIBERO-plus'),R/'artifacts/jepawam_deploy_smoke_20261009/python_deps'])))
 env['LD_LIBRARY_PATH']='/data/pxchen/envs/jepa-wam/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
 for k in list(env):
  if k.startswith('LIBERO_WORKER_'):env.pop(k)
 template=json.loads((R/'evaluate_results/soup_tomato_robot41_seed7_sameT5_20261028/command.json').read_text())
 completed=[]
 for index,item in enumerate(manifest['tasks']):
  taskout=Path(item['output']);taskout.mkdir(exist_ok=True);(taskout/'weighted').mkdir(exist_ok=True)
  while not any(g[1]==gpu[1] for g in available()):
   save('progress.json',dict(state='WAITING_SELECTED_GPU',completed_tasks=len(completed),task=item['base_task'],gpu=int(gpu[0]),time=time.strftime('%F %T')));time.sleep(5)
  (taskout/'release_gpu').touch()
  env.update(ROBOT42_ROOT=str(taskout),SOUP41_RUN_ROOT=str(taskout),ROBOT42_ARM='weighted',GUIDANCE_SIX_OUT=str(taskout/'weighted'),ACTION_WEIGHTS_REGISTRY=str(taskout/'weights_registry.json'))
  changes={'EVALUATION.output_dir':str(taskout/'weighted'),'EVALUATION.task_id':str(item['first_task_id']),'EVALUATION.vjepa2_ac.device':'cpu'}
  args=[a.split('=',1)[0]+'='+changes[a.split('=',1)[0]] if a.split('=',1)[0] in changes else a for a in template[2:]]
  cmd=[PY,str(SRC/'eval_three.py'),*args];(taskout/'command.json').write_text(json.dumps(cmd,indent=2))
  with (taskout/'run.log').open('w') as log:
   child=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
   save('progress.json',dict(state='RUNNING',completed_tasks=len(completed),total_tasks=7,task=item['base_task'],task_index=index+1,worker_pid=child.pid,gpu=int(gpu[0]),time=time.strftime('%F %T')))
   rc=child.wait()
  if rc:raise RuntimeError('Worker failed: '+item['base_task']+' rc='+str(rc))
  assert json.loads((taskout/'experiment_completed.json').read_text())['complete']
  for arm in ['weighted','control','unweighted']:
   s=json.loads((taskout/arm/'summary.json').read_text());assert not s['partial'] and s['completed']==item['variants']
  completed.append(item['base_task']);save('completed_tasks.json',completed)
 save('progress.json',dict(state='COMPLETED',completed_tasks=7,total_tasks=7,total_episodes=813,time=time.strftime('%F %T')))
if __name__=='__main__':
 try:main()
 except Exception:save('failure.json',dict(time=time.strftime('%F %T'),traceback=traceback.format_exc()));raise
