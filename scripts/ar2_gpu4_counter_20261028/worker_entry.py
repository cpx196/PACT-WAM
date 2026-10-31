"""Drop administrator identity before running the experiment; inherit Nsight injection."""
import os,sys,json,pathlib,pwd
HERE=pathlib.Path(__file__).resolve().parent;ROOT=pathlib.Path('/data/pxchen/PACT-WAM');BASE=ROOT/'evaluate_results/white_mug_robot42_seed7_sameT5_20261025';SRC=HERE/'worker'
if os.getuid()==0:
 u=pwd.getpwnam('pxchen');os.initgroups(u.pw_name,u.pw_gid);os.setgid(u.pw_gid);os.setuid(u.pw_uid)
assert os.getuid()==pwd.getpwnam('pxchen').pw_uid
out=pathlib.Path(sys.argv[1]);env=os.environ.copy()
for k in list(env):
 if k.startswith('LIBERO_WORKER_') or k in ['ASYNC_TIMING_FORCE_GPU']:env.pop(k)
cache=ROOT/'evaluate_results/ar2_compiled_latency_20261026/compiler_cache'
env.update(HOME='/home/pxchen',USER='pxchen',LOGNAME='pxchen',ASYNC_PRELOAD_CPU='1',TZ='Asia/Shanghai',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',LIBERO_CONFIG_PATH='/data/pxchen/FastWAM/.libero_plus_config',DIFFSYNTH_MODEL_BASE_PATH='/data/pxchen/FastWAM/checkpoints',DIFFSYNTH_SKIP_DOWNLOAD='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',GUIDANCE_SIX_OUT=str(out),ACTION_WEIGHTS_REGISTRY=str(BASE/'weights_registry.json'),TORCHINDUCTOR_CACHE_DIR=str(cache),TRITON_CACHE_DIR=str(cache/'triton'),TORCHINDUCTOR_COMPILE_THREADS='2',TORCH_LOGS='recompiles,graph_breaks',PYTHONPATH=':'.join(map(str,[SRC,ROOT,ROOT/'src',pathlib.Path('/data/pxchen/LIBERO-plus'),ROOT/'artifacts/jepawam_deploy_smoke_20261009/python_deps'])))
env['LD_LIBRARY_PATH']='/data/pxchen/envs/jepa-wam/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
old=json.loads((BASE/'weighted/command.json').read_text());args=[('EVALUATION.output_dir='+str(out)) if a.startswith('EVALUATION.output_dir=') else ('EVALUATION.task_id=559') if a.startswith('EVALUATION.task_id=') else a for a in old[2:]]
cmd=['/data/pxchen/envs/jepa-wam/bin/python',str(SRC/'bench_async_timing.py'),*args]
(out/'command.json').write_text(json.dumps(cmd,indent=2));(out/'worker_uid.json').write_text(json.dumps(dict(uid=os.getuid(),gid=os.getgid(),pid=os.getpid(),identity=dict(pid=os.getpid(),ppid=os.getppid(),start=pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],uid=os.getuid()))))
os.chdir(ROOT);os.execve(cmd[0],cmd,env)
