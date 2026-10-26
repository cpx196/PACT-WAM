"""Small single-GPU no-guidance cross-OOD comparison, original native cameras."""
import os,json,time,runpy,subprocess,fcntl,traceback
from pathlib import Path
from omegaconf import OmegaConf
ROOT=Path('/data/pxchen/PACT-WAM')
OUT=ROOT/'evaluate_results/fastwam_noguidance_soup_tomato_ood21_seed7_20261016'
OLD=ROOT/'evaluate_results/fastwam_noguidance_long200_fullvideo_camera_fixed_20261015'
BASE='LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket'
PLAN={
 'Background Textures':[151,153,155],
 'Robot Initial States':[299,314,316],
 'Camera Viewpoints':[682,690,691],
 'Language Instructions':[1117,1122,1127],
 'Sensor Noise':[1486,1505,1506],
 'Objects Layout':[2060,2066,2068],
 'Light Conditions':[2403,2404,2402],
}
def dump(name,x):
 p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def status(s):
 (OUT/'status.txt').write_text(s+'\n');print(time.strftime('%F %T'),s,flush=True)
def prepare():
 OUT.mkdir(exist_ok=True)
 db=Path('/data/pxchen/LIBERO-plus/libero/libero/benchmark')
 names=runpy.run_path(str(db/'libero_suite_task_map.py'))['libero_task_map']['libero_10']
 cls={r['name']:r for r in json.loads((db/'task_classification.json').read_text())['libero_10']}
 rows=[]
 for slot in range(3):
  for cat,ids in PLAN.items():
   tid=ids[slot];name=names[tid];r=cls[name]
   assert name.startswith(BASE+'_') and r['category']==cat
   assert r['difficulty_level']==3 or (tid==2402 and r['difficulty_level']==4)
   rows.append(dict(suite='libero_10',task_id=tid,seed=7,initial_state_index=0,base_task=BASE,name=name,category=cat,difficulty_level=r['difficulty_level'],variant=name[len(BASE):]))
 assert len(rows)==21 and len({r['task_id'] for r in rows})==21
 dump('task_manifest.json',dict(tasks=rows,selection='3 variants/category, difficulty3 when available, light third difficulty4; deterministic ID selection without using outcomes',guidance=False,seed=7))
 (OUT/'tasks21.txt').write_text(''.join(f'libero_10,{r["task_id"]}\n' for r in rows))
 cmd=json.loads((OLD/'formal_command.json').read_text())
 replacement={'EVALUATION.output_dir':str(OUT/'formal'),'MULTIRUN.task_file':str(OUT/'tasks21.txt'),'MULTIRUN.num_gpus':'1'}
 cmd=[next((f'{k}={v}' for k,v in replacement.items() if arg.startswith(k+'=')),arg) for arg in cmd]
 dump('command.json',cmd)
 (OUT/'EXPERIMENT.md').write_text('# 单基础任务跨OOD小规模测试\n\n任务：把字母汤罐与番茄酱罐放入篮子。7类OOD各3个变体，共21条；seed7，初始状态索引0，每变体1次。光照仅有两个难度3变体，因此第三个取难度4。其余全部难度3。\n\n全部沿用相机修正版200条的无guidance配置：optional-IDM FastWAM权重，BF16，10步denoise，replan8，video/action shift=1/1，wait30，max700，不编译。JEPA、guidance、trigger关闭。相机position/quaternion/fovy均null；原生图像旋转180度口径不变。保存全部可用32步预测视频及实际双视角视频。GPU6单卡。\n\n每类仅3个变体用于行为对照，不构成稳定成功率估计。旧结果不用于选择变体；151也重跑以检查复现。各OOD单独使用LIBERO-plus自带变体，不叠加table9到其他OOD。预计60–90分钟，预留视频与日志约50–150MB。\n')
 return cmd
def gpu_safe():
 g=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 line=next(x for x in g.splitlines() if x.split(',')[0].strip()=='6')
 _,uuid,used,total,util=[x.strip() for x in line.split(',')]
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
 matched=[x for x in apps.splitlines() if x.split(',')[0].strip()==uuid]
 live=[x for x in matched if Path('/proc/'+x.split(',')[1].strip()).exists() or x.split(',')[2].strip()!='[Not Found]']
 return int(used)<6000 and int(total)-int(used)>18000 and int(util)<5 and not live,dict(gpu=line,apps=matched,live=live)
def main():
 cmd=prepare()
 lock=open(OUT/'launcher.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'formal').exists(),'Refuse duplicate formal evaluation'
 (OUT/'launcher.pid').write_text(str(os.getpid()))
 status('GPU6_PREFLIGHT')
 snapshots=[]
 for k in range(3):
  safe,snap=gpu_safe();snapshots.append(snap)
  assert safe,'GPU6 has a live process or insufficient headroom; no processes will be terminated'
  if k<2:time.sleep(10)
 dump('gpu_preflight.json',snapshots)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='6',MUJOCO_EGL_DEVICE_ID='6',PYTHONUNBUFFERED='1',
   PYTHONPATH='/data/pxchen/LIBERO-plus:/data/pxchen/PACT-WAM:'+env.get('PYTHONPATH',''),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
 subprocess.run([cmd[0],'-c','from libero.libero import get_libero_path; import fastwam; print("LIBERO_BDDL",get_libero_path("bddl_files"),flush=True)'],env=env,cwd=ROOT,check=True)
 (OUT/'formal').mkdir()
 status('RUNNING_GPU6')
 with (OUT/'formal/run.log').open('w') as log:
  p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  dump('process.json',dict(pid=p.pid,gpu=6,start=time.strftime('%F %T')))
  rc=p.wait()
 assert rc==0,f'manager exit {rc}'
 cfg=OmegaConf.load(OUT/'formal/manager_config.yaml')
 e=cfg.EVALUATION
 assert e.replan_steps==8 and e.num_inference_steps==10
 assert not e.vjepa2_ac.enabled and not e.vjepa2_ac.guidance.enabled
 assert all(e.vjepa2_ac.camera.get(k) is None for k in ['position','quaternion','fovy'])
 results=[json.loads(p.read_text()) for p in (OUT/'formal/libero_10').glob('*_results.json')]
 assert len(results)==21,f'{len(results)} results, expected21'
 tasks=json.loads((OUT/'task_manifest.json').read_text())['tasks'];by={r['task_id']:r for r in results}
 assert {r['task_id'] for r in tasks}==set(by)
 cats={cat:dict(successes=sum(by[tid]['successes'] for tid in ids),episodes=3,task_ids=ids) for cat,ids in PLAN.items()}
 dump('summary.json',dict(completed=time.strftime('%F %T'),episodes=21,successes=sum(r['successes'] for r in results),by_category=cats))
 status('COMPLETED')
if __name__=='__main__':
 import sys
 if '--prepare-only' in sys.argv:prepare();print('PREPARED',OUT,flush=True)
 else:
  try:main()
  except BaseException:
   if OUT.exists():dump('failure.json',dict(traceback=traceback.format_exc()));status('FAILED')
   raise
