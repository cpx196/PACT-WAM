"""CPU-only review sheets of all failed native-camera runs. No model inference."""
import json,os,time,concurrent.futures
from pathlib import Path
import numpy as np,imageio.v2 as imageio
from PIL import Image,ImageDraw
ROOT=Path('/data/pxchen/PACT-WAM')
RUN=ROOT/'evaluate_results/fastwam_noguidance_long200_fullvideo_camera_fixed_20261015'
MEDIA=RUN/'formal/libero_10'
OUT=ROOT/'artifacts/fullvideo200_review64_20261016'
OUT.mkdir(parents=True,exist_ok=True);(OUT/'sheets').mkdir(exist_ok=True)
manifest={r['task_id']:r for r in json.loads((RUN/'task_manifest.json').read_text())['tasks']}
results=sorted([json.loads(p.read_text()) for p in MEDIA.glob('*_results.json') if json.loads(p.read_text())['successes']==0],key=lambda r:r['task_id'])
assert len(results)==64
W=192;H=218
def read(path):
    r=imageio.get_reader(path,input_params=['-threads','1']);frames=[x for x in r];r.close();return frames
def tile(canvas,xy,arr,label):
    x,y=xy;d=ImageDraw.Draw(canvas);d.text((x+3,y+3),label,fill='black')
    if arr is not None:canvas.paste(Image.fromarray(np.asarray(arr)[:,:192]).convert('RGB').resize((192,192)),(x,y+23))
    else:d.text((x+10,y+80),'Beyond recorded horizon',fill='black')
def one(d):
    tid=d['task_id'];outfile=OUT/'sheets'/f'task{tid:04d}.jpg'
    if outfile.exists():return tid
    act=next((MEDIA/'videos').glob(f'task{tid}_trial0--*--actual.mp4'))
    frames=read(act);am=json.loads(act.with_suffix('.json').read_text());controls=np.asarray(am['source_control_indices'])
    ix=json.loads((MEDIA/'full_predictions'/f'task{tid}_trial0/index.json').read_text());replans=ix['replans']
    rows=6;canvas=Image.new('RGB',(6*W,(2+rows)*H+74),'white');draw=ImageDraw.Draw(canvas)
    task=manifest[tid]
    draw.text((5,5),f'Task {tid} | {task["category"]} | actual FAILED | {len(replans)} replans',fill='black')
    instruction=task['instruction']
    for k in range(0,min(len(instruction),240),135):draw.text((5,23+(k//135)*16),instruction[k:k+135],fill='black')
    for j,k in enumerate(np.linspace(0,len(frames)-1,12,dtype=int)):
        tile(canvas,((j%6)*W,74+(j//6)*H),frames[k],f'Actual control {controls[k]}')
    selected=np.unique(np.rint(np.linspace(0,len(replans)-1,rows)).astype(int))
    for rownum,k in enumerate(selected):
        entry=replans[k];pred=read(MEDIA/'full_predictions'/f'task{tid}_trial0'/entry['file']);assert len(pred)==9
        start=entry['control_start']
        for j,delta in enumerate([0,16,32]):
            tile(canvas,(j*W,74+(2+rownum)*H),pred[delta//4],f'W r{k} at {start}+{delta}')
            target=start+delta
            actual=frames[int(np.abs(controls-target).argmin())] if target<=controls[-1] else None
            tile(canvas,((3+j)*W,74+(2+rownum)*H),actual,f'Actual at {target}')
    canvas.save(outfile,quality=85)
    print('SHEET',tid,flush=True)
    return tid
started=time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(one,results))
rows=[dict(task_id=d['task_id'],instruction=manifest[d['task_id']]['instruction'],base_task=manifest[d['task_id']]['base_task'],category=manifest[d['task_id']]['category'],
           sheet=f'sheets/task{d["task_id"]:04d}.jpg',review_status='UNREVIEWED',duration=d['duration']) for d in results]
(OUT/'manifest.json').write_text(json.dumps(rows,indent=2))
(OUT/'summary.json').write_text(json.dumps(dict(status='sheets_ready',windows=64,seconds=time.time()-started,raw_videos_copied=False,models_loaded=False),indent=2))
print('DONE',len(rows),time.time()-started,flush=True)
