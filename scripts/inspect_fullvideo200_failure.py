"""Read-only audit of completed no-guidance results; generate diagnostic images."""
import json,collections
from pathlib import Path
import numpy as np,imageio.v2 as imageio
from PIL import Image,ImageDraw
ROOT=Path('/data/pxchen/PACT-WAM')
RUN=ROOT/'evaluate_results/fastwam_noguidance_long200_fullvideo_20261015/formal/libero_10'
OUT=ROOT/'artifacts/fullvideo200_failure_audit_20261015';OUT.mkdir(exist_ok=True)
current=[json.loads(p.read_text()) for p in RUN.glob('*_results.json')]
oldroot=ROOT/'evaluate_results/long800_paired_seed7_20260927/control_noguidance'
old={}
for p in oldroot.rglob('*_results.json'):
    d=json.loads(p.read_text());old[d['task_id']]=d
overlap=[dict(task=d['task_id'],now=d['successes'],old=old[d['task_id']]['successes'],old_description=old[d['task_id']].get('task_description'),now_description=d.get('task_description')) for d in current if d['task_id'] in old]
report=dict(completed=len(current),successes=sum(d['successes'] for d in current),historical_overlap=overlap)
(OUT/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
selected=[d['task'] for d in overlap if d['old']>0][:3]
selected+= [d['task_id'] for d in sorted(current,key=lambda d:d['start_time']) if d['task_id'] not in selected][:3-len(selected)]
canvas=Image.new('RGB',(384*4,222*2*len(selected)),'white');draw=ImageDraw.Draw(canvas)
for i,tid in enumerate(selected):
    actual=next((RUN/'videos').glob(f'task{tid}_trial0--*--actual.mp4'))
    reader=imageio.get_reader(actual);n=reader.count_frames()
    ids=np.linspace(0,n-1,4,dtype=int)
    movement=[];first=None
    for j,k in enumerate(ids):
        a=reader.get_data(int(k));first=a if first is None else first
        movement.append(float(np.abs(a.astype(float)-first.astype(float)).mean()))
        canvas.paste(Image.fromarray(a),(384*j,444*i+26));draw.text((384*j+3,444*i+5),f'Task {tid}: actual frame {k}/{n}',fill='black')
    reader.close()
    index=json.loads((RUN/'full_predictions'/f'task{tid}_trial0/index.json').read_text())
    clip=index['replans'][0];p=RUN/'full_predictions'/f'task{tid}_trial0'/clip['file'];reader=imageio.get_reader(p)
    for j,k in enumerate((0,2,5,8)):
        canvas.paste(Image.fromarray(reader.get_data(k)),(384*j,444*i+248));draw.text((384*j+3,444*i+227),f'Task {tid}: first imagined t+{k*4}',fill='black')
    reader.close();print('MOVEMENT',tid,movement,flush=True)
canvas.save(OUT/'actual_and_imagined.jpg',quality=85)
