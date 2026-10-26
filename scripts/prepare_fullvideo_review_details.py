"""Full9-frame main/wrist comparison for selected review candidates; CPU only."""
import json,sys,concurrent.futures
from pathlib import Path
import numpy as np,imageio.v2 as imageio
from PIL import Image,ImageDraw
ROOT=Path('/data/pxchen/PACT-WAM');OUT=ROOT/'artifacts/fullvideo200_review64_20261016'
MEDIA=ROOT/'evaluate_results/fastwam_noguidance_long200_fullvideo_camera_fixed_20261015/formal/libero_10'
requests=json.loads((OUT/'detail_requests.json').read_text());(OUT/'details').mkdir(exist_ok=True)
def read(p):
    r=imageio.get_reader(p,input_params=['-threads','1']);a=[x for x in r];r.close();return a
def one(req):
    tid=req['id'];p=next((MEDIA/'videos').glob(f'task{tid}_trial0--*actual.mp4'));actual=read(p)
    controls=np.array(json.loads(p.with_suffix('.json').read_text())['source_control_indices'])
    base=MEDIA/'full_predictions'/f'task{tid}_trial0';index=json.loads((base/'index.json').read_text())
    for rid in req['r']:
        ent=index['replans'][rid];pred=read(base/ent['file']);assert len(pred)==9
        # Columns are offsets0/4/8/12/16 then20/24/28/32. Main+Wrist separate rows.
        canvas=Image.new('RGB',(5*192,8*215+40),'white');d=ImageDraw.Draw(canvas)
        d.text((4,4),f'Task {tid}, replan {rid}, start control {ent["control_start"]}. W=predicted; A=actual.',fill='black')
        for k in range(9):
            block=k//5;col=k%5;target=ent['control_start']+4*k
            a=actual[int(np.abs(controls-target).argmin())] if target<=controls[-1] else None
            for q,(src,label) in enumerate([(pred[k],'W main'),(a,'A main'),(pred[k],'W wrist'),(a,'A wrist')]):
                x=col*192;y=40+(block*4+q)*215
                d.text((x+2,y+2),f'{label} t={target} (+{k*4})',fill='black')
                if src is not None:
                    start=192 if q>=2 else 0;im=Image.fromarray(np.asarray(src)[:,start:start+192]);canvas.paste(im,(x,y+21))
                else:d.text((x+5,y+90),'Not recorded',fill='black')
        canvas.save(OUT/'details'/f'task{tid:04d}_r{rid:03d}.jpg',quality=86)
    print('DETAIL',tid,req['r'],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(one,requests))
