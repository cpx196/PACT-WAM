import pathlib,re,json
import cv2
from PIL import Image,ImageDraw
root=pathlib.Path('/data/pxchen/PACT-WAM/evaluate_results/long800_paired_seed7_20260927')
out=root/'grasp_frame_candidates';out.mkdir(exist_ok=True)
canvas=Image.new('RGB',(672,8*250),'white');draw=ImageDraw.Draw(canvas)
entries=[]
for row,(task,rp) in enumerate([(1071,18),(1071,20),(1071,22),(1071,24),(139,6),(139,8),(139,10),(139,12)]):
    suite='libero_10'
    group='control_noguidance'
    folder=root/group/suite/'predicted_videos'
    paths=sorted(p for p in folder.glob(f'*task{task}_trial0*gt-pred.mp4') if 'replan=all' not in p.name)
    for col,idx in enumerate([0,1,2]):
        found=[p for p in paths if f'replan={rp:04d}' in p.name]
        if not found:continue
        p=found[0];cap=cv2.VideoCapture(str(p));cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,f=cap.read();cap.release()
        if not ok:continue
        h,w=f.shape[:2];area=f[h//2:,:w//2] if col==0 else f[:h//2,:w//2]
        im=Image.fromarray(cv2.cvtColor(area,cv2.COLOR_BGR2RGB))
        canvas.paste(im,(col*224,row*250));draw.text((col*224+3,row*250+226),f'task{task} replan{rp} '+('OBS' if col==0 else f'PRED{idx}'),fill='black')
        entries.append(dict(task=task,replan=rp,path=str(p)))
canvas.save(out/'candidates.jpg');(out/'candidates.json').write_text(json.dumps(entries,indent=2))
