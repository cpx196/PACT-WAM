import pathlib, json
import cv2
from PIL import Image, ImageDraw

root=pathlib.Path('/data/pxchen/PACT-WAM/evaluate_results/plus_short315_paired_rot180_20260926')
out=root/'harm_inspection'; out.mkdir(exist_ok=True)
for task in [159,263,372,489,732,1074,2376,2427]:
    canvas=Image.new('RGB',(1792,510),'white'); draw=ImageDraw.Draw(canvas)
    for row,group in enumerate(['control_noguidance','experiment_rot180_guidance4x']):
        paths=list((root/group/'libero_goal'/'videos').glob(f'*task{task}_trial0*.mp4'))
        assert len(paths)==1, paths
        cap=cv2.VideoCapture(str(paths[0])); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        draw.text((5,row*255+2),f'{group} task{task} frames={n}',fill='black')
        for col,idx in enumerate([60,90,120,n-1]):
            idx=min(idx,n-1);cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,frame=cap.read();assert ok
            im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im.thumbnail((448,224))
            canvas.paste(im,(col*448,row*255+20));draw.text((col*448+3,row*255+242),f'frame {idx}',fill='black')
        cap.release()
    canvas.save(out/f'detail_{task}.jpg')
