from pathlib import Path
import imageio.v2 as imageio
from PIL import Image,ImageDraw
import json
R=Path('/data/pxchen/PACT-WAM/evaluate_results/weighted_six_four_seeds_sameT5_20261022')
O=Path('/data/pxchen/PACT-WAM/artifacts/seed37_audit_20261022')
selections={151:[72,76,80,84,86,88,90,92,94,96,100,104],559:[104,112,118,128,264,277,288,304,416,430,432,448],590:[88,96,104,112,118,124,128,136,144,160,176,192]}
for tid,steps in selections.items():
 canvas=Image.new('RGB',(2048,1728),'white');draw=ImageDraw.Draw(canvas)
 for ai,arm in enumerate(['control','weighted']):
  video=next((R/arm/'seed_37/libero_10/videos').glob(f'*task{tid}_trial0*'));reader=imageio.get_reader(video);n=reader.count_frames()
  for i,t in enumerate(steps):
   x=i%4*512;y=(i//4*2+ai)*288;draw.text((x+4,y+4),f'{arm} task{tid} control{t}',fill='black')
   if t<n:canvas.paste(Image.fromarray(reader.get_data(t)),(x,y+24))
  reader.close()
 canvas.save(O/f'task{tid}_details.jpg',quality=90)
# Key saved imagined/actual clips, no new inference. Native comparison=predicted top, actual bottom.
for tid,plans in {151:[10,11],559:[14,34,53],590:[14,15]}.items():
 for arm in ['weighted']:
  tiles=[]
  for r in plans:
   files=list((R/arm/'seed_37/libero_10/predicted_videos').glob(f'*task{tid}_trial0*replan={r:04d}--gt-pred.mp4'))
   assert len(files)==1,(tid,r,files)
   rd=imageio.get_reader(files[0]);n=rd.count_frames()
   print('CLIP',tid,r,n,flush=True)
   row=[]
   for k in range(n):row.append((r,k,Image.fromarray(rd.get_data(k))))
   rd.close();tiles.append(row)
  height=sum(max(im.height for _,_,im in row)+26 for row in tiles)
  width=max(sum(im.width for _,_,im in row) for row in tiles)
  canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas);y=0
  for row in tiles:
   x=0
   for r,k,im in row:
    draw.text((x+5,y+3),f'weighted task{tid} r{r} control{r*8}+{k*4} | pred top / actual bottom',fill='black');canvas.paste(im,(x,y+24));x+=im.width
   y+=max(im.height for _,_,im in row)+26
  canvas.save(O/f'task{tid}_pred_actual.jpg',quality=90)
