import pathlib,json
import cv2
from PIL import Image
root=pathlib.Path('/data/pxchen/PACT-WAM/evaluate_results/long800_paired_seed7_20260927/control_noguidance/libero_10')
out=pathlib.Path('/data/pxchen/PACT-WAM/evaluate_results/diagram_libero_longhorizon_lift_frames');out.mkdir(exist_ok=True)
roll=list((root/'videos').glob('*task139_trial0*.mp4'));pred=list((root/'predicted_videos').glob('*task139_trial0*replan=0008--gt-pred.mp4'))
assert len(roll)==len(pred)==1
def read(path,index):
    cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read();cap.release();assert ok,(path,index)
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
files=[]
for name,path,index,prediction in [('obs_previous',roll[0],63,False),('obs_current',pred[0],0,False),('future_1',pred[0],1,True),('future_2',pred[0],2,True)]:
    frame=read(path,index)
    # Rollout: main and wrist side-by-side; prediction: pred above GT, two views side-by-side.
    h,w=frame.shape[:2]
    if name=='obs_current':main=frame[h//2:,:w//2]
    else:main=frame[:h//2,:w//2] if prediction else frame[:,:w//2]
    image=Image.fromarray(main)
    image.save(out/f'{name}.png')
    files.append(dict(file=f'{name}.png',source=str(path),frame_index=index,orientation='unchanged saved video orientation; no extra rotation',original_shape=list(frame.shape),output_size=list(image.size)))
(out/'provenance.json').write_text(json.dumps(dict(task_id=139,task='Put both the alphabet soup and the cream cheese box in the basket.',group='control_noguidance',replan=8,notes='Obs current is the exact GT frame0 of the prediction comparison clip; future1/2 are actual WAM outputs from that same replan. Baked-in labels retained. No additional rotation. Presentation instruction omits OOD texture suffix.',images=files),indent=2))
(out/'language_instruction.txt').write_text('Put both the alphabet soup and the cream cheese box in the basket.\n')
print(json.dumps(files,indent=2))
