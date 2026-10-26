"""AR2 guidance with passive full-horizon video recording; preserve target selection."""
import inspect,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from fastwam_ar2_eval import ARRanker,RawAdapter
from fastwam_fullvideo200_eval import compact,mp4,rollout
ROOT=Path('/data/pxchen/PACT-WAM')
def install(ns):
 native=ns['run_single_episode'];signature=inspect.signature(native)
 select=ns['_select_predicted_future_frames']
 def record_selection(pred,cfg):
  # Keep the complete decoded video as recording-only side data. The native
  # selector's return value (and therefore guidance targets) stays unchanged.
  ns['_recording_full_future_frames']=list(pred)
  return select(pred,cfg)
 ns['_select_predicted_future_frames']=record_selection
 def recorded(*args,**kwargs):
  bound=signature.bind(*args,**kwargs).arguments;cfg=bound['cfg']
  assert cfg.EVALUATION.vjepa2_ac.guidance.enabled and not cfg.EVALUATION.vjepa2_ac.guidance.trigger.enabled
  assert all(cfg.EVALUATION.vjepa2_ac.camera.get(k) is None for k in ('position','quaternion','fovy'))
  result=native(*args,**kwargs);success,replay,clips,psnr,timing=result
  base=Path(str(cfg.EVALUATION.output_dir))/str(cfg.EVALUATION.task_suite_name)/'full_predictions'/f'task{cfg.EVALUATION.task_id}_trial{bound["episode_idx"]}'
  base.mkdir(parents=True,exist_ok=True);records=[];tic=time.perf_counter()
  for clip in clips:
   full=clip['full_pred_frames'];assert len(full)==9
   file=base/f'replan_{int(clip["replan_idx"]):04d}.mp4'
   def frames():
    for j,frame in enumerate(full):
     im=Image.fromarray(compact(frame));ImageDraw.Draw(im).text((3,3),f'replan {clip["replan_idx"]} | t+{j*4}',fill='white',stroke_width=1,stroke_fill='black');yield np.asarray(im)
   size=mp4(frames(),file,5)
   records.append(dict(replan_index=int(clip['replan_idx']),control_start=int(clip['control_start']),prediction_control_offsets=list(range(0,33,4)),prediction_frame_count=9,file=file.name,bytes=size,executed_prefix_steps=8,actual_prefix_frames=len(clip['gt_frames'])))
  report=dict(task_id=int(cfg.EVALUATION.task_id),seed=int(cfg.seed),success=bool(success),task_instruction=bound['task_description'],guidance=True,trigger=False,replan_steps=8,full_prediction_frames=9,full_prediction_horizon_control_steps=32,prediction_fps=5,cameras=['agentview','wrist'],view_resolution=[192,192],replans=records,encoding_seconds=time.perf_counter()-tic,video_bytes=sum(x['bytes'] for x in records),note='Independent 32-step imagined futures; recording uses full video only, guidance retains native t+4/t+8 target selection.')
  (base/'index.json').write_text(json.dumps(report,indent=2));print('GUIDANCE_FULLVIDEO_COMPLETE '+json.dumps(dict(task_id=report['task_id'],replans=len(records))),flush=True)
  return result
 ns['run_single_episode']=recorded;ns['save_rollout_video']=rollout;ns['save_prediction_video']=lambda *a,**kw:None
def main():
 source=ROOT/'experiments/libero/eval_libero_single.py';code=source.read_text()
 needle='"pred_frames": predicted_future_frames,';assert code.count(needle)==1
 # trigger_future_frames is the complete decoded video already available in the
 # native path. Do NOT change _select_predicted_future_frames or guidance inputs.
 code=code.replace(needle,needle+'\n                    "full_pred_frames": list(_recording_full_future_frames),\n                    "control_start": t-num_steps_wait,')
 end='if __name__ == "__main__":\n    eval_single_process()';assert code.count(end)==1
 code=code.replace(end,'if __name__ == "__main__":\n    VJEPA2ACRanker = ARRanker\n    LiberoACAdapter = RawAdapter\n    install_recording(globals())\n    eval_single_process()')
 exec(compile(code,str(source),'exec'),dict(__name__='__main__',__file__=str(source),ARRanker=ARRanker,RawAdapter=RawAdapter,install_recording=install))
if __name__=='__main__':main()
