"""Native no-guidance evaluation; run-local full-horizon media recording only."""
import os,json,inspect,time
from pathlib import Path
from wand.api import library as _wand_library
import imageio
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path('/data/pxchen/PACT-WAM')
VIEW=192
def compact(frame):
    if isinstance(frame,dict):
        views=[Image.fromarray(np.asarray(v,dtype=np.uint8)).convert('RGB').resize((VIEW,VIEW),Image.Resampling.BILINEAR) for v in frame.values()]
        return np.concatenate([np.asarray(x) for x in views],axis=1)
    im=frame.convert('RGB') if isinstance(frame,Image.Image) else Image.fromarray(np.asarray(frame,dtype=np.uint8)).convert('RGB')
    return np.asarray(im.resize((round(im.width*VIEW/im.height/2)*2,VIEW),Image.Resampling.BILINEAR))

def mp4(frames,path,fps):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.stem+'.tmp.mp4')
    writer=imageio.get_writer(str(tmp),format='FFMPEG',fps=fps,codec='libx264',pixelformat='yuv420p',
        macro_block_size=16,ffmpeg_params=['-crf','28','-preset','veryfast','-threads','1'])
    try:
        for frame in frames:writer.append_data(frame)
    finally:writer.close()
    tmp.replace(path)
    return path.stat().st_size

def rollout(rollout_dir,rollout_images,idx,success,task_description,log_file=None,fps=24):
    # Actual images are sampled at every second control step; prediction frames
    # are NEVER subsampled. Keep the final frame even for odd episode lengths.
    ids=list(range(0,len(rollout_images),2))
    if ids and ids[-1]!=len(rollout_images)-1:ids.append(len(rollout_images)-1)
    path=Path(rollout_dir)/f'{idx}--success={bool(success)}--actual.mp4'
    mp4((compact(rollout_images[i]) for i in ids),path,10)
    path.with_suffix('.json').write_text(json.dumps(dict(task=task_description,success=bool(success),
        source_control_indices=ids,saved_fps=10,source_frame_count=len(rollout_images),
        cameras=['agentview','wrist'],view_resolution=[192,192]),indent=2))
    print('FULLVIDEO_ACTUAL '+str(path),flush=True)
    return str(path)

def install(ns):
    native=ns['run_single_episode'];signature=inspect.signature(native)
    # Native evaluator selects only 1+replan_steps/4 frames before creating the
    # rollout clip. Bypass that visualization selector as well as preserving
    # the full list before the later executed-prefix truncation. No policy
    # consumes this selector in the no-guidance/no-ranking configuration.
    ns['_select_predicted_future_frames']=lambda pred,cfg:list(pred)
    def recorded(*args,**kwargs):
        bound=signature.bind(*args,**kwargs).arguments
        cfg=bound['cfg']
        assert not cfg.EVALUATION.vjepa2_ac.enabled and not cfg.EVALUATION.vjepa2_ac.guidance.enabled
        camera=cfg.EVALUATION.vjepa2_ac.camera
        assert all(camera.get(k) is None for k in ('position','quaternion','fovy')), 'Refuse overridden native task camera'
        assert int(camera.image_rotation_degrees)==180
        result=native(*args,**kwargs)
        success,replay,clips,psnr,timing=result
        base=Path(str(cfg.EVALUATION.output_dir))/str(cfg.EVALUATION.task_suite_name)/'full_predictions'/f'task{cfg.EVALUATION.task_id}_trial{bound["episode_idx"]}'
        base.mkdir(parents=True,exist_ok=True)
        records=[];tic=time.perf_counter()
        for clip in clips:
            full=clip['full_pred_frames'];assert len(full)==9, len(full)
            file=base/f'replan_{int(clip["replan_idx"]):04d}.mp4'
            def frames():
                for j,frame in enumerate(full):
                    im=Image.fromarray(compact(frame))
                    ImageDraw.Draw(im).text((3,3),f'replan {clip["replan_idx"]} | t+{j*4}',fill='white',stroke_width=1,stroke_fill='black')
                    yield np.asarray(im)
            size=mp4(frames(),file,5)
            records.append(dict(replan_index=int(clip['replan_idx']),control_start=int(clip['control_start']),
                prediction_control_offsets=list(range(0,33,4)),prediction_frame_count=len(full),
                file=file.name,bytes=size,executed_prefix_steps=8,actual_prefix_frames=len(clip['gt_frames'])))
        report=dict(task_id=int(cfg.EVALUATION.task_id),seed=int(cfg.seed),success=bool(success),
            task_instruction=bound['task_description'],guidance=False,replan_steps=8,full_prediction_frames=9,
            full_prediction_horizon_control_steps=32,prediction_fps=5,control_frequency_hz=float(getattr(bound['env'],'control_freq',20)),
            cameras=['agentview','wrist'],view_resolution=[192,192],replans=records,
            note='Each replan is an independent 32-control-step imagined future, NOT one continuous imagined-success trajectory.',
            encoding_seconds=time.perf_counter()-tic,video_bytes=sum(x['bytes'] for x in records))
        (base/'index.json').write_text(json.dumps(report,indent=2))
        print('FULLVIDEO_COMPLETE '+json.dumps(dict(task_id=report['task_id'],replans=len(records),frames=len(records)*9,bytes=report['video_bytes'],encoding_s=report['encoding_seconds'])),flush=True)
        return result
    ns['run_single_episode']=recorded
    ns['save_rollout_video']=rollout
    # Superseded by complete predictions + actual rollout + explicit index.
    # Do not duplicate truncated 3-frame comparisons or concatenate imagined plans.
    ns['save_prediction_video']=lambda *a,**kw:None

def main():
    import sys
    sys.path.insert(0,str(ROOT))
    source=ROOT/'experiments/libero/eval_libero_single.py';code=source.read_text()
    needle='"pred_frames": predicted_future_frames,'
    assert code.count(needle)==1
    code=code.replace(needle,needle+'\n                    "full_pred_frames": list(predicted_future_frames),\n                    "control_start": t-num_steps_wait,')
    ending='if __name__ == "__main__":\n    eval_single_process()'
    assert code.count(ending)==1
    code=code.replace(ending,'if __name__ == "__main__":\n    install_recording(globals())\n    eval_single_process()')
    exec(compile(code,str(source),'exec'),dict(__name__='__main__',__file__=str(source),install_recording=install))

if __name__=='__main__':main()
