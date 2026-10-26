"""CPU-only compression proxy; historical rollout frames, not new inference."""
import json,glob,tempfile,time,math,statistics
from pathlib import Path
import imageio
from fastwam_fullvideo200_eval import compact,mp4
ROOT=Path('/data/pxchen/PACT-WAM')
OUT=ROOT/'evaluate_results/fastwam_noguidance_long200_fullvideo_20261015'
OLD=ROOT/'evaluate_results/long800_paired_seed7_20260927/control_noguidance/libero_10'
manifest=json.loads((OUT/'task_manifest.json').read_text());available=[];durations=[]
for row in manifest['tasks']:
    files=list((OLD/'videos').glob(f'*episode=task{row["task_id"]}_trial0--*.mp4'))
    results=list(OLD.glob(f'gpu*_task{row["task_id"]}_results.json'))
    if results:durations.append(json.loads(results[0].read_text())['duration'])
    if files:available.append((row,files[0]))
chosen=[]
for category in ('Sensor Noise','Camera Viewpoints','Robot Initial States','Objects Layout'):
    entries=[x for x in available if x[0]['category']==category]
    if entries:chosen.append(entries[0])
records=[]
with tempfile.TemporaryDirectory(prefix='fullvideo200_cpu_',dir=str(ROOT/'artifacts')) as tmp:
    for k,(row,file) in enumerate(chosen):
        reader=imageio.get_reader(str(file));n=reader.count_frames();reader.close();tic=time.perf_counter()
        reader=imageio.get_reader(str(file))
        def actual():
            for j,frame in enumerate(reader):
                if j%2==0 or j==n-1:yield compact(frame)
        actual_bytes=mp4(actual(),Path(tmp)/f'actual{k}.mp4',10);reader.close()
        reader=imageio.get_reader(str(file));indices=[min(n-1,n//3+4*j) for j in range(9)]
        frames=[compact(reader.get_data(i)) for i in indices];reader.close()
        prediction_bytes=mp4(frames,Path(tmp)/f'prediction{k}.mp4',5)
        check=imageio.get_reader(str(Path(tmp)/f'prediction{k}.mp4'));assert check.count_frames()==9;check.close()
        records.append(dict(task_id=row['task_id'],category=row['category'],source_frames=n,actual_bytes=actual_bytes,
            prediction_clip_bytes=prediction_bytes,estimated_episode_bytes=actual_bytes+math.ceil(n/8)*prediction_bytes,
            elapsed_s=time.perf_counter()-tic))
summary=dict(cpu_only=True,temporary_files_removed=True,proxy='Actual historical images used as moving-frame compression proxy; generated future entropy may differ',
    measurements=records,overlapping_old_episodes=len(durations),historical_selected_mean_duration_s=statistics.mean(durations) if durations else None,
    dual200_historical_compute_hours=100*statistics.mean(durations)/3600 if durations else None,
    estimated200_video_gib=200*statistics.mean(r['estimated_episode_bytes'] for r in records)/2**30 if records else None)
(OUT/'storage_time_estimate.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
