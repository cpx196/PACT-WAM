"""Extract paired rollout contact sheets, without rotating displayed videos."""
from pathlib import Path
import cv2
from PIL import Image, ImageDraw

root = Path('/data/pxchen/PACT-WAM/evaluate_results/long800_paired_seed7_20260927')
out = Path('/data/pxchen/PACT-WAM/evaluate_results/harm_diagnostic_20260928/video_inspection')
out.mkdir(exist_ok=True)
for task in (2455, 64, 2217):
    canvas = Image.new('RGB', (1792, 520), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, group in enumerate(('control_noguidance', 'experiment_rot180_guidance4x')):
        paths = list((root/group/'libero_10'/'videos').glob(f'*task{task}_trial0*.mp4'))
        assert len(paths) == 1, paths
        cap = cv2.VideoCapture(str(paths[0]))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        draw.text((5, row*260+2), f'{group} task {task} frames {n}', fill='black')
        for col, index in enumerate((40, 80, 120, 160, 220, 300, n-1)):
            index = min(index, n-1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            assert ok
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.thumbnail((256, 224))
            canvas.paste(image, (col*256, row*260+20))
            draw.text((col*256+3, row*260+243), f'frame {index}', fill='black')
        cap.release()
    canvas.save(out/f'paired_task{task}.jpg')
    print(f'task {task}: {out}/paired_task{task}.jpg')
