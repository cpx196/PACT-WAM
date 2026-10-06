"""Download one official task file for four-window smoke, without touching GPUs."""
import json
from pathlib import Path
import h5py
from huggingface_hub import hf_hub_download

root = Path('/data/pxchen/PACT-WAM/artifacts/libero10_guidance_backward_smoke_20260929')
root.mkdir(exist_ok=True)
path = hf_hub_download(
    repo_id='yifengzhu-hf/LIBERO-datasets', repo_type='dataset',
    filename='libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5',
    local_dir=str(root/'data'),
)
with h5py.File(path, 'r') as f:
    demos = sorted(f['data'], key=lambda s: int(s.split('_')[-1]))
    windows = []
    for demo in demos[:4]:
        g = f['data'][demo]
        n = len(g['actions'])
        assert n >= 17 and g['actions'].shape[-1] == 7
        for key in ['agentview_rgb', 'ee_states', 'gripper_states', 'eye_in_hand_rgb']:
            assert key in g['obs'], key
        windows.append(dict(demo=demo, t=max(8,min(n//3,n-9)), length=n))
report = dict(source=path, windows=windows, purpose='mechanical gradient smoke only; not benchmark tuning')
(root/'data_manifest.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
print('DATA_PREFLIGHT_PASSED', flush=True)
