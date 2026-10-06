"""CPU-only data-contract validation before corrected predictor training."""
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_vjepa2_ac_libero90_euler_fixed import DATA_FORMAT_VERSION, libero_ee_state_as_euler
from experiments.libero.vjepa2_ac_ranker import LiberoACAdapter

data = Path('/data/pxchen/libero-suite/LIBERO/datasets/libero_90')
adapter = LiberoACAdapter(low_level_steps_per_ac_step=4)
files = sorted(data.glob('*.hdf5'))
assert len(files) == 90, len(files)
checked, train, validation = 0, 0, 0
for path in files:
    with h5py.File(path) as f:
        for demo in f['data']:
            group = f[f'data/{demo}']
            n = len(group['actions'])
            if n < 17:
                continue
            if int(demo.split('_')[-1]) % 10 == 0:
                validation += 1
            else:
                train += 1
        # Inspect both a train and a validation trajectory in every file.
        for demo in ('demo_0', 'demo_1'):
            group = f[f'data/{demo}']
            n = len(group['actions'])
            for t in (8, (n-1)//2):
                ee = np.asarray(group['obs/ee_states'][t], dtype=np.float32)
                fingers = np.asarray(group['obs/gripper_states'][t], dtype=np.float32)
                actions = np.asarray(group['actions'][t:t+8], dtype=np.float32)
                canonical = libero_ee_state_as_euler(ee)
                assert ee.shape == (6,) and actions.shape == (8,7)
                assert len(group['obs/agentview_rgb']) > t+8
                obs = {'robot0_eef_pos': ee[:3], 'robot0_eef_quat': Rotation.from_rotvec(ee[3:]).as_quat(),
                       'robot0_gripper_qpos': fingers}
                initial = np.r_[canonical, 1-np.clip(abs(float(fingers[0]-fingers[1]))/.08,0,1)].astype(np.float32)
                np.testing.assert_allclose(Rotation.from_rotvec(ee[3:]).as_matrix(),
                                           Rotation.from_euler('xyz',canonical[3:]).as_matrix(),atol=1e-6)
                np.testing.assert_allclose(initial, adapter.state_from_observation(obs),atol=1e-6)
                ac = adapter.convert(actions,initial)
                tc = adapter.convert_torch(torch.from_numpy(actions),torch.from_numpy(initial))
                np.testing.assert_allclose(ac.actions,tc.actions.numpy(),atol=1e-5)
                np.testing.assert_allclose(ac.states,tc.states.numpy(),atol=1e-5)
                assert np.isfinite(ac.actions).all() and np.isfinite(ac.states).all()
                checked += 1
assert (train,validation) == (4050,450), (train,validation)
script = Path(__file__).with_name('train_vjepa2_ac_libero90_euler_fixed.py')
print(json.dumps({'status':'PASSED','data_format_version':DATA_FORMAT_VERSION,
                  'files':len(files),'checked_windows':checked,
                  'train_trajectories':train,'validation_trajectories':validation,
                  'training_script_sha256':hashlib.sha256(script.read_bytes()).hexdigest()},indent=2))
