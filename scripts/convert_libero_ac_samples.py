"""CPU-only, bounded LIBERO -> canonical JEPA-AC sample conversion.

Keeps source HDF5 files unchanged. This is a format/semantics smoke test,
not a full dataset export and not a training or evaluation job.
"""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.libero.vjepa2_ac_ranker import LiberoACAdapter
from scripts.train_vjepa2_ac_libero90_euler_fixed import libero_ee_state_as_euler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--file-limit', type=int, default=3)
    parser.add_argument('--windows-per-file', type=int, default=3)
    parser.add_argument('--image-rotation', type=int, choices=(0,180), default=180)
    args = parser.parse_args()
    if not 1 <= args.file_limit <= 3 or not 1 <= args.windows_per_file <= 3:
        parser.error('Small-scale smoke export only: max 3 files x 3 windows')
    if args.output.exists():
        raise FileExistsError(f'Refuse to overwrite existing output: {args.output}')
    files = sorted(args.data.glob('*.hdf5'))[:args.file_limit]
    if not files:
        raise FileNotFoundError(args.data)
    args.output.mkdir(parents=True)
    adapter = LiberoACAdapter(low_level_steps_per_ac_step=4)
    manifest = {'format_version': 'libero_ac_euler_v1', 'cpu_only': True,
                'source_unchanged': True, 'image_rotation_degrees': args.image_rotation,
                'camera': 'agentview only', 'eef_orientation': 'Euler XYZ, radians',
                'gripper_state': 'closedness [0,1], observed width / 0.08m',
                'raw_action_gripper': '-1 open, +1 close (LIBERO executable commands)',
                'ac_action_semantics': 'sum scaled OSC deltas over 4 low-level steps; rotation composition; closedness delta',
                'ac_state_semantics': 'initial observed state; later states command-integrated by adapter, NOT measured future poses',
                'osc_position_scale': .05, 'osc_rotation_scale': .5,
                'frame_times': '[t-8,t]; targets [t,t+4] and [t+4,t+8]',
                'checks': ['rotation-matrix equivalence', 'online state equivalence',
                           'NumPy/Torch action/state adapter equivalence', 'finite shapes'],
                'samples': []}
    for path in files:
        with h5py.File(path) as f:
            demo = sorted(f['data'], key=lambda name:int(name.split('_')[-1]))[0]
            group = f[f'data/{demo}']
            n = len(group['actions'])
            if n < 17:
                continue
            for t in np.unique(np.linspace(8,n-9,args.windows_per_file).astype(int)):
                ids = [int(t-8),int(t),int(t),int(t+4),int(t+4),int(t+8)]
                frames = np.stack([group['obs/agentview_rgb'][i] for i in ids])
                if args.image_rotation == 180:
                    frames = np.rot90(frames,2,axes=(1,2)).copy()
                ee = np.asarray(group['obs/ee_states'][t], dtype=np.float32)
                fingers = np.asarray(group['obs/gripper_states'][t], dtype=np.float32)
                actions = np.asarray(group['actions'][t:t+8], dtype=np.float32)
                state = np.r_[libero_ee_state_as_euler(ee),
                              1-np.clip(abs(float(fingers[0]-fingers[1]))/.08,0,1)].astype(np.float32)
                assert ee.shape == (6,) and actions.shape == (8,7)
                np.testing.assert_allclose(Rotation.from_rotvec(ee[3:]).as_matrix(),
                                           Rotation.from_euler('xyz',state[3:6]).as_matrix(),atol=1e-6)
                obs = {'robot0_eef_pos':ee[:3], 'robot0_eef_quat':Rotation.from_rotvec(ee[3:]).as_quat(),
                       'robot0_gripper_qpos':fingers}
                np.testing.assert_allclose(state, adapter.state_from_observation(obs),atol=1e-6)
                ac = adapter.convert(actions, state)
                tc = adapter.convert_torch(torch.from_numpy(actions),torch.from_numpy(state))
                np.testing.assert_allclose(ac.actions,tc.actions.numpy(),atol=1e-5)
                np.testing.assert_allclose(ac.states,tc.states.numpy(),atol=1e-5)
                for array in (state,ac.actions,ac.states):
                    assert np.isfinite(array).all()
                name = f'{path.stem}__{demo}__t{t:06d}.npz'
                arrays = dict(context_rgb=frames[:2],target_rgb=frames[2:].reshape(2,2,*frames.shape[1:]),
                              initial_state=state,low_level_actions=actions,
                              ac_actions=ac.actions[0],ac_states=ac.states[0],source_frame_ids=np.array(ids))
                np.savez_compressed(args.output/name,**arrays)
                manifest['samples'].append({'file':name,'source_file':str(path),'demo':demo,'t':int(t),
                                            'split':'validation: demo index 0',
                                            'shapes':{k:list(v.shape) for k,v in arrays.items()}})
    if not manifest['samples']:
        raise RuntimeError('No valid windows')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'samples':len(manifest['samples']),'output':str(args.output),
                      'all_checks':'PASSED','example_shapes':manifest['samples'][0]['shapes']},indent=2))


if __name__ == '__main__':
    main()
