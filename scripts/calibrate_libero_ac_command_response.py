"""Train-only constant OSC command response calibration; held-out pose audit.

No predictor fitting, no GPU use. Constants approximate realized motion and
must not be mistaken for exact physics or deployed into old JEPA silently.
"""
import json
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation

ROOT=Path('/data/pxchen/PACT-WAM')
MANIFEST=ROOT/'artifacts/libero10_idm_dense_windows_20261004/windows_stride32.json'
DATA=ROOT/'artifacts/libero10_guidance_backward_smoke_20260929/data/libero_10'
OUT=ROOT/'artifacts/libero_ac_command_response_calibration_20261007'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'summary.json').exists()
    rows=json.loads(MANIFEST.read_text());bytask=defaultdict(list)
    for i,row in enumerate(rows):bytask[row['task_file']].append((i,row))
    commands=np.empty((len(rows),8,6),dtype=np.float64)
    xyz=np.empty((len(rows),3,3),dtype=np.float64)
    rotvec=np.empty((len(rows),3,3),dtype=np.float64)
    for task,items in sorted(bytask.items()):
        with h5py.File(DATA/task,'r') as file:
            demos={}
            for i,row in items:
                demo=row['demo'];t=row['t']
                if demo not in demos:
                    g=file['data'][demo]
                    demos[demo]=(np.asarray(g['actions']),np.asarray(g['obs']['ee_states']))
                action,ee=demos[demo]
                commands[i]=np.clip(action[t:t+8,:6],-1,1)
                xyz[i]=ee[[t,t+4,t+8],:3];rotvec[i]=ee[[t,t+4,t+8],3:6]
        print(json.dumps(dict(event='TASK_LOADED',task=task,windows=len(items))),flush=True)
    train=np.array([i for i,r in enumerate(rows) if r['split']=='train'])
    val=np.array([i for i,r in enumerate(rows) if r['split']=='val'])
    assert (len(train),len(val))==(3871,436)
    assert {(rows[i]['task_file'],rows[i]['demo']) for i in train}.isdisjoint(
        {(rows[i]['task_file'],rows[i]['demo']) for i in val})
    nominal=np.cumsum(commands[:,:,:3]*.05,axis=1)[:,[3,7]]
    actual=xyz[:,1:]-xyz[:,:1]
    translation_gain=float((nominal[train]*actual[train]).sum()/(nominal[train]**2).sum())
    assert 0<translation_gain<=1
    relative=[]
    for k in [1,2]:
        relative.append(Rotation.from_rotvec(rotvec[:,k])*Rotation.from_rotvec(rotvec[:,0]).inv())
    def rotations(gain):
        current=Rotation.identity(len(rows));endpoints=[]
        for k in range(8):
            current=Rotation.from_rotvec(commands[:,k,3:6]*.5*gain)*current
            if k in [3,7]:endpoints.append(current)
        return endpoints
    def angle_error(gain):
        pred=rotations(gain)
        return np.stack([(pred[k]*relative[k].inv()).magnitude() for k in range(2)],axis=1)
    fit=minimize_scalar(lambda gain:float((angle_error(gain)[train]**2).mean()),bounds=(.001,1.),method='bounded',options={'xatol':1e-6})
    assert fit.success
    rotation_gain=float(fit.x)
    baseline_pos=np.linalg.norm(nominal-actual,axis=2)
    calibrated_pos=np.linalg.norm(nominal*translation_gain-actual,axis=2)
    baseline_rot=angle_error(1.)*180/np.pi
    calibrated_rot=angle_error(rotation_gain)*180/np.pi
    def stats(a):
        return dict(mean=a.mean(0).tolist(),median=np.median(a,axis=0).tolist(),p95=np.quantile(a,.95,axis=0).tolist())
    def report(ids):
        return dict(windows=len(ids),baseline_position_cm=stats(baseline_pos[ids]*100),
                    calibrated_position_cm=stats(calibrated_pos[ids]*100),
                    baseline_rotation_deg=stats(baseline_rot[ids]),calibrated_rotation_deg=stats(calibrated_rot[ids]),
                    position_better_fraction=(calibrated_pos[ids]<baseline_pos[ids]).mean(0).tolist(),
                    rotation_better_fraction=(calibrated_rot[ids]<baseline_rot[ids]).mean(0).tolist())
    config=dict(schema='libero_ac_command_response_v1',fitted_split='train_only',train_windows=len(train),val_windows=len(val),
                manifest=str(MANIFEST),osc_position_scale=.05,osc_rotation_scale=.5,
                translation_response_gain=translation_gain,rotation_response_gain=rotation_gain,
                fit='least-squares XYZ and squared SO3 geodesic endpoint error at4/8steps',
                gripper_response='unchanged command target, not modeled',
                caveat='effective response of this dataset/controller; not exact dynamics; not default for old Predictor',
                predictor_changed=False,online_default_changed=False)
    (OUT/'calibration.json').write_text(json.dumps(config,indent=2))
    result=dict(config=config,train=report(train),val=report(val),per_task_val={
        task:report(np.array([i for i in val if rows[i]['task_file']==task])) for task in sorted(bytask)})
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(event='CALIBRATION_COMPLETED',config=config,train=result['train'],val=result['val']),indent=2),flush=True)

if __name__=='__main__':main()
