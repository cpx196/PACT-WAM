"""Check integrated OSC conditions against measured EEF trajectories; no training."""
import fcntl
import json
import os
import time
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from scripts.train_libero10_future_correction_reconstruct10 import ROOT,CACHE,MANIFEST,emit

OUT=ROOT/'artifacts/libero10_ac_measured_condition_diagnostic_20261007'
PREV=ROOT/'artifacts/libero10_jepa_action_ordering_diagnostic_single_20261007'
DATA=ROOT/'artifacts/libero10_guidance_backward_smoke_20260929/data/libero_10'
SOURCE=ROOT/'checkpoints/vjepa2-ac-libero90-euler-fixed-rot180-step2000-inference.pt'


def main():
    from experiments.libero.vjepa2_ac_ranker import VJEPA2ACRanker,LiberoACAdapter
    torch.set_num_threads(4);torch.manual_seed(239)
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'summary.json').exists() and not (OUT/'metrics.jsonl').exists()
    (OUT/'pid.txt').write_text(str(os.getpid())+'\n')
    rows=json.loads(MANIFEST.read_text());ids=json.loads((PREV/'config.json').read_text())['ids']
    assert len(ids)==100 and all(rows[i]['split']=='val' for i in ids)
    config=dict(ids=ids,windows=100,held_out_demos=50,checkpoint=str(SOURCE),dtype='float32',seed=239,
                frozen=True,training=False,rollouts=False,image_orientation='same cached rot180 main view',
                action_source='real executable demonstration actions, t:t+8',
                measured_states='HDF5 ee_states XYZ+rotvec converted to Euler XYZ at t,t+4,t+8',
                measured_motion='world XYZ difference, left-composed relative rotation converted to Euler XYZ',
                gripper='closedness=1-abs(left-right)/0.08, clipped0..1',
                arms=['adapter','measured_state_pose_only','measured_pose','measured_all'],
                primary_metric='autoregressive real-future latent L1',secondary_metric='latent MSE',
                caveat='measured future EEF conditions unavailable online; distribution differs from adapter-based finetuning; prior candidate expert gripper conversion was wrong, do not compare its baseline')
    (OUT/'config.json').write_text(json.dumps(config,indent=2));emit('CONFIG',**config)
    adapter=LiberoACAdapter(low_level_steps_per_ac_step=4)
    geometry=[]
    for n,index in enumerate(ids,1):
        row=rows[index];t=row['t']
        item=torch.load(CACHE/f'{index:05d}.pt',map_location='cpu',weights_only=False)
        assert item['index']==index and item['row']==row
        with h5py.File(DATA/row['task_file'],'r') as f:
            g=f['data'][row['demo']]
            ee=np.asarray(g['obs']['ee_states'][[t,t+4,t+8]],dtype=np.float64)
            fingers=np.asarray(g['obs']['gripper_states'][[t,t+4,t+8]],dtype=np.float64)
            actions=np.asarray(g['actions'][t:t+8],dtype=np.float32)
        assert np.array_equal(actions,item['expert_executable'].numpy())
        rotations=Rotation.from_rotvec(ee[:,3:6])
        closed=np.clip(1-np.abs(fingers[:,0]-fingers[:,1])/.08,0,1)
        states=np.concatenate([ee[:,:3],rotations.as_euler('xyz'),closed[:,None]],axis=1).astype(np.float32)
        ac=adapter.convert(actions[None],item['state'].numpy())
        torch_ac=adapter.convert_torch(torch.from_numpy(actions)[None],item['state'])
        assert np.allclose(ac.actions,torch_ac.actions.numpy(),atol=2e-5,rtol=2e-5)
        assert np.allclose(ac.states,torch_ac.states.numpy(),atol=2e-5,rtol=2e-5)
        assert np.allclose(states[0,:3],item['state'].numpy()[:3],atol=1e-6)
        initial_angle=(Rotation.from_euler('xyz',item['state'].numpy()[3:6])*rotations[0].inv()).magnitude()
        assert initial_angle<1e-5 and abs(float(states[0,6]-item['state'][6]))<1e-5
        actual_motion=np.concatenate([np.diff(ee[:,:3],axis=0),
            (rotations[1:]*rotations[:-1].inv()).as_euler('xyz'),np.diff(closed)[:,None]],axis=1).astype(np.float32)
        integrated=np.concatenate([ac.states[0],ac.terminal_states],axis=0)
        pos_err=np.linalg.norm(integrated[1:,:3]-states[1:,:3],axis=1)
        rot_err=(Rotation.from_euler('xyz',integrated[1:,3:6])*rotations[1:].inv()).magnitude()*180/np.pi
        record=dict(index=index,row=row,position_error_m_per_time=pos_err.tolist(),
                    orientation_error_deg_per_time=rot_err.tolist(),
                    gripper_closedness_error_per_time=np.abs(integrated[1:,6]-closed[1:]).tolist(),
                    true_translation_m_per_time=np.linalg.norm(np.diff(ee[:,:3],axis=0),axis=1).tolist(),
                    true_rotation_deg_per_time=(rotations[1:]*rotations[:-1].inv()).magnitude().tolist(),
                    adapter_actions=ac.actions[0].tolist(),adapter_states=ac.states[0].tolist(),
                    measured_actions=actual_motion.tolist(),measured_states=states[:-1].tolist())
        # Rename/convert the explicit angle magnitude from radians to degrees.
        record['true_rotation_deg_per_time']=(np.asarray(record['true_rotation_deg_per_time'])*180/np.pi).tolist()
        geometry.append(record)
        if n%25==0:emit('GEOMETRY_PROGRESS',completed=n,total=len(ids))
    (OUT/'geometry.json').write_text(json.dumps(geometry,indent=2))
    geometry_summary={}
    for key in ['position_error_m_per_time','orientation_error_deg_per_time','gripper_closedness_error_per_time']:
        a=np.array([r[key] for r in geometry])
        geometry_summary[key]=dict(mean=a.mean(0).tolist(),median=np.median(a,axis=0).tolist(),p95=np.quantile(a,.95,axis=0).tolist())
    emit('GEOMETRY_COMPLETED',**geometry_summary)
    ranker=VJEPA2ACRanker(checkpoint_path=str(SOURCE),vjepa2_repo='/data/pxchen/vjepa2',device='cpu',dtype=torch.float32,max_ac_steps=2)
    del ranker.encoder;predictor=ranker.predictor.cuda().eval().requires_grad_(False)
    emit('PREDICTOR_READY')
    records=[];started=time.monotonic()
    for n,record in enumerate(geometry,1):
        index=record['index'];item=torch.load(CACHE/f'{index:05d}.pt',map_location='cpu',weights_only=False)
        base_a=torch.tensor(record['adapter_actions'],device='cuda')[None]
        base_s=torch.tensor(record['adapter_states'],device='cuda')[None]
        real_a=torch.tensor(record['measured_actions'],device='cuda')[None]
        real_s=torch.tensor(record['measured_states'],device='cuda')[None]
        pose_a=base_a.clone();pose_a[:,:,:6]=real_a[:,:,:6]
        pose_s=base_s.clone();pose_s[:,:,:6]=real_s[:,:,:6]
        arms={'adapter':(base_a,base_s),'measured_state_pose_only':(base_a,pose_s),
              'measured_pose':(pose_a,pose_s),'measured_all':(real_a,real_s)}
        metrics={}
        with torch.no_grad():
            for name,(a,s) in arms.items():
                history=item['current'].cuda()[None];future=[]
                for k in range(2):
                    p=predictor(history,a[:,:k+1],s[:,:k+1])
                    p=F.layer_norm(p[:,-256:],(1408,));future.append(p[0]);history=torch.cat((history,p),1)
                p=torch.stack(future);diff=p-item['real_future'].cuda()
                l1=diff.abs().mean((1,2));mse=diff.square().mean((1,2))
                metrics[name]=dict(l1=float(l1.mean()),mse=float(mse.mean()),l1_per_time=l1.cpu().tolist(),mse_per_time=mse.cpu().tolist())
        output=dict(index=index,row=record['row'],metrics=metrics)
        records.append(output)
        with (OUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(output)+'\n')
        if n==1:emit('SMOKE_PASSED',executable_expert_gripper_used=True,numpy_torch_adapter_match=True,metrics=metrics,peak_gib=torch.cuda.max_memory_allocated()/2**30)
        if n%10==0 or n==len(ids):emit('PROGRESS',completed=n,total=len(ids),seconds=time.monotonic()-started)
    def aggregate(subset):
        base=np.array([r['metrics']['adapter']['l1'] for r in subset]);result={}
        for arm in config['arms']:
            values=np.array([r['metrics'][arm]['l1'] for r in subset])
            demo={}
            for r,v,b in zip(subset,values,base):demo.setdefault((r['row']['task_file'],r['row']['demo']),[]).append(b-v)
            differences=np.array([np.mean(v) for v in demo.values()])
            boot=np.random.default_rng(239).choice(differences,size=(2000,len(differences)),replace=True).mean(1)
            result[arm]=dict(mean_l1=float(values.mean()),mean_mse=float(np.mean([r['metrics'][arm]['mse'] for r in subset])),
                l1_per_time=np.mean([r['metrics'][arm]['l1_per_time'] for r in subset],axis=0).tolist(),
                fraction_better_than_adapter=float(np.mean(values<base)),mean_l1_reduction=float((base-values).mean()),
                demo_bootstrap_reduction_95ci=np.quantile(boot,[.025,.975]).tolist())
        return result
    summary=dict(config=config,geometry=geometry_summary,overall=aggregate(records),
                 per_task={task:aggregate([r for r in records if r['row']['task_file']==task]) for task in sorted({r['row']['task_file'] for r in records})},
                 seconds=time.monotonic()-started)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));emit('ALL_COMPLETED',geometry=geometry_summary,overall=summary['overall'],seconds=summary['seconds'])

if __name__=='__main__':main()
