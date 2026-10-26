"""Update only Predictor latents; reuse frozen current/WAM/real Encoder cache."""
import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scripts.train_libero10_future_correction_reconstruct10 import CACHE, MANIFEST, emit, save
from experiments.libero.vjepa2_ac_ranker import LiberoACAdapter

ROOT=Path('/data/pxchen/PACT-WAM')
SOURCE=ROOT/'artifacts/jepa_libero10_response_calibrated_rot180_20261007/train_2000/step_002000.pt'
CAL=ROOT/'artifacts/libero_ac_command_response_calibration_20261007/calibration.json'

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--batch',type=int,default=8);args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(239)
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'tokens').mkdir(exist_ok=True)
    lock=(args.out/f'shard{args.shard}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert 0<=args.shard<args.shards
    rows=json.loads(MANIFEST.read_text());cal=json.loads(CAL.read_text())
    saved=torch.load(SOURCE,map_location='cpu',weights_only=False,mmap=True)
    assert saved['step']==2000 and saved['config']['calibration_config']==cal
    assert saved['config']['data_format_version']=='libero_ac_euler_response_calibrated_v1'
    assert saved['config']['image_rotation_degrees']==180
    metadata=dict(checkpoint=str(SOURCE),checkpoint_step=2000,calibration=cal,
                  manifest=str(MANIFEST),source_cache=str(CACHE),dtype='float32',
                  training=False,encoder_reencoded=False,main_view_rotation=180,
                  cached_field='predicted_future only',batch=args.batch)
    config_path=args.out/'config.json'
    if config_path.exists():assert json.loads(config_path.read_text())==metadata
    else:config_path.write_text(json.dumps(metadata,indent=2))
    sys.path.insert(0,'/data/pxchen/vjepa2')
    from src.hub.backbones import vjepa2_ac_vit_giant
    encoder,predictor=vjepa2_ac_vit_giant(pretrained=False,num_frames=6);del encoder
    predictor.load_state_dict(saved['predictor'],strict=True);del saved
    predictor.cuda().eval().requires_grad_(False);predictor.use_activation_checkpointing=False
    adapter=LiberoACAdapter(osc_position_scale=.05*cal['translation_response_gain'],
                           osc_rotation_scale=.5*cal['rotation_response_gain'])
    ids=list(range(args.shard,len(rows),args.shards));done=0;started=time.monotonic()
    for start in range(0,len(ids),args.batch):
        pending=[i for i in ids[start:start+args.batch] if not (args.out/'tokens'/f'{i:05d}.pt').exists()]
        if not pending:continue
        items=[torch.load(CACHE/f'{i:05d}.pt',map_location='cpu',weights_only=False,mmap=True) for i in pending]
        acs=[]
        for i,item in zip(pending,items):
            assert item['index']==i and item['row']==rows[i]
            raw=(item['action']-item['offset'])/item['scale']
            executable=torch.cat((raw[:,:6],1-2*raw[:,6:7]),-1)
            ac=adapter.convert(executable.numpy(),item['state'].numpy());acs.append(ac)
        actions=torch.tensor(np.concatenate([a.actions for a in acs])).cuda()
        states=torch.tensor(np.concatenate([a.states for a in acs])).cuda()
        history=torch.stack([item['current'] for item in items]).float().cuda()
        with torch.no_grad():
            predictions=[]
            for k in range(2):
                latent=predictor(history,actions[:,:k+1],states[:,:k+1])
                latent=F.layer_norm(latent[:,-256:],(1408,));predictions.append(latent)
                history=torch.cat((history,latent),1)
            prediction=torch.stack(predictions,1).cpu()
        assert prediction.shape==(len(pending),2,256,1408) and torch.isfinite(prediction).all()
        if done==0:
            # Batch-shape drift must remain small, compared with a batch1 path.
            with torch.no_grad():
                h=items[0]['current'][None].float().cuda();single=[]
                for k in range(2):
                    latent=predictor(h,actions[:1,:k+1],states[:1,:k+1])
                    latent=F.layer_norm(latent[:,-256:],(1408,));single.append(latent[0]);h=torch.cat((h,latent),1)
                diff=(torch.stack(single).cpu()-prediction[0]).abs()
            assert float(diff.mean())<2e-5 and float(diff.max())<1e-3
            emit('CACHE_SMOKE_PASSED',batch1_difference_mean=float(diff.mean()),batch1_difference_max=float(diff.max()))
        for j,i in enumerate(pending):
            save(dict(index=i,row=rows[i],predicted_future=prediction[j].clone(),
                      checkpoint=str(SOURCE),calibration=cal),args.out/'tokens'/f'{i:05d}.pt')
        done+=len(pending)
        if done<=args.batch or done%128<args.batch or start+args.batch>=len(ids):
            emit('CACHE_PROGRESS',shard=args.shard,generated=done,total=len(ids),
                 seconds=time.monotonic()-started,seconds_per_window=(time.monotonic()-started)/done)
    assert all((args.out/'tokens'/f'{i:05d}.pt').exists() for i in ids)
    (args.out/f'shard{args.shard}_complete.json').write_text(json.dumps(dict(completed=len(ids),metadata=metadata)))
    emit('CACHE_COMPLETED',shard=args.shard,windows=len(ids),seconds=time.monotonic()-started)

if __name__=='__main__':main()
