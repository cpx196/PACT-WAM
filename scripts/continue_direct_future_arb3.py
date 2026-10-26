"""Continue both matched ARBs from pilot step100 for three full epochs.

Restore model AND AdamW moments/counters, change ONLY LR to1e-5, expand training
coverage as authorized. Preserve old pilot outputs. No validation early stop.
"""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pilot_direct_future_arb100 import (
    ROOT, CACHE, PCACHE, MANIFEST, DirectFutureARB, atomic_json, emit, gradient_norm)

SOURCE = ROOT/'artifacts/libero10_direct_future_arb_pilot100_20261008'


def source_checkpoint(source, variant):
    path = source/variant/'step100.pt'
    ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    assert ck['variant']==variant and ck['step']==100 and ck['next_batch']==100
    c = ck['config']
    assert c['parameters']==58015238 and c['batch']==8 and c['lr']==1e-4
    assert c['layers']==6 and c['hidden']==768 and c['heads']==12 and c['ffn']==3072
    assert not c['idm'] and not c['la'] and not c['feedback'] and not c['real_future_input']
    assert c['predictor_metadata']['checkpoint_step']==2000
    assert len(ck['optimizer']['state'])>0 and ck['optimizer']['param_groups'][0]['lr']==1e-4
    for state in ck['optimizer']['state'].values():
        assert float(state['step'])==100 and torch.isfinite(state['exp_avg']).all()
        assert torch.isfinite(state['exp_avg_sq']).all()
    assert 'torch_rng' in ck and 'cuda_rng' in ck and 'numpy_rng' in ck
    return path, ck


def load_data():
    rows=json.loads(MANIFEST.read_text())
    tr=[i for i,r in enumerate(rows) if r['split']=='train']
    va=[i for i,r in enumerate(rows) if r['split']=='val']
    assert (len(tr),len(va))==(3871,436)
    assert {(rows[i]['task_file'],rows[i]['demo']) for i in tr}.isdisjoint(
        {(rows[i]['task_file'],rows[i]['demo']) for i in va})
    meta=json.loads((PCACHE/'config.json').read_text())
    assert meta['checkpoint_step']==2000 and meta['calibration']['fitted_split']=='train_only'
    assert meta['manifest']==str(MANIFEST) and meta['main_view_rotation']==180
    z=torch.empty(len(rows),2,2,256,1408,dtype=torch.float32)
    actions=[]; targets=[]; started=time.monotonic()
    for i,row in enumerate(rows):
        item=torch.load(CACHE/f'{i:05d}.pt',map_location='cpu',weights_only=False,mmap=True)
        updated=torch.load(PCACHE/'tokens'/f'{i:05d}.pt',map_location='cpu',weights_only=False,mmap=True)
        assert item['index']==updated['index']==i and item['row']==updated['row']==row
        assert updated['checkpoint']==meta['checkpoint'] and updated['calibration']==meta['calibration']
        feature=torch.stack((updated['predicted_future'],item['wam_future']),1).float()
        assert feature.shape==(2,2,256,1408) and torch.isfinite(feature).all()
        z[i].copy_(feature)
        a=item['action'].float(); y=item['expert_normalized_first6'].float()
        assert a.shape==(8,7) and y.shape==(8,6) and torch.isfinite(a).all() and torch.isfinite(y).all()
        assert torch.allclose(y,(item['expert_executable']*item['scale']+item['offset'])[:,:6],
                              atol=1e-6,rtol=1e-6)
        actions.append(a); targets.append(y)
        if (i+1)%512==0:
            emit('CACHE_LOAD_PROGRESS',loaded=i+1,total=len(rows),seconds=time.monotonic()-started)
    a=torch.stack(actions).cuda(); y=torch.stack(targets).cuda()
    probe=np.asarray(tr)[np.linspace(0,len(tr)-1,256,dtype=int)].tolist()
    val_groups={}; shuffle={}
    for i in va:
        val_groups.setdefault(rows[i]['task_file'],[]).append(i)
    for group in val_groups.values():
        assert len(group)>1
        shuffle.update(dict(zip(group,group[1:]+group[:1])))
    emit('DATA_READY',train_windows=len(tr),val_windows=len(va),load_seconds=time.monotonic()-started)
    return rows,tr,va,probe,shuffle,z,a,y,meta


def run_arm(variant,args,data,config):
    rows,tr,va,probe,shuffle,z,a,y,meta=data
    include=variant=='full_future'
    path,ck=source_checkpoint(args.source,variant)
    assert ck['config']['predictor_metadata']==meta
    assert ck['config']['val_ids']==va and ck['config']['train_probe_ids']==probe
    mean=a[tr].mean(0).cpu(); std=a[tr].std(0).clamp_min(.01).cpu()
    assert torch.allclose(ck['model']['mean'],mean,atol=2e-7,rtol=1e-6)
    assert torch.allclose(ck['model']['std'],std,atol=2e-7,rtol=1e-6)
    model=DirectFutureARB(mean,std).cuda(); model.load_state_dict(ck['model'])
    assert sum(p.numel() for p in model.parameters())==58015238
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    optimizer.load_state_dict(ck['optimizer'])
    # Check nonzero restored Adam moments, unchanged tensors and counters.
    parameter=model.visual_project.weight
    parameter_index=next(i for i,p in enumerate(model.parameters()) if p is parameter)
    old_id=ck['optimizer']['param_groups'][0]['params'][parameter_index]
    old_state=ck['optimizer']['state'][old_id]
    restored=optimizer.state[parameter]
    assert torch.equal(restored['exp_avg'].cpu(),old_state['exp_avg'])
    assert torch.equal(restored['exp_avg_sq'].cpu(),old_state['exp_avg_sq'])
    assert float(restored['step'])==100 and float(restored['exp_avg'].abs().sum())>0
    for group in optimizer.param_groups:
        assert group['lr']==1e-4 and group['weight_decay']==.01
        group['lr']=1e-5
    torch.set_rng_state(ck['torch_rng'])
    torch.cuda.set_rng_state_all(ck['cuda_rng'])
    np.random.set_state(ck['numpy_rng'])
    del ck,old_state,restored
    dest=args.out/variant; dest.mkdir()
    first_project=model.visual_project.weight.detach().clone()
    first_cross=model.blocks[0].cross_attention.in_proj_weight.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic(); history=[]; epoch_results=[]

    def predict(ix,mode='normal'):
        visual=z[ix].cuda()
        if mode=='shuffle_predictor':
            visual[:,:,0]=z[[shuffle[i] for i in ix],:,0].cuda()
        return model(a[ix],visual,include_predictor=include)

    @torch.no_grad()
    def evaluate(ix,mode='normal'):
        model.eval(); errors=[]
        for start in range(0,len(ix),8):
            batch=ix[start:start+8]
            errors.append((predict(batch,mode)-y[batch]).square().mean((1,2)).cpu())
        return float(torch.cat(errors).mean())

    def save(name,epoch,total_step,next_batch):
        temporary=dest/(name+'.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            variant=variant,source_checkpoint=str(path),config=config,epoch=epoch,
            step=total_step,next_batch=next_batch,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state()),temporary)
        temporary.replace(dest/name)

    initial=dict(train_mse=evaluate(tr),train_probe_mse=evaluate(probe),val_mse=evaluate(va),
        raw_train_mse=float((a[tr,:,:6]-y[tr]).square().mean()),
        raw_val_mse=float((a[va,:,:6]-y[va]).square().mean()))
    old_result=json.loads((args.source/variant/'summary.json').read_text())
    assert abs(initial['val_mse']-old_result['final']['val_mse'])<2e-7
    atomic_json(initial,dest/'initial.json'); emit('RESUME_VERIFIED',variant=variant,
        source=str(path),optimizer_restored=True,optimizer_step=100,lr=1e-5,**initial)
    best=initial['val_mse']; best_epoch=0; best_step=100
    save('best.pt',0,100,0)
    for epoch in range(1,4):
        order=np.random.default_rng(239+epoch).permutation(tr).tolist()
        seen=[]; losses=[]; epoch_start=time.monotonic()
        for batch_no,start in enumerate(range(0,len(order),8),1):
            ix=order[start:start+8]; seen.extend(ix)
            continued_step=(epoch-1)*484+batch_no; total_step=100+continued_step
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss=F.mse_loss(predict(ix),y[ix]); assert torch.isfinite(loss)
            loss.backward()
            visual_g=gradient_norm(model.visual_project.parameters())
            cross_g=gradient_norm(model.blocks[0].cross_attention.parameters())
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
            optimizer.step(); losses.append((float(loss),len(ix)))
            if continued_step==3:
                assert visual_g>0 and cross_g>0
                assert not torch.equal(model.visual_project.weight,first_project)
                assert not torch.equal(model.blocks[0].cross_attention.in_proj_weight,first_cross)
                smoke=dict(passed=True,source_step=100,cumulative_step=103,optimizer_restored=True,
                    visual_grad_norm=visual_g,cross_grad_norm=cross_g,finite=True,
                    peak_gib=torch.cuda.max_memory_allocated()/2**30)
                atomic_json(smoke,dest/'smoke.json'); emit('SMOKE_PASSED',variant=variant,**smoke)
            record=dict(variant=variant,epoch=epoch,batch=batch_no,continued_step=continued_step,
                step=total_step,batch_mse=float(loss),visual_grad_norm=visual_g,cross_grad_norm=cross_g,
                global_grad_norm=grad,clipped=grad>1,lr=optimizer.param_groups[0]['lr'],
                peak_gib=torch.cuda.max_memory_allocated()/2**30)
            if continued_step==3 or continued_step%100==0:
                record.update(train_probe_mse=evaluate(probe),val_mse=evaluate(va),
                              elapsed=time.monotonic()-started)
                emit('EVALUATION',**record)
            history.append(record)
            with (args.out/'metrics.jsonl').open('a') as log:
                log.write(json.dumps(record)+'\n')
        assert sorted(seen)==tr and len(seen)==3871
        result=dict(epoch=epoch,step=total_step,train_mse=evaluate(tr),
            train_probe_mse=evaluate(probe),val_mse=evaluate(va),
            epoch_online_batch_mse=sum(l*n for l,n in losses)/3871,
            epoch_seconds=time.monotonic()-epoch_start,elapsed=time.monotonic()-started)
        epoch_results.append(result)
        atomic_json(epoch_results,dest/'epochs.json')
        if result['val_mse']<best:
            best=result['val_mse']; best_epoch=epoch; best_step=total_step
            save('best.pt',epoch,total_step,0)
        save(f'epoch{epoch:03d}.pt',epoch,total_step,0)
        # Last is a hardlink to immutable epoch checkpoint, atomically replaced.
        temporary=dest/'last.tmp'
        os.link(dest/f'epoch{epoch:03d}.pt',temporary)
        temporary.replace(dest/'last.pt')
        saved=torch.load(dest/f'epoch{epoch:03d}.pt',map_location='cpu',weights_only=False,mmap=True)
        assert saved['epoch']==epoch and saved['step']==100+484*epoch
        assert all(float(s['step'])==saved['step'] for s in saved['optimizer']['state'].values())
        del saved
        emit('EPOCH_COMPLETED',variant=variant,**result)
    final=dict(train_mse=evaluate(tr),train_probe_mse=evaluate(probe),val_mse=evaluate(va),
        shuffled_predictor_val_mse=evaluate(va,'shuffle_predictor') if include else None)
    saved=torch.load(dest/'last.pt',map_location='cpu',weights_only=False,mmap=True)
    model.eval()
    with torch.no_grad():
        reference=predict(va[:2]).cpu()
    clone=DirectFutureARB(mean,std).cuda(); clone.load_state_dict(saved['model']); clone.eval()
    with torch.no_grad():
        output=clone(a[va[:2]],z[va[:2]].cuda(),include).cpu()
    assert torch.equal(reference,output)
    del saved,clone
    summary=dict(initial=initial,final=final,epochs=epoch_results,best_val_mse=best,
        best_epoch=best_epoch,best_step=best_step,source_checkpoint=str(path),
        cumulative_steps=1552,additional_epochs=3,additional_steps=1452,
        seconds=time.monotonic()-started,optimizer_restored=True,checkpoint_roundtrip=True,
        clip_rate=float(np.mean([r['clipped'] for r in history])),
        visual_nonzero_grad_steps=sum(r['visual_grad_norm']>0 for r in history),
        visual_delta_rms=float((model.visual_project.weight-first_project).square().mean().sqrt()),
        cross_delta_rms=float((model.blocks[0].cross_attention.in_proj_weight-first_cross).square().mean().sqrt()),
        peak_gib=max(r['peak_gib'] for r in history))
    atomic_json(summary,dest/'summary.json'); emit('ARM_COMPLETED',variant=variant,**summary)
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--source',type=Path,default=SOURCE)
    ap.add_argument('--cpu-preflight',action='store_true')
    args=ap.parse_args(); torch.set_num_threads(4)
    for variant in ('full_future','no_predictor'):
        path,ck=source_checkpoint(args.source,variant)
        emit('SOURCE_CHECKPOINT_VERIFIED',variant=variant,source=str(path),step=100,
             optimizer_state=True,parameters=ck['config']['parameters'])
        del ck
    if args.cpu_preflight:
        emit('CPU_PREFLIGHT_PASSED'); return
    args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'training.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (args.out/'metrics.jsonl').exists() and not (args.out/'summary.json').exists()
    (args.out/'train.pid').write_text(str(os.getpid()))
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    data=load_data(); rows,tr,va,probe,shuffle,z,a,y,meta=data
    source_config=json.loads((args.source/'config.json').read_text())
    config=dict(source_config=source_config,source=str(args.source),parameters=58015238,
        additional_epochs=3,steps_per_epoch=484,additional_steps_per_arm=1452,
        source_step=100,final_step=1552,batch=8,lr=1e-5,weight_decay=.01,clip_norm=1.,
        precision='FP32; TF32 disabled',dropout=0.,optimizer='AdamW restored, not reset',
        unchanged='architecture, caches, normalization, loss, batch, dropout, WD, clip, validation',
        train_ids=tr,val_ids=va,train_probe_ids=probe,train_windows=3871,val_windows=436,
        data_seed=239,epoch_shuffle_seeds=[240,241,242],early_stopping=False,
        supervision='expert normalized executable first8x6 action MSE ONLY; gripper untouched',
        predictor_metadata=meta,feedback=False,idm=False,real_future_input=False,
        evaluation='full train/val at start and each epoch; fixed256train/full436val every100updates',
        best_selection='epoch endpoints including initial; report fixed final separately',
        save='every epoch: model, optimizer, RNG, last, best',
        caveat='offline action MSE; no robot rollout; tests more data/updates and lower LR together, not causal attribution')
    atomic_json(config,args.out/'config.json')
    atomic_json([dict(index=i,**r) for i,r in enumerate(rows)],args.out/'window_manifest.json')
    results={}
    for variant in ('full_future','no_predictor'):
        results[variant]=run_arm(variant,args,data,config)
        torch.cuda.empty_cache()
    atomic_json(dict(config=config,results=results),args.out/'summary.json')
    emit('CONTINUATION_COMPLETED')


if __name__=='__main__':
    main()
