"""Matched ten-epoch joint/zero-LA runs from separate epoch20 checkpoints."""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scripts.train_libero10_future_correction_reconstruct10 import Model, CACHE, MANIFEST, emit, save
from scripts.train_libero10_transformer_la450 import Corrector

ROOT=Path('/data/pxchen/PACT-WAM')
OUT=ROOT/'artifacts/libero10_idm_arb_joint5_20261007'
SOURCE=ROOT/'artifacts/libero10_arb_reconstruction_idm20_20261007'
IDM=ROOT/'artifacts/libero10_future_correction_reconstruct40_20261007/epoch020.pt'


def norm(parameters):
    gradients=[p.grad.square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(gradients).sum().sqrt()) if gradients else 0.


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--predictor-cache',type=Path,required=True)
    parser.add_argument('--variant',choices=['joint','zero_la'],required=True)
    parser.add_argument('--epochs',type=int,default=10)
    args=parser.parse_args();OUT=args.out
    torch.set_num_threads(4); torch.manual_seed(239)
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'launcher.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'summary.json').exists() and not (OUT/'metrics.jsonl').exists()
    (OUT/'pid.txt').write_text(str(os.getpid())+'\n')
    rows=json.loads(MANIFEST.read_text())
    train=[i for i,r in enumerate(rows) if r['split']=='train']
    val=[i for i,r in enumerate(rows) if r['split']=='val']
    assert (len(rows),len(train),len(val))==(4307,3871,436)
    assert {(rows[i]['task_file'],rows[i]['demo']) for i in train}.isdisjoint(
        {(rows[i]['task_file'],rows[i]['demo']) for i in val})
    probe=np.asarray(train)[np.linspace(0,len(train)-1,256,dtype=int)].tolist()
    metadata=json.loads((args.predictor_cache/'config.json').read_text())
    assert metadata['checkpoint_step']==2000 and metadata['calibration']['fitted_split']=='train_only'
    assert metadata['manifest']==str(MANIFEST) and metadata['source_cache']==str(CACHE)
    z=torch.empty(len(rows),2,3,256,1408) if args.variant=='joint' else None
    actions=[]; targets=[]
    started=time.monotonic()
    for i,row in enumerate(rows):
        item=torch.load(CACHE/f'{i:05d}.pt',map_location='cpu',weights_only=False,mmap=True)
        assert item['row']==row and item['index']==i
        if z is not None:
            updated=torch.load(args.predictor_cache/'tokens'/f'{i:05d}.pt',map_location='cpu',weights_only=False,mmap=True)
            assert updated['index']==i and updated['row']==row
            assert updated['checkpoint']==metadata['checkpoint'] and updated['calibration']==metadata['calibration']
            for j,key in enumerate(['predicted_future','wam_future','real_future']):
                f=(updated[key] if j==0 else item[key]).float()
                assert f.shape==(2,256,1408) and torch.isfinite(f).all()
                z[i,:,j].copy_(f)
        a,y=item['action'].float(),item['expert_normalized_first6'].float()
        assert a.shape==(8,7) and y.shape==(8,6)
        expected=(item['expert_executable']*item['scale']+item['offset'])[:,:6]
        assert torch.allclose(y,expected,atol=1e-6,rtol=1e-6)
        assert torch.isfinite(a).all() and torch.isfinite(y).all()
        actions.append(a);targets.append(y)
        if (i+1)%512==0 or i+1==len(rows):
            emit('DATA_LOAD_PROGRESS',loaded=i+1,seconds=time.monotonic()-started)
    actions=torch.stack(actions).cuda();targets=torch.stack(targets).cuda()
    config=dict(epochs=args.epochs,start_arb_epoch=20,no_early_stopping=True,train=len(train),val=len(val),
                stride=32,batch=8,seed=239,dtype='float32',idm_lr=1e-5,arb_lr=1e-4,
                weight_decay=.01,grad_clip=1.,reconstruction_weight=.01,kl_weight=1e-5,
                decoder_frozen=True,decoder_eval=True,gradient_through_decoder=True,
                encoder_frozen=True,predictor_frozen=True,wam_frozen=True,
                code='posterior mean mu, deterministic; no sampling',
                input='matching-time Predictor and WAM future patches + raw first8x7 WAM actions',
                action_target='expert normalized first8x6; gripper unchanged',
                true_future='reconstruction label only, never input',
                idm_checkpoint=str(IDM),arb_source=str(SOURCE/'correction_code/epoch020.pt'),manifest=str(MANIFEST),
                predictor_metadata=metadata,matched_arb_initialization=True,matched_batch_order=True,
                reconstruction='MSE through frozen Decoder back to IDM/posterior',
                variants=['joint','zero_la'],evaluation='offline action error; not rollout success',
                caveat='both arms load the SAME separately trained epoch20 ARB; no joint5 weights; not rollout success')
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    emit('CONFIG',**config)
    def batch(ix):
        if z is None:return None,None,actions[ix],targets[ix]
        f=z[ix].cuda().flatten(0,1)
        return f[:,:2],f[:,2],actions[ix],targets[ix]
    results={}
    for variant in [args.variant]:
        torch.manual_seed(239)
        dest=OUT
        ck=torch.load(SOURCE/'correction_code/epoch020.pt',map_location='cpu',weights_only=False)
        assert ck['epoch']==20 and ck['step']==9680
        assert ck['config']['frozen_idm'] and ck['config']['future_source']=='correction'
        arb=Corrector(ck['model']['mean'],ck['model']['std']).cuda()
        arb.load_state_dict(ck['model'],strict=True)
        arb_opt=torch.optim.AdamW(arb.parameters(),lr=1e-4,weight_decay=.01)
        arb_opt.load_state_dict(ck['optimizer'])
        assert all(g['lr']==1e-4 for g in arb_opt.param_groups)
        del ck
        future=None;idm_opt=None;idm_params=[]
        if variant=='joint':
            future=Model().cuda()
            ck=torch.load(IDM,map_location='cpu',weights_only=False)
            assert ck['epoch']==20 and ck['config']['source_order']==['predictor','wam']
            future.load_state_dict(ck['model'],strict=True);del ck
            future.decoder.eval().requires_grad_(False)
            idm_params=list(future.idm.parameters())+list(future.posterior.parameters())
            idm_opt=torch.optim.AdamW(idm_params,lr=1e-5,weight_decay=.01)
        @torch.no_grad()
        def evaluate(ids,shuffle=False):
            arb.eval()
            if future is not None: future.eval()
            errors=[];base=[];rms=[];cos=[];recon=[];codes=[]
            permutation=np.random.default_rng(239).permutation(ids).tolist()
            for start in range(0,len(ids),8):
                ix=ids[start:start+8]
                pairs,truth,a,y=batch(ix)
                code=a.new_zeros(len(ix),2,128)
                if future is not None:
                    mu,_=future.encode(pairs)
                    if shuffle:
                        other,_,_,_=batch(permutation[start:start+len(ix)])
                        mu=future.encode(other)[0]
                    code=mu.reshape(len(ix),2,128)
                    prediction=future.decoder(pairs[:,0],mu)
                    recon.append((prediction-truth).square().mean((1,2)).cpu())
                    codes.append(code.cpu())
                pred,delta=arb(a,code)
                errors.append((pred-y).square().mean((1,2)).cpu())
                base.append((a[:,:,:6]-y).square().mean((1,2)).cpu())
                rms.append(delta.square().mean((1,2)).sqrt().cpu())
                cos.append(F.cosine_similarity(delta.flatten(1),(y-a[:,:,:6]).flatten(1),dim=1).cpu())
            e,b=torch.cat(errors),torch.cat(base)
            return dict(action_mse=float(e.mean()),baseline_mse=float(b.mean()),
                        better=int((e<b).sum()),worse=int((e>b).sum()),
                        correction_rms=float(torch.cat(rms).mean()),correction_cosine=float(torch.cat(cos).mean()),
                        reconstruction_mse=float(torch.cat(recon).mean()) if recon else None,
                        code_std=float(torch.cat(codes).flatten(1).std(0).mean()) if codes else None)
        initial=dict(val=evaluate(val),train_probe=evaluate(probe))
        assert np.isfinite(initial['val']['action_mse'])
        (dest/'initial.json').write_text(json.dumps(initial,indent=2));emit('INITIAL',variant=variant,**initial)
        # Pure graph diagnostic: no optimizer step, no checkpoint modification.
        arb.train()
        if future is not None:
            future.train();future.decoder.eval()
        pairs,truth,a,y=batch(train[:8])
        code=a.new_zeros(len(a),2,128)
        if future is not None:
            mu,logvar=future.encode(pairs);code=mu.reshape(len(a),2,128)
        pred,delta=arb(a,code);action_loss=F.mse_loss(pred,y)
        smoke=dict(action_mse=float(action_loss))
        if future is not None:
            grads=torch.autograd.grad(action_loss,idm_params,retain_graph=True,allow_unused=True)
            smoke['action_only_idm_grad']=float(torch.stack([g.square().sum() for g in grads if g is not None]).sum().sqrt())
            reconstruction=F.mse_loss(future.decoder(pairs[:,0],mu),truth)
            grads=torch.autograd.grad(reconstruction,idm_params,allow_unused=True)
            smoke['reconstruction_only_idm_grad']=float(torch.stack([g.square().sum() for g in grads if g is not None]).sum().sqrt())
            smoke['weighted_reconstruction_to_action_grad_ratio']=.01*smoke['reconstruction_only_idm_grad']/max(smoke['action_only_idm_grad'],1e-12)
            assert all(np.isfinite(smoke[k]) and smoke[k]>0 for k in ['action_only_idm_grad','reconstruction_only_idm_grad'])
            assert all(not p.requires_grad and p.grad is None for p in future.decoder.parameters())
        else:
            grads=torch.autograd.grad(action_loss,arb.parameters(),allow_unused=True)
            smoke['arb_grad']=float(torch.stack([g.square().sum() for g in grads if g is not None]).sum().sqrt())
            assert smoke['arb_grad']>0 and np.isfinite(smoke['arb_grad'])
        (dest/'smoke.json').write_text(json.dumps(smoke,indent=2));emit('SMOKE_PASSED',variant=variant,**smoke)
        del pairs,truth,a,y,pred,delta,action_loss,grads
        torch.manual_seed(239)
        params=list(arb.parameters())+idm_params
        best=float('inf');best_epoch=0;step=9680;started=time.monotonic()
        def checkpoint(name,epoch):
            save(dict(arb=arb.state_dict(),future=future.state_dict() if future is not None else None,
                      arb_optimizer=arb_opt.state_dict(),idm_optimizer=idm_opt.state_dict() if idm_opt else None,
                      config=config,variant=variant,epoch=epoch,additional_epoch=epoch-20,step=step,
                      best_val=best,best_epoch=best_epoch,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),dest/name)
        for epoch in range(21,21+args.epochs):
            arb.train()
            if future is not None: future.train();future.decoder.eval()
            ids=np.random.default_rng(238+epoch).permutation(train).tolist()
            sums=np.zeros(3);count=0;norms=[];idm_norms=[];epoch_started=time.monotonic()
            for start in range(0,len(ids),8):
                ix=ids[start:start+8];arb_opt.zero_grad(set_to_none=True)
                if idm_opt is not None: idm_opt.zero_grad(set_to_none=True)
                pairs,truth,a,y=batch(ix)
                code=a.new_zeros(len(ix),2,128);rec=a.new_zeros(());kl=a.new_zeros(())
                if future is not None:
                    mu,logvar=future.encode(pairs);code=mu.reshape(len(ix),2,128)
                    rec=F.mse_loss(future.decoder(pairs[:,0],mu),truth)
                    kl=.5*(mu.square()+logvar.exp()-1-logvar).mean()
                pred,delta=arb(a,code);act=F.mse_loss(pred,y)
                loss=act+.01*rec+1e-5*kl
                assert torch.isfinite(loss)
                loss.backward();idm_norms.append(norm(idm_params))
                total=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                norms.append(float(total));arb_opt.step()
                if idm_opt is not None: idm_opt.step()
                sums+=np.asarray([float(act),float(rec),float(loss)])*len(ix);count+=len(ix);step+=1
                if len(norms)%100==0:
                    emit('TRAIN_PROGRESS',variant=variant,epoch=epoch,batch=len(norms),step=step,
                         action_mse=sums[0]/count,reconstruction_mse=sums[1]/count,
                         seconds_per_batch=(time.monotonic()-epoch_started)/len(norms))
            row=dict(epoch=epoch,additional_epoch=epoch-20,step=step,optimization_action_mse=sums[0]/count,
                     optimization_reconstruction_mse=sums[1]/count,optimization_total_loss=sums[2]/count,
                     val=evaluate(val),train_probe=evaluate(probe),clip_rate=float(np.mean(np.asarray(norms)>1)),
                     grad_norm_p50=float(np.median(norms)),idm_grad_norm_p50=float(np.median(idm_norms)),
                     peak_gib=torch.cuda.max_memory_allocated()/2**30)
            if row['val']['action_mse']<best:
                best,best_epoch=row['val']['action_mse'],epoch;checkpoint('best.pt',epoch)
            checkpoint('last.pt',epoch);checkpoint(f'epoch{epoch:03d}.pt',epoch)
            row.update(epoch_seconds=time.monotonic()-epoch_started,seconds=time.monotonic()-started)
            with (dest/'metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            emit('EPOCH_COMPLETED',variant=variant,**row)
        results[variant]=dict(initial=initial,final=row,best_val_mse=best,best_epoch=best_epoch,
                              shuffled_final=evaluate(val,shuffle=True) if future is not None else None)
        (dest/'summary.json').write_text(json.dumps(results[variant],indent=2))
        emit('VARIANT_COMPLETED',variant=variant,**results[variant])
        del arb,arb_opt,future,idm_opt,params,idm_params
        torch.cuda.empty_cache()
    (OUT/'summary.json').write_text(json.dumps(dict(config=config,variants=results),indent=2))
    emit('ALL_COMPLETED')

if __name__=='__main__':
    main()
