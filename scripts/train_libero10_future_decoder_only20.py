"""Matched 20-epoch Decoder-only control: Predictor future -> real future.

Same AdaLN decoder as comparison IDM experiment; constant zero conditioning.
No IDM, posterior, WAM input, KL, or stochastic code. This tests the whole
comparison-code pathway, not separately WAM information or VAE noise effects.
"""
import fcntl
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scripts.train_libero10_future_correction_reconstruct10 import Model, CACHE, MANIFEST, emit, save

OUT = Path('/data/pxchen/PACT-WAM/artifacts/libero10_future_decoder_only20_20261007')


def main():
    torch.set_num_threads(4)
    torch.manual_seed(239)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'launcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT/'metrics.jsonl').exists() and not (OUT/'summary.json').exists()
    (OUT/'pid.txt').write_text(str(os.getpid())+'\n')
    rows = json.loads(MANIFEST.read_text())
    train = [i for i,r in enumerate(rows) if r['split']=='train']
    val = [i for i,r in enumerate(rows) if r['split']=='val']
    assert (len(rows),len(train),len(val)) == (4307,3871,436)
    assert {(rows[i]['task_file'],rows[i]['demo']) for i in train}.isdisjoint(
        {(rows[i]['task_file'],rows[i]['demo']) for i in val})
    z = torch.empty(len(rows),2,2,256,1408)
    started = time.monotonic()
    for i,row in enumerate(rows):
        item = torch.load(CACHE/f'{i:05d}.pt',map_location='cpu',weights_only=False)
        assert item['row']==row and item['index']==i
        for j,key in enumerate(['predicted_future','real_future']):
            v = item[key].float()
            assert v.shape==(2,256,1408) and torch.isfinite(v).all()
            z[i,:,j].copy_(v)
        if (i+1)%512==0 or i+1==len(rows):
            emit('DATA_LOAD_PROGRESS',loaded=i+1,seconds=time.monotonic()-started)
    # Instantiate the full model only on CPU to reproduce exactly the original
    # random Decoder initialization, then discard the unused IDM/posterior.
    template = Model()
    decoder = template.decoder
    del template
    decoder = decoder.cuda()
    initial = {k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()}
    config = dict(epochs=20,early_stopping=False,seed=239,batch=8,steps_per_epoch=484,
                  train=len(train),val=len(val),stride=32,dtype='float32',lr=1e-4,
                  weight_decay=.01,dropout=.1,grad_clip=1.,parameters=sum(p.numel() for p in decoder.parameters()),
                  model='identical AdaLN Decoder, constant zero 128D conditioning',
                  initialization='exact seed239 original Decoder initialization; fresh optimizer',
                  input='Predictor future full patches only',target='real future latent labels only',
                  wam_input=False,idm_used=False,posterior_used=False,kl_used=False,
                  loss='full patch future reconstruction MSE',arb_trained=False,
                  comparison_checkpoint='/data/pxchen/PACT-WAM/artifacts/libero10_future_correction_reconstruct40_20261007/epoch020.pt',
                  comparison='fixed epoch20 vs epoch20; best validation separately reported',
                  caveat='ablates entire comparison-code pathway, including VAE sampling and KL; not WAM-only causal isolation',
                  manifest=str(MANIFEST),token_cache=str(CACHE))
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    emit('MODEL_READY',**config)
    def optimizer_new():
        return torch.optim.AdamW(decoder.parameters(),lr=1e-4,weight_decay=.01)
    def batch(ids):
        f = z[ids].cuda().flatten(0,1)
        return f[:,0],f[:,1]
    def predict(p):
        return decoder(p,p.new_zeros(len(p),128))
    optimizer = optimizer_new()
    smoke=[]
    for step in range(5):
        decoder.train()
        optimizer.zero_grad(set_to_none=True)
        p,r = batch(train[:8])
        loss = F.mse_loss(predict(p),r)
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        smoke.append(dict(step=step+1,mse=float(loss),grad_norm=float(norm)))
    assert not torch.equal(decoder.project.weight.detach().cpu(),initial['project.weight'])
    emit('SMOKE_PASSED',steps=smoke)
    (OUT/'smoke.json').write_text(json.dumps(smoke,indent=2))
    decoder.load_state_dict(initial)
    del initial,optimizer,p,r,loss
    optimizer=optimizer_new()
    torch.manual_seed(239)
    probe=np.asarray(train)[np.linspace(0,len(train)-1,256,dtype=int)].tolist()
    @torch.no_grad()
    def evaluate(ids):
        decoder.eval()
        errors,base=[] ,[]
        for start in range(0,len(ids),8):
            p,r=batch(ids[start:start+8])
            errors.append((predict(p)-r).square().mean((1,2)).reshape(-1,2).cpu())
            base.append((p-r).square().mean((1,2)).reshape(-1,2).cpu())
        return dict(reconstruction_mse=float(torch.cat(errors).mean()),
                    reconstruction_mse_per_time=torch.cat(errors).mean(0).tolist(),
                    predictor_baseline_mse=float(torch.cat(base).mean()))
    initial_eval=dict(val=evaluate(val),train_probe=evaluate(probe))
    # Exact initial val agreement verifies matched Decoder initialization.
    assert abs(initial_eval['val']['reconstruction_mse']-.4751654863357544)<1e-6
    emit('INITIAL_EVALUATION',**initial_eval)
    (OUT/'initial_evaluation.json').write_text(json.dumps(initial_eval,indent=2))
    best,best_epoch,global_step=float('inf'),0,0
    started=time.monotonic()
    def checkpoint(name,epoch):
        save(dict(decoder=decoder.state_dict(),optimizer=optimizer.state_dict(),config=config,
                  epoch=epoch,step=global_step,best_val=best,best_epoch=best_epoch,
                  torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),OUT/name)
    for epoch in range(1,21):
        decoder.train()
        ids=np.random.default_rng(238+epoch).permutation(train).tolist()
        weighted,count,norms=0.,0,[]
        epoch_started=time.monotonic()
        for start in range(0,len(ids),8):
            ix=ids[start:start+8]
            optimizer.zero_grad(set_to_none=True)
            p,r=batch(ix)
            loss=F.mse_loss(predict(p),r)
            assert torch.isfinite(loss)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            norms.append(float(norm)); weighted+=float(loss)*len(ix); count+=len(ix); global_step+=1
            if len(norms)%100==0:
                emit('TRAIN_PROGRESS',epoch=epoch,batch=len(norms),step=global_step,
                     reconstruction_mse=weighted/count,seconds_per_batch=(time.monotonic()-epoch_started)/len(norms))
        row=dict(epoch=epoch,step=global_step,optimization_recon_mse=weighted/count,
                 val=evaluate(val),train_probe=evaluate(probe),clip_rate=float(np.mean(np.asarray(norms)>1)),
                 grad_norm_p50=float(np.median(norms)),grad_norm_p95=float(np.quantile(norms,.95)),
                 peak_gib=torch.cuda.max_memory_allocated()/2**30)
        if row['val']['reconstruction_mse']<best:
            best,best_epoch=row['val']['reconstruction_mse'],epoch
            checkpoint('best.pt',epoch)
        checkpoint('last.pt',epoch); checkpoint(f'epoch{epoch:03d}.pt',epoch)
        row.update(epoch_seconds=time.monotonic()-epoch_started,seconds=time.monotonic()-started)
        with (OUT/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        emit('EPOCH_COMPLETED',**row)
    summary=dict(config=config,initial=initial_eval,best_val_mse=best,best_epoch=best_epoch,
                 final=row,seconds=time.monotonic()-started)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    emit('ALL_COMPLETED',best_epoch=best_epoch,best_val_mse=best,final_val=row['val']['reconstruction_mse'])

if __name__=='__main__':
    main()
