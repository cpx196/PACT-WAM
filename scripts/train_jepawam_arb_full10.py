"""Matched full-data P/W versus no-P/W runs; no early stopping."""
import os,json,time,math,fcntl,gc,traceback
import numpy as np
import torch
import train_jepawam_arb_pilot100 as t

def main():
    out=t.OUT
    out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'run.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'config.json').exists():raise RuntimeError('Refusing duplicate full training')
    (out/'pid').write_text(str(os.getpid()))
    torch.set_num_threads(2);t.seed()
    assert json.load(open(t.CACHE/'summary.json'))['status']=='complete'
    train=t.Windows('train');val=t.Windows('val')
    assert (len(train),len(val))==(31495,3500)
    keys=lambda ds:{(r['suite'],r['task'],r['demo']) for r in ds.rows}
    assert keys(train).isdisjoint(keys(val))
    stats=json.load(open(t.BASE/'train_only_stats.json'))
    config=json.load(open(t.ROOT/'jepawam_arb_clip5_pilot100_20261013/config.json'))
    config.pop('train_order',None)
    config.update(epochs=10,steps_per_epoch=math.ceil(len(train)/t.BATCH),steps=10*math.ceil(len(train)/t.BATCH),
        train_sampled_windows=len(train),initialization='fresh identical seed713; not resumed from pilot',
        no_pw=t.NO_PW,gradient_clip_norm=t.CLIP,save_steps='every epoch',eval_steps='baseline and every epoch',
        lr_schedule='100-step linear warmup1e-5 to1e-4 then cosine decay to1e-5 at10epochs',
        warmup_steps=100,minimum_learning_rate=1e-5,train_order='numpy default_rng(713+epoch_index).permutation(all31495)',
        visual_control='zero content before LayerNorm; keep full architecture' if t.NO_PW else 'native P/W content',
        smoke='3 discarded optimizer steps; reinitialize same seed',early_stopping=False)
    t.savejson('config.json',config)
    def newmodel():
        t.seed();return t.ARB(stats).cuda()
    m=newmodel();opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.01)
    b=t.device(next(iter(t.loader(train,list(range(8)),workers=0))))
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):_,a,d=m.loss(b)
    assert torch.equal(a,b['action'][...,:6]) and bool((d==0).all())
    smoke=[]
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,_,_=m.loss(b)
        assert torch.isfinite(loss);loss.backward();gn=t.gradnorm(m);assert math.isfinite(gn)
        if step>0:
            proj=m.action_project if t.NO_PW else m.p_project
            assert float(proj.weight.grad.norm())>0
        torch.nn.utils.clip_grad_norm_(m.parameters(),t.CLIP);opt.step()
        smoke.append(dict(step=step+1,loss=float(loss),gradient_norm=gn))
    if t.NO_PW:
        m.eval();changed=dict(b);changed['P']=torch.randn_like(b['P']);changed['W']=torch.randn_like(b['W'])
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            assert torch.equal(m(b)[0],m(changed)[0])
    t.savejson('smoke.json',dict(status='passed',records=smoke,no_pw=t.NO_PW))
    t.emit('SMOKE_PASSED',records=smoke,no_pw=t.NO_PW)
    del m,opt,b,a,d;gc.collect();torch.cuda.empty_cache()
    m=newmodel();opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.01)
    vl=t.loader(val,list(range(len(val))))
    started=time.time();baseline=t.evaluate(m,vl);validations=[dict(epoch=0,step=0,**baseline)]
    t.savejson('validation.json',validations);t.emit('BASELINE_VALIDATION',**baseline)
    globalstep=0;clipped=0;history=[];torch.cuda.reset_peak_memory_stats()
    with open(out/'metrics.jsonl','a',buffering=1) as metrics:
        for epoch in range(1,11):
            order=np.random.default_rng(t.SEED+epoch-1).permutation(len(train)).tolist()
            tl=t.loader(train,order);epochstart=time.time();epochloss=0.;windows=0;previous=time.time()
            for cpu in tl:
                tick=time.time();wait=tick-previous;b=t.device(cpu);m.train();opt.zero_grad(set_to_none=True)
                nextstep=globalstep+1
                if nextstep<=100:
                    lr=1e-5+(1e-4-1e-5)*nextstep/100
                else:
                    fraction=(nextstep-100)/(config['steps']-100)
                    lr=1e-5+.5*(1e-4-1e-5)*(1+math.cos(math.pi*min(fraction,1)))
                for group in opt.param_groups:group['lr']=lr
                with torch.autocast('cuda',dtype=torch.bfloat16):loss,_,delta=m.loss(b)
                assert torch.isfinite(loss);loss.backward();gn=t.gradnorm(m);assert math.isfinite(gn)
                torch.nn.utils.clip_grad_norm_(m.parameters(),t.CLIP);opt.step();torch.cuda.synchronize()
                globalstep+=1;clipped+=gn>t.CLIP;n=b['action'].shape[0];epochloss+=float(loss)*n;windows+=n
                row=dict(epoch=epoch,step=globalstep,loss=float(loss),gradient_norm=gn,clipped=gn>t.CLIP,learning_rate=lr,
                    compute_s=time.time()-tick,data_wait_s=wait,delta_rms=float(delta.square().mean().sqrt()))
                metrics.write(json.dumps(row)+'\n')
                if globalstep<=3 or globalstep%100==0:
                    elapsed=time.time()-started;rate=(time.time()-epochstart)/(windows/t.BATCH)
                    progress=dict(status='running',epoch=epoch,step=globalstep,total_steps=config['steps'],
                        epoch_windows=windows,epoch_total_windows=len(train),elapsed_s=elapsed,
                        mean_wall_step_s=rate,estimated_remaining_s=(config['steps']-globalstep)*rate,
                        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                    t.savejson('progress.json',progress);t.emit('TRAIN',**row,progress=progress)
                previous=time.time()
            assert windows==len(train)
            v=t.evaluate(m,vl);validations.append(dict(epoch=epoch,step=globalstep,**v));t.savejson('validation.json',validations)
            record=dict(epoch=epoch,step=globalstep,train_mse=epochloss/windows,val_mse=v['normalized_mse'],
                epoch_seconds=time.time()-epochstart,clip_rate=clipped/globalstep)
            history.append(record);t.savejson('epochs.json',history)
            path=out/f'epoch{epoch:02d}_step{globalstep}.pt';tmp=path.with_suffix('.tmp')
            torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),epoch=epoch,step=globalstep,config=config,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),
                python_rng=t.random.getstate(),scheduler=dict(kind='warmup_cosine_floor',step=globalstep,
                    total_steps=config['steps'],warmup_steps=100,peak_lr=1e-4,min_lr=1e-5)),tmp);tmp.replace(path)
            t.emit('EPOCH_COMPLETE',**record,checkpoint=str(path));del tl
    summary=dict(status='complete',epochs=10,steps=globalstep,no_pw=t.NO_PW,baseline=baseline,
        validation=validations,epoch_history=history,elapsed_s=time.time()-started,checkpoint=str(path),
        clip_rate=clipped/globalstep,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    t.savejson('summary.json',summary);t.savejson('progress.json',dict(status='complete',epoch=10,step=globalstep));t.emit('COMPLETE',**summary)

if __name__=='__main__':
    try:main()
    except Exception:
        t.OUT.mkdir(parents=True,exist_ok=True)
        t.savejson('failure.json',dict(status='failed',traceback=traceback.format_exc()));raise
