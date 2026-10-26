"""Resume both matched ARB runs from epoch10, retaining optimizer; fixed LR1e-5."""
import json,time,math,fcntl,traceback,os
from pathlib import Path
import numpy as np
import torch
import train_jepawam_arb_pilot100 as t

def main():
    out=t.OUT;out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'run.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'config.json').exists():raise RuntimeError('Refusing duplicate continuation')
    (out/'pid').write_text(str(os.getpid()))
    source=t.ROOT/('jepawam_arb_no_pw_10epoch_clip5_20261013' if t.NO_PW else 'jepawam_arb_pw_10epoch_clip5_20261013')
    checkpoint=source/'epoch10_step39370.pt'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['epoch']==10 and ck['step']==39370 and ck['config']['no_pw']==t.NO_PW
    torch.set_num_threads(2)
    m=t.ARB(json.load(open(t.BASE/'train_only_stats.json'))).cuda()
    m.load_state_dict(ck['model'],strict=True)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-5,weight_decay=.01);opt.load_state_dict(ck['optimizer'])
    for group in opt.param_groups:group['lr']=1e-5
    config=dict(ck['config']);config.update(epochs=15,steps=59055,resume=str(checkpoint),
        learning_rate=1e-5,lr_schedule='fixed1e-5 for epochs11-15; restored optimizer',save_steps='every epoch',eval_steps='resume verification and every epoch')
    t.savejson('config.json',config)
    torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
    np.random.set_state(ck['numpy_rng']);t.random.setstate(ck['python_rng']);del ck
    train=t.Windows('train');val=t.Windows('val');assert (len(train),len(val))==(31495,3500)
    vl=t.loader(val,list(range(len(val))))
    baseline=t.evaluate(m,vl);original=json.load(open(source/'summary.json'))
    assert abs(baseline['normalized_mse']-original['validation'][-1]['normalized_mse'])<1e-6
    t.savejson('smoke.json',dict(status='passed',checkpoint_reproduction=baseline,optimizer_restored=True))
    t.emit('RESUME_VERIFIED',**baseline)
    history=list(original['epoch_history']);validations=list(original['validation'])
    t.savejson('epochs.json',history);t.savejson('validation.json',validations)
    started=time.time();globalstep=39370;clipped=0;continued=0
    with open(out/'metrics.jsonl','a',buffering=1) as metrics:
        for epoch in range(11,16):
            order=np.random.default_rng(t.SEED+epoch-1).permutation(len(train)).tolist()
            tl=t.loader(train,order);epochstart=time.time();sumloss=0.;windows=0;previous=time.time()
            for cpu in tl:
                tick=time.time();wait=tick-previous;b=t.device(cpu);m.train();opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):loss,_,delta=m.loss(b)
                assert torch.isfinite(loss);loss.backward();gn=t.gradnorm(m);assert math.isfinite(gn)
                torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();torch.cuda.synchronize()
                globalstep+=1;continued+=1;clipped+=gn>5;n=b['action'].shape[0];windows+=n;sumloss+=float(loss)*n
                row=dict(epoch=epoch,step=globalstep,loss=float(loss),gradient_norm=gn,clipped=gn>5,
                    learning_rate=1e-5,compute_s=time.time()-tick,data_wait_s=wait,delta_rms=float(delta.square().mean().sqrt()))
                metrics.write(json.dumps(row)+'\n')
                if continued<=3 or globalstep%100==0:
                    rate=(time.time()-epochstart)/(windows/t.BATCH)
                    progress=dict(status='running',epoch=epoch,step=globalstep,total_steps=59055,
                        epoch_windows=windows,mean_wall_step_s=rate,elapsed_s=time.time()-started,
                        estimated_remaining_s=(59055-globalstep)*rate,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                    t.savejson('progress.json',progress);t.emit('TRAIN',**row,progress=progress)
                previous=time.time()
            assert windows==len(train)
            v=t.evaluate(m,vl);validations.append(dict(epoch=epoch,step=globalstep,**v));t.savejson('validation.json',validations)
            record=dict(epoch=epoch,step=globalstep,train_mse=sumloss/windows,val_mse=v['normalized_mse'],
                epoch_seconds=time.time()-epochstart,clip_rate=clipped/continued)
            history.append(record);t.savejson('epochs.json',history)
            path=out/f'epoch{epoch:02d}_step{globalstep}.pt';temp=path.with_suffix('.tmp')
            torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),epoch=epoch,step=globalstep,config=config,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),
                python_rng=t.random.getstate(),scheduler=dict(kind='fixed_floor',learning_rate=1e-5,step=globalstep)),temp)
            temp.replace(path);t.emit('EPOCH_COMPLETE',**record,checkpoint=str(path));del tl
    t.savejson('summary.json',dict(status='complete',epochs=15,steps=globalstep,no_pw=t.NO_PW,
        validation=validations,epoch_history=history,checkpoint=str(path),elapsed_s=time.time()-started,
        clip_rate=clipped/continued,optimizer_restored=True))
    t.savejson('progress.json',dict(status='complete',epoch=15,step=globalstep));t.emit('COMPLETE',step=globalstep)

if __name__=='__main__':
    try:main()
    except Exception:
        t.OUT.mkdir(parents=True,exist_ok=True);t.savejson('failure.json',dict(traceback=traceback.format_exc()));raise
