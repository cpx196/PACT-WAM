"""Full10-epoch AR2 run, reproducing the successful joint20 hyperparameters."""
import os,json,time,math,fcntl,traceback,random,zipfile,struct,hashlib
from pathlib import Path
from collections import OrderedDict
import numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from prepare_fastwam_ar2_and_smoke import ARPredictor,DATA,OUT as CACHE
from vectorized_ac_attention import enable_vectorized
OUT=Path('/data/pxchen/PACT-WAM/artifacts/fastwam_predictor21_ar2_full10epoch_bs16_20261014')
HISTORY=Path('/data/pxchen/PACT-WAM/artifacts/predictor21_joint20_train10epoch_bs16_gpu3_20261011/config.json')
TOTAL=19690;ANCHOR=3750;WARMUP=296;ORIGINAL_END=5907

def save(name,obj):
    p=OUT/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,indent=2));q.replace(p)
def emit(event,**kw):print(json.dumps(dict(event=event,**kw)),flush=True)

class CacheWindows(Dataset):
    def __init__(self,rows):
        self.rows=rows;self.items=[(i,j) for i,r in enumerate(rows) for j in range(r['windows'])];self.lru=OrderedDict()
    def __len__(self):return len(self.items)
    def offsets(self,path):
        name=str(path)
        if name not in self.lru:
            result=[]
            with zipfile.ZipFile(path) as archive,open(path,'rb') as f:
                for storage in range(4 if path.parent.parent==DATA else 1):
                    item=next(x for x in archive.infolist() if x.filename.endswith('/data/'+str(storage)))
                    assert item.compress_type==zipfile.ZIP_STORED
                    f.seek(item.header_offset+26);fn,extra=struct.unpack('<HH',f.read(4));result.append(item.header_offset+30+fn+extra)
            self.lru[name]=result
            if len(self.lru)>16:self.lru.popitem(last=False)
        self.lru.move_to_end(name);return self.lru[name]
    def __getitem__(self,index):
        i,j=self.items[index];row=self.rows[i];source=DATA/row['file'];target=CACHE/row['target_file']
        so=self.offsets(source);to=self.offsets(target)
        def read(f,offset,shape):
            n=int(np.prod(shape));f.seek(offset+j*n*4);buf=f.read(n*4);assert len(buf)==n*4
            return torch.from_numpy(np.frombuffer(buf,dtype=np.float32).copy().reshape(shape))
        with open(source,'rb') as f:
            z=read(f,so[0],(576,1024));a=read(f,so[2],(20,7))[:8];s=read(f,so[3],(8,))
        with open(target,'rb') as f:y=read(f,to[0],(2,576,1024))
        return dict(z=z,a=a,s=s,y=y)

def lr_at(step,anchor_lr):
    if step<ANCHOR:
        if step<WARMUP:return 1e-4*(step+1)/WARMUP
        return max(1e-5,1e-4*.5*(1+math.cos(math.pi*(step-WARMUP)/(ORIGINAL_END-WARMUP))))
    phase=min(1.,max(0.,(step-ANCHOR)/(TOTAL-ANCHOR)))
    return 1e-5+(anchor_lr-1e-5)*.5*(1+math.cos(math.pi*phase))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=open(OUT/'run.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'summary.json').exists():emit('ALREADY_COMPLETE');return
    resume=os.environ.get('RESUME_CHECKPOINT')
    if (OUT/'config.json').exists() and not resume:
        # Safe fresh retry only for a failed initialization, never completed updates.
        initialized_only=(OUT/'failure.json').exists() and not list(OUT.glob('step*.pt'))
        if (OUT/'metrics.jsonl').exists():
            initialized_only=initialized_only and not any(json.loads(line).get('kind')=='train' for line in (OUT/'metrics.jsonl').read_text().splitlines())
        if os.environ.get('FRESH_RETRY')!='1' or not initialized_only:
            raise RuntimeError('Existing run; explicit resume checkpoint required')
        (OUT/'config.json').rename(OUT/f'config_failed_initialization_{int(time.time())}.json')
    assert json.load(open(CACHE/'summary.json'))['status']=='complete'
    assert json.load(open(CACHE/'smoke_summary.json'))['status']=='passed'
    h=json.load(open(HISTORY));assert h['batch_size']==16 and h['steps']==TOTAL and h['warmup']==WARMUP
    rows=json.load(open(CACHE/'index.json'));assert len(rows)==5000
    ds={split:CacheWindows([r for r in rows if r['split']==split]) for split in ['train','val']}
    assert len(ds['train'])==31495 and len(ds['val'])==3500
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(7);np.random.seed(7);random.seed(7)
    val_ids=np.random.default_rng(17).choice(3500,128,replace=False).tolist()
    assert val_ids==h['validation_indices']
    # Verify the bounded reader against normal torch.load on both data splits.
    for split in ['train','val']:
        for idx in [0,len(ds[split])-1]:
            sample=ds[split][idx];ri,j=ds[split].items[idx];r=ds[split].rows[ri]
            d=torch.load(DATA/r['file'],map_location='cpu',weights_only=False,mmap=True)
            y=torch.load(CACHE/r['target_file'],map_location='cpu',weights_only=False,mmap=True)
            reference=dict(z=d['current_latent'][j],a=d['raw_actions'][j,:8],s=d['state'][j],y=y['targets'][j])
            assert all(torch.equal(sample[k],reference[k]) for k in sample),'Storage reader mismatch'
            assert torch.equal(d['starts'],y['starts'])
    emit('DIRECT_READER_VERIFIED')
    batches=[]
    for epoch in range(10):
        order=np.random.default_rng(7+epoch).permutation(31495).tolist()
        batches.extend(order[i:i+16] for i in range(0,len(order),16))
    assert len(batches)==TOTAL
    validation=[b for b in DataLoader(ds['val'],batch_size=16,sampler=val_ids,num_workers=0,pin_memory=False)]
    stats=json.load(open(CACHE/'train_only_stats.json'))
    norm={k:(torch.tensor(v['mean'],device='cuda'),torch.tensor(v['std'],device='cuda').clamp_min(1e-6)) for k,v in stats.items()}
    m=ARPredictor().cuda();enable_vectorized(m)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.04)
    anchor_lr=h['lr_anchor_value'];start_step=0
    config=dict(batch_size=16,epochs=10,steps=TOTAL,steps_per_epoch=1969,parameters=sum(p.numel() for p in m.parameters()),
        seed=7,initialization='fresh random, not pilot200 checkpoint',learning_rate=1e-4,weight_decay=.04,clip=1,
        warmup=WARMUP,original_decay_end=ORIGINAL_END,lr_anchor_step=ANCHOR,lr_anchor_value=anchor_lr,lr_floor=1e-5,
        lr_decay_end=TOTAL,precision='BF16 autocast, FP32 parameters/cache',checkpoint_mode='reentrant',single_thread_backward=True,
        validation_windows=128,validation_seed=17,validation_indices=val_ids,save_every=250,
        train_windows=31495,val_windows=3500,data_root=str(CACHE),base_cache=str(DATA),
        manifest_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
        loss='equal two-step autoregressive latent L1',action_steps_per_call=4,action_steps_total=8,state_dim=8,
        teacher_forcing=False,detach_between_steps=False,encoder_frozen=True,compile=False,
        num_workers=0,pin_memory=False,cpu_threads=1,epoch_tail='7 actual windows; no padding or dropping',
        hyperparameter_reference=str(HISTORY),vectorized_qkv=True)
    if (OUT/'config.json').exists():assert json.load(open(OUT/'config.json'))==config
    else:save('config.json',config)
    if resume:
        ck=torch.load(resume,map_location='cpu',weights_only=False)
        assert ck['config']==config
        m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start_step=ck['step']
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng']);del ck
        assert start_step<TOTAL
    if (OUT/'failure.json').exists():(OUT/'failure.json').rename(OUT/f'failure_previous_{int(time.time())}.json')
    emit('MODEL_READY',**{k:v for k,v in config.items() if k!='validation_indices'})
    def prepare(b):
        b={k:v.cuda(non_blocking=True) for k,v in b.items()};am,ast=norm['action'];sm,sst=norm['state']
        b['a']=(b['a']-am)/ast;b['s']=(b['s']-sm)/sst;return b
    def forward(b,mode='normal'):
        a=b['a'] if mode=='normal' else b['a'].roll(1,0) if mode=='shuffled' else torch.zeros_like(b['a'])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            p1=m(b['z'],a[:,:4],b['s'],0);p2=m(p1,a[:,4:8],b['s'],1)
        return torch.stack((p1,p2),dim=1).float()
    def evaluate(step):
        m.eval();sums={k:np.zeros(2) for k in ['normal','shuffled_action','mean_action','copy_current']};changes=0.
        with torch.no_grad():
            for cpu in validation:
                b=prepare(cpu);p=forward(b)
                sums['normal']+=(p-b['y']).abs().mean((0,2,3)).cpu().numpy()
                shifted=forward(b,'shuffled');sums['shuffled_action']+=(shifted-b['y']).abs().mean((0,2,3)).cpu().numpy()
                changes+=float((shifted-p).abs().mean())
                sums['mean_action']+=(forward(b,'mean')-b['y']).abs().mean((0,2,3)).cpu().numpy()
                sums['copy_current']+=(b['z'][:,None]-b['y']).abs().mean((0,2,3)).cpu().numpy()
        result=dict(kind='validation',step=step,epoch=step/1969,
            **{k:float(v.mean()/len(validation)) for k,v in sums.items()},
            interval_l1={k:(v/len(validation)).tolist() for k,v in sums.items()},action_output_change=changes/len(validation))
        with open(OUT/'metrics.jsonl','a') as f:f.write(json.dumps(result)+'\n')
        emit('VALIDATION',**result);m.train();return result
    baseline=evaluate(start_step);started=time.perf_counter();clipped=0;last_validation=baseline
    prefetch_workers=int(os.environ.get('AR2_PREFETCH_WORKERS','0'))
    if prefetch_workers:
        from ar2_prefetch import PrefetchBatches
        loader=PrefetchBatches(ds['train'],batches[start_step:],workers=prefetch_workers,depth=2*prefetch_workers,pin=True)
    else:loader=DataLoader(ds['train'],batch_sampler=batches[start_step:],num_workers=0,pin_memory=False)
    emit('INPUT_PIPELINE',read_threads=prefetch_workers,prefetch_batches=2*prefetch_workers,
         pinned=bool(prefetch_workers),sampler_order_unchanged=True,start_step=start_step)
    save('input_pipeline.json',dict(read_threads=prefetch_workers,prefetch_batches=2*prefetch_workers,
         pinned=bool(prefetch_workers),sampler_order_unchanged=True,start_step=start_step,
         note='Runtime IO settings override historical reference pin_memory field; training hyperparameters unchanged.'))
    for step,cpu in enumerate(loader,start_step+1):
        tick=time.perf_counter();b=prepare(cpu);opt.zero_grad(set_to_none=True)
        lr=lr_at(step-1,anchor_lr)
        for group in opt.param_groups:group['lr']=lr
        pred=forward(b);assert pred.shape==b['y'].shape
        loss=(pred-b['y']).abs().mean();assert torch.isfinite(loss)
        with torch.autograd.set_multithreading_enabled(False):loss.backward()
        g=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True));clipped+=g>1
        opt.step();torch.cuda.synchronize()
        record=dict(kind='train',step=step,epoch=step/1969,loss=float(loss.detach()),
            step1_l1=float((pred[:,0]-b['y'][:,0]).abs().mean().detach()),
            step2_l1=float((pred[:,1]-b['y'][:,1]).abs().mean().detach()),
            gradient_norm_before_clip=g,clipped=g>1,lr=lr,step_s=time.perf_counter()-tick)
        with open(OUT/'metrics.jsonl','a') as f:f.write(json.dumps(record)+'\n')
        if step<=3 or step%10==0:emit('STEP',**record)
        if step%250==0 or step==TOTAL:
            last_validation=evaluate(step)
            path=OUT/f'step{step:05d}.pt';tmp=path.with_suffix('.tmp')
            torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),step=step,config=config,
                scheduler_state=dict(next_step=step,anchor_step=ANCHOR,anchor_lr=anchor_lr,decay_end=TOTAL,warmup=WARMUP,lr_floor=1e-5),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate()),tmp);tmp.replace(path)
            emit('CHECKPOINT_SAVED',step=step,path=str(path))
        elapsed=time.perf_counter()-started;done=step-start_step
        if step<=3 or step%10==0:
            save('progress.json',dict(status='running',step=step,total_steps=TOTAL,epoch=step/1969,elapsed_s=elapsed,
                estimated_remaining_s=elapsed/done*(TOTAL-step),last_loss=record['loss'],validation=last_validation,clip_rate=clipped/done,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
    save('summary.json',dict(status='complete',steps=TOTAL,epochs=10,baseline=baseline,final_validation=last_validation,
        elapsed_s=time.perf_counter()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
    save('progress.json',dict(status='complete',step=TOTAL,total_steps=TOTAL,epoch=10));emit('COMPLETE')

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True);save('failure.json',dict(traceback=traceback.format_exc()));raise
