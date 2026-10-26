"""Fresh random initialization, frozen latent cache, two-step AR pilot200."""
import json,os,time,fcntl,traceback,hashlib,random
from pathlib import Path
from collections import OrderedDict
import numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from prepare_fastwam_ar2_and_smoke import ARPredictor,DATA,OUT as CACHE
from vectorized_ac_attention import enable_vectorized
OUT=Path('/data/pxchen/PACT-WAM/artifacts/fastwam_predictor21_ar2_partial_pilot200_20261014')

def save(name,obj):
    p=OUT/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,indent=2));q.replace(p)
def emit(event,**kw):print(json.dumps(dict(event=event,**kw)),flush=True)

class Windows(Dataset):
    def __init__(self,rows):
        self.rows=rows;self.lookup=[(i,j) for i,r in enumerate(rows) for j in range(r['windows'])]
        self.lru=OrderedDict()
    def __len__(self):return len(self.lookup)
    def __getitem__(self,index):
        i,j=self.lookup[index]
        if i not in self.lru:
            row=self.rows[i]
            d=torch.load(DATA/row['file'],map_location='cpu',weights_only=False,mmap=True)
            y=torch.load(CACHE/row['target_file'],map_location='cpu',weights_only=False,mmap=True)
            assert y['source_shard']==row['file'] and torch.equal(y['starts'],d['starts'])
            self.lru[i]=(d,y)
            if len(self.lru)>8:self.lru.popitem(last=False)
        self.lru.move_to_end(i);d,y=self.lru[i]
        return dict(z=d['current_latent'][j].float(),a=d['raw_actions'][j,:8].float().clone(),
                    s=d['state'][j].float().clone(),y=y['targets'][j].float())

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=open(OUT/'run.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'config.json').exists():raise RuntimeError('Refusing duplicate fresh200 run')
    assert json.load(open(CACHE/'smoke_summary.json'))['status']=='passed'
    torch.set_num_threads(4);torch.manual_seed(713);np.random.seed(713);random.seed(713)
    source=json.load(open(DATA/'index.json'))
    rows=[dict(**r,target_file='shards/'+Path(r['file']).name) for r in source if (CACHE/'shards'/Path(r['file']).name).exists()]
    save('snapshot_manifest.json',rows)
    splits={split:[r for r in rows if r['split']==split] for split in ['train','val']}
    datasets={split:Windows(v) for split,v in splits.items()}
    assert len(datasets['train'])>=1600 and len(datasets['val'])>=256
    # Freeze the available-shard snapshot; cache expansion cannot change this run.
    rng=np.random.default_rng(713)
    validation_indices=rng.choice(len(datasets['val']),256,replace=False).tolist()
    train_probe_indices=rng.choice(len(datasets['train']),128,replace=False).tolist()
    save('probe_indices.json',dict(val=validation_indices,train=train_probe_indices))
    generators=torch.Generator().manual_seed(713)
    loaders={
        'train':DataLoader(datasets['train'],batch_size=8,shuffle=True,generator=generators,num_workers=4,pin_memory=True,drop_last=True,persistent_workers=True),
        'val':DataLoader(torch.utils.data.Subset(datasets['val'],validation_indices),batch_size=8,num_workers=2,pin_memory=True,persistent_workers=True),
        'probe':DataLoader(torch.utils.data.Subset(datasets['train'],train_probe_indices),batch_size=8,num_workers=2,pin_memory=True,persistent_workers=True)}
    stats=json.load(open(CACHE/'train_only_stats.json'))
    norm={k:(torch.tensor(v['mean'],device='cuda'),torch.tensor(v['std'],device='cuda').clamp_min(1e-6)) for k,v in stats.items()}
    m=ARPredictor().cuda();enable_vectorized(m)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-5,weight_decay=.04)
    config=dict(initialization='fresh random713, not smoke weights',parameters=sum(p.numel() for p in m.parameters()),
        encoder_frozen=True,encoder_used='precomputed frozen2.1 ViT-L384 cache only',batch_size=8,steps=200,
        train_windows=len(datasets['train']),val_available_windows=len(datasets['val']),val_probe=256,train_probe=128,
        suites={suite:sum(r['windows'] for r in rows if r['suite']==suite) for suite in sorted({r['suite'] for r in rows})},
        snapshot_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
        learning_rate='warmup20:1e-5 floor to1e-4, then constant1e-4',weight_decay=.04,clip=1.,
        precision='BF16 autocast, FP32 parameters',compile=False,teacher_forcing=False,detach_between_steps=False,
        loss='0.5*L1(step1)+0.5*L1(step2)',validation_interval=25,checkpoint_interval=50,
        note='Partial available cache; not a full-data epoch. No encoder or WAM weights updated.')
    save('config.json',config);emit('MODEL_READY',**config)
    def move(b):
        b={k:v.cuda(non_blocking=True) for k,v in b.items()}
        am,ast=norm['action'];sm,sst=norm['state']
        b['a']=(b['a']-am)/ast;b['s']=(b['s']-sm)/sst;return b
    def forward(b,shuffle=False):
        a=b['a'].roll(1,0) if shuffle else b['a']
        with torch.autocast('cuda',dtype=torch.bfloat16):
            p1=m(b['z'],a[:,:4],b['s'],0)
            p2=m(p1,a[:,4:8],b['s'],1)
        l1=(p1.float()-b['y'][:,0]).abs().mean();l2=(p2.float()-b['y'][:,1]).abs().mean()
        return (l1+l2)*.5,l1,l2
    def evaluate(loader,shuffled=False):
        m.eval();acc=np.zeros(3);count=0
        with torch.no_grad():
            for cpu in loader:
                b=move(cpu);values=forward(b,shuffle=shuffled);n=len(b['z'])
                acc+=np.array([float(v) for v in values])*n;count+=n
        m.train();return dict(zip(['loss','step1_l1','step2_l1'],(acc/count).tolist()))
    records=[];validation=[];started=time.perf_counter()
    initial=dict(step=0,validation=evaluate(loaders['val']),train_probe=evaluate(loaders['probe']))
    validation.append(initial);save('validation.json',validation);emit('VALIDATION',**initial)
    iterator=iter(loaders['train']);clipped=0
    for step in range(1,201):
        tick=time.perf_counter()
        try:cpu=next(iterator)
        except StopIteration:iterator=iter(loaders['train']);cpu=next(iterator)
        b=move(cpu);opt.zero_grad(set_to_none=True)
        lr=max(1e-5,1e-4*min(1.,step/20))
        for group in opt.param_groups:group['lr']=lr
        loss,l1,l2=forward(b);assert torch.isfinite(loss)
        loss.backward()
        gradients={k:float(v.weight.grad.float().norm()) for k,v in [('action',m.backbone.action_encoder),('state',m.backbone.state_encoder)]}
        assert all(np.isfinite(v) for v in gradients.values())
        g=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True));clipped+=g>1
        opt.step();torch.cuda.synchronize()
        rec=dict(step=step,loss=float(loss.detach()),step1_l1=float(l1.detach()),step2_l1=float(l2.detach()),lr=lr,
            grad_norm_before_clip=g,projection_gradients=gradients,step_s=time.perf_counter()-tick)
        records.append(rec)
        with open(OUT/'metrics.jsonl','a') as f:f.write(json.dumps(rec)+'\n')
        if step<=3 or step%10==0:emit('STEP',**rec)
        if step%25==0:
            v=dict(step=step,validation=evaluate(loaders['val']),train_probe=evaluate(loaders['probe']))
            validation.append(v);save('validation.json',validation);emit('VALIDATION',**v)
        if step%50==0:
            path=OUT/f'step{step:04d}.pt';tmp=path.with_suffix('.tmp')
            torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),step=step,config=config,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),
                python_rng=random.getstate(),shuffle_generator_rng=generators.get_state()),tmp);tmp.replace(path)
        elapsed=time.perf_counter()-started
        save('progress.json',dict(status='running',step=step,total_steps=200,elapsed_s=elapsed,
            estimated_remaining_s=elapsed/step*(200-step),last_loss=rec['loss'],clipping_rate=clipped/step))
    shuffled=evaluate(loaders['val'],shuffled=True)
    summary=dict(status='complete',config=config,initial=initial,final=validation[-1],
        shuffled_action_validation=shuffled,clipping_rate=clipped/200,
        mean_training_step_s=float(np.mean([x['step_s'] for x in records])),
        elapsed_s=time.perf_counter()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    save('summary.json',summary);save('progress.json',dict(status='complete',step=200,total_steps=200));emit('COMPLETE',**summary)

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True);save('failure.json',dict(traceback=traceback.format_exc()));raise
