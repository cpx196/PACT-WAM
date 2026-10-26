"""Frozen cached JEPA-WAM P/W -> full-token residual Transformer; bounded pilot."""
import os,json,time,math,random,fcntl,traceback,gc
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader

ROOT=Path('/data/pxchen/PACT-WAM/artifacts')
CACHE=ROOT/'jepawam_pw_joint20_stride20_full_20261012'
BASE=ROOT/'predictor21_joint20_stride20_20261011'
OUT=Path(os.environ.get('ARB_OUT',str(ROOT/'jepawam_arb_pilot100_20261013')))
NO_PW=os.environ.get('ARB_NO_PW','0')=='1'
CLIP=float(os.environ.get('ARB_CLIP_NORM','1'))
SEED=713;BATCH=8
def emit(event,**kw):print(json.dumps(dict(event=event,time=time.time(),**kw)),flush=True)
def savejson(name,value):
    p=OUT/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2));q.replace(p)
def seed():torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)

class Windows(Dataset):
    def __init__(self,split):
        self.rows=[r for r in json.load(open(CACHE/'index.json')) if r['split']==split]
        self.items=[(i,j) for i,r in enumerate(self.rows) for j in range(r['windows'])]
        self.cache=OrderedDict()
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        ri,j=self.items[i];r=self.rows[ri]
        if ri not in self.cache:
            d=torch.load(CACHE/r['cache_file'],map_location='cpu',weights_only=False,mmap=True)
            b=torch.load(BASE/r['file'],map_location='cpu',weights_only=False,mmap=True)
            assert d['source_shard']==r['file'] and d['split']==b['split']==r['split']
            assert torch.equal(d['starts'],b['starts'])
            self.cache[ri]=(d,b)
            if len(self.cache)>16:self.cache.popitem(last=False)
        self.cache.move_to_end(ri);d,b=self.cache[ri]
        return dict(P=d['P'][j],W=d['W'][j],action=d['wam_actions'][j],state=b['state'][j],target=b['raw_actions'][j],index=i)

class Block(nn.Module):
    def __init__(self):
        super().__init__();self.norm1=nn.LayerNorm(1024);self.norm2=nn.LayerNorm(1024)
        self.attn=nn.MultiheadAttention(1024,16,dropout=0,batch_first=True)
        self.ffn=nn.Sequential(nn.Linear(1024,4096),nn.GELU(),nn.Linear(4096,1024))
    def forward(self,x):
        n=self.norm1(x);x=x+self.attn(n,n,n,need_weights=False)[0]
        return x+self.ffn(self.norm2(x))

class ARB(nn.Module):
    def __init__(self,stats):
        super().__init__()
        for k,v in stats.items():
            self.register_buffer(k+'_mean',torch.tensor(v['mean'],dtype=torch.float32))
            self.register_buffer(k+'_std',torch.tensor(v['std'],dtype=torch.float32).clamp_min(1e-6))
        self.p_norm=nn.LayerNorm(1024);self.w_norm=nn.LayerNorm(1024)
        self.p_project=nn.Linear(1024,1024);self.w_project=nn.Linear(1024,1024)
        self.action_project=nn.Linear(7,1024);self.state_project=nn.Linear(8,1024)
        self.source=nn.Parameter(torch.randn(4,1024)*.02)
        self.action_pos=nn.Parameter(torch.randn(1,20,1024)*.02)
        ids=torch.arange(24,dtype=torch.float32);freq=1/(10000**(torch.arange(256,dtype=torch.float32)/256))
        v=ids[:,None]*freq[None];one=torch.cat((v.sin(),v.cos()),-1)
        pos=torch.cat((one[:,None,:].expand(24,24,512),one[None,:,:].expand(24,24,512)),-1).reshape(1,576,1024)
        self.register_buffer('spatial_pos',pos)
        self.blocks=nn.ModuleList([Block() for _ in range(8)])
        self.output=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,6))
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)
    def forward(self,b):
        # Matched control: same 1173-token architecture and compute, no sample-specific P/W.
        # Constant spatial/source tokens remain; only visual content is removed.
        p_input=torch.zeros_like(b['P']) if NO_PW else b['P']
        w_input=torch.zeros_like(b['W']) if NO_PW else b['W']
        p=self.p_project(self.p_norm(p_input))+self.spatial_pos+self.source[0]
        w=self.w_project(self.w_norm(w_input))+self.spatial_pos+self.source[1]
        s=self.state_project((b['state']-self.state_mean)/self.state_std)[:,None]+self.source[2]
        a=self.action_project((b['action']-self.action_mean)/self.action_std)+self.action_pos+self.source[3]
        x=torch.cat((p,w,s,a),dim=1);assert x.shape[1:]==(1173,1024)
        for block in self.blocks:x=block(x)
        delta=self.output(x[:,-20:]).float()
        # Residual is in train-only normalized command space. Gripper is untouched.
        corrected=b['action'][...,:6]+delta*self.action_std[:6]
        return corrected,delta
    def loss(self,b):
        corrected,delta=self(b)
        error=(corrected-b['target'][...,:6])/self.action_std[:6]
        return error.square().mean(),corrected,delta

def device(b):return {k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in b.items()}
def gradnorm(m):return float(torch.stack([p.grad.detach().float().square().sum() for p in m.parameters() if p.grad is not None]).sum().sqrt())
def loader(ds,ids,workers=2):
    return DataLoader(ds,batch_size=BATCH,sampler=ids,num_workers=workers,pin_memory=True,
                      persistent_workers=bool(workers),prefetch_factor=2 if workers else None)
def evaluate(m,val_loader):
    m.eval();total=0;sums=np.zeros(6,dtype=np.float64);raw=np.zeros(6,dtype=np.float64)
    absolute=np.zeros(6,dtype=np.float64);rawabsolute=np.zeros(6,dtype=np.float64)
    with torch.no_grad():
        for cpu in val_loader:
            b=device(cpu)
            with torch.autocast('cuda',dtype=torch.bfloat16):_,out,_=m.loss(b)
            e=(out-b['target'][...,:6])/m.action_std[:6];r=(b['action'][...,:6]-b['target'][...,:6])/m.action_std[:6]
            sums+=e.square().sum((0,1)).cpu().numpy();raw+=r.square().sum((0,1)).cpu().numpy()
            absolute+=(out-b['target'][...,:6]).square().sum((0,1)).cpu().numpy()
            rawabsolute+=(b['action'][...,:6]-b['target'][...,:6]).square().sum((0,1)).cpu().numpy()
            total+=out.shape[0]*20
    m.train()
    return dict(windows=total//20,normalized_mse=float(sums.mean()/total),wam_normalized_mse=float(raw.mean()/total),
                normalized_mse_per_dim=(sums/total).tolist(),wam_normalized_mse_per_dim=(raw/total).tolist(),
                raw_mse_per_dim=(absolute/total).tolist(),wam_raw_mse_per_dim=(rawabsolute/total).tolist())

def main():
    OUT.mkdir(exist_ok=True);lock=open(OUT/'run.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'metrics.jsonl').exists() or (OUT/'summary.json').exists():raise RuntimeError('Refusing duplicate pilot')
    (OUT/'pid').write_text(str(os.getpid()))
    torch.set_num_threads(2);seed()
    assert json.load(open(CACHE/'summary.json'))['status']=='complete'
    rows=json.load(open(CACHE/'index.json'))
    trainkeys={(r['suite'],r['task'],r['demo']) for r in rows if r['split']=='train'}
    valkeys={(r['suite'],r['task'],r['demo']) for r in rows if r['split']=='val'};assert trainkeys.isdisjoint(valkeys)
    train=Windows('train');val=Windows('val');assert (len(train),len(val))==(31495,3500)
    stats=json.load(open(BASE/'train_only_stats.json'))
    order=np.random.default_rng(SEED).permutation(len(train))[:800].tolist();assert len(set(order))==800
    if NO_PW:
        original_config=json.load(open(ROOT/'jepawam_arb_pilot100_20261013/config.json'))
        assert order==original_config['train_order'] and original_config['seed']==SEED and original_config['batch']==BATCH
    train_loader=loader(train,order);val_loader=loader(val,list(range(len(val))))
    seed();m=ARB(stats).cuda();parameters=sum(p.numel() for p in m.parameters())
    savejson('config.json',dict(parameters=parameters,layers=8,width=1024,heads=16,ffn=4096,input_tokens=1173,
        token_layout=['P576','W576','state1','action20'],output_shape=[20,6],gripper='unchanged',residual='normalized command space; zero head initialization',
        P_W='separate per-token LayerNorm and projection; no pooling',spatial_positions='fixed24x24 2D sin/cos; shared position grid for P/W',
        source_embeddings=4,action_positions=20,dropout=0,precision='BF16 autocast FP32 parameters',compile=False,batch=BATCH,
        optimizer='AdamW',learning_rate=1e-4,weight_decay=.01,gradient_clip_norm=CLIP,lr_schedule='constant for bounded pilot',
        loss='train-only normalized first6 action MSE; no other loss',frozen=['WAM','Encoder','Predictor'],
        cache=str(CACHE),base=str(BASE),train_windows=len(train),val_windows=len(val),train_sampled_windows=800,
        train_order=order,steps=100,eval_steps=[0,50,100],save_steps=[50,100],validation='all3500 fixed trajectory-disjoint windows',
        seed=SEED,num_workers=2,smoke='3 optimization steps then discard weights and optimizer; restart from identical seed',
        early_stopping=False,rms_cap=False,real_future_input=False,no_pw=NO_PW,
        visual_control='zero P/W content before per-token LayerNorm; retain positions, sources and full token count' if NO_PW else 'native P/W content'))
    emit('MODEL_READY',parameters=parameters)
    batches=iter(train_loader);b=device(next(batches));m.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):loss,out,delta=m.loss(b)
    assert torch.equal(out,b['action'][...,:6]) and bool((delta==0).all()),'Zero-head identity failed'
    emit('ZERO_HEAD_IDENTITY_PASSED',baseline_loss=float(loss))
    m.train();opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.01);smoke=[]
    torch.cuda.reset_peak_memory_stats()
    for step in range(3):
        tick=time.perf_counter();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,out,delta=m.loss(b)
        assert torch.isfinite(loss)
        loss.backward();gn=gradnorm(m);assert math.isfinite(gn)
        upstream=float(m.p_project.weight.grad.norm())
        if step>=1:
            if NO_PW:assert float(m.action_project.weight.grad.norm())>0,'No gradient reaches action projection'
            else:assert upstream>0,'No gradient reaches P projection'
        torch.nn.utils.clip_grad_norm_(m.parameters(),CLIP);opt.step();torch.cuda.synchronize()
        smoke.append(dict(step=step+1,loss=float(loss),gradient_norm=gn,P_projection_grad_norm=upstream,step_s=time.perf_counter()-tick))
        emit('SMOKE_STEP',**smoke[-1])
    if NO_PW:
        m.eval();changed=dict(b);changed['P']=torch.randn_like(b['P'])*100;changed['W']=torch.randn_like(b['W'])*100
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            reference=m(b)[0];perturbed=m(changed)[0]
        assert torch.equal(reference,perturbed),'No-P/W control leaks visual content'
        emit('NO_PW_CONTENT_INVARIANCE_PASSED')
    savejson('smoke.json',dict(status='passed',records=smoke,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,estimated_compute100_s=float(np.mean([r['step_s'] for r in smoke[1:]]))*100))
    del opt,m,b,batches;gc.collect();torch.cuda.empty_cache();seed();m=ARB(stats).cuda()
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.01)
    emit('PILOT_STARTED',batch=BATCH,steps=100)
    started=time.perf_counter();tick=time.perf_counter();baseline=evaluate(m,val_loader)
    assert abs(baseline['normalized_mse']-baseline['wam_normalized_mse'])<1e-6
    eval_s=time.perf_counter()-tick;emit('BASELINE_VALIDATION',seconds=eval_s,**baseline)
    history=[];validations=[dict(step=0,**baseline)];clipped=0;steps=[]
    for step,cpu in enumerate(train_loader,1):
        tick=time.perf_counter();b=device(cpu);m.train();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,out,delta=m.loss(b)
        assert torch.isfinite(loss)
        loss.backward();gn=gradnorm(m);assert math.isfinite(gn);clipped+=gn>CLIP
        torch.nn.utils.clip_grad_norm_(m.parameters(),CLIP);opt.step();torch.cuda.synchronize();step_s=time.perf_counter()-tick
        row=dict(step=step,loss=float(loss),gradient_norm=gn,clipped=gn>CLIP,step_s=step_s,delta_rms=float(delta.square().mean().sqrt()))
        with open(OUT/'metrics.jsonl','a') as f:f.write(json.dumps(row)+'\n')
        history.append(row);steps.append(step_s)
        if step<=3 or step%10==0:emit('TRAIN',**row)
        if step%10==0:savejson('progress.json',dict(status='running',step=step,total_steps=100,elapsed_s=time.perf_counter()-started,
            mean_step_s=float(np.mean(steps[-10:])),estimated_remaining_s=(100-step)*float(np.mean(steps[-10:]))+(2 if step<50 else 1)*eval_s))
        if step in (50,100):
            checkpoint=dict(model=m.state_dict(),optimizer=opt.state_dict(),step=step,config=json.load(open(OUT/'config.json')),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
            path=OUT/f'step{step}.pt';tmp=path.with_suffix('.tmp');torch.save(checkpoint,tmp);tmp.replace(path)
            tick=time.perf_counter();v=evaluate(m,val_loader);validations.append(dict(step=step,**v));emit('VALIDATION',step=step,seconds=time.perf_counter()-tick,**v)
            savejson('validation.json',validations)
        if step==100:break
    summary=dict(status='complete',steps=100,parameters=parameters,baseline=baseline,validation=validations,
        first10_train_mean=float(np.mean([r['loss'] for r in history[:10]])),last10_train_mean=float(np.mean([r['loss'] for r in history[-10:]])),
        clip_rate=clipped/100,mean_step_s=float(np.mean(steps)),elapsed_s=time.perf_counter()-started,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        checkpoint=str(OUT/'step100.pt'),no_pw=NO_PW,caveat='Offline action MSE; not rollout success.')
    savejson('summary.json',summary);savejson('progress.json',dict(status='complete',step=100,total_steps=100));emit('COMPLETE',**summary)
if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);savejson('failure.json',dict(status='failed',time=time.time(),traceback=traceback.format_exc()));raise
