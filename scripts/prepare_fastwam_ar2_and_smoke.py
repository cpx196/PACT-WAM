"""Resume-safe two-interval labels and five-step random-Predictor AR smoke."""
import sys,os,json,time,fcntl,traceback,types
from pathlib import Path
import numpy as np,torch,h5py
from PIL import Image
sys.path[:0]=['/data/pxchen/PACT-WAM/artifacts/predictor2.1ac1009','/data/pxchen/vjepa2']
from smoke_joint_predictor import JointPredictor,DATA
from vectorized_ac_attention import enable_vectorized
for package in ['prismatic','prismatic.models','prismatic.models.backbones','prismatic.models.backbones.vision']:
    module=types.ModuleType(package);module.__path__=['/data/pxchen/JEPA_WAM/'+package.replace('.','/')];sys.modules[package]=module
from prismatic.models.backbones.vision.vjepa_vit import VJEPA21ViTBackbone
OUT=Path('/data/pxchen/PACT-WAM/artifacts/fastwam_predictor21_ar2_stride20_20261014')
CKPT='/data/pxchen/PACT-WAM/checkpoints/jepa_wam_assets_20261009/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt'

def save(name,x):
    p=OUT/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def emit(event,**kw):print(json.dumps(dict(event=event,**kw)),flush=True)

class ARPredictor(JointPredictor):
    def __init__(self):
        super().__init__()
        self.action_steps=4
        self.action_position=torch.nn.Parameter(self.action_position[:,:4].detach().clone())
        self.phase=torch.nn.Embedding(2,1024);torch.nn.init.trunc_normal_(self.phase.weight,std=.02)
    def forward(self,z,a,s,phase):
        # Explicit interval identity; both steps share all model weights.
        return super().forward(z+self.phase.weight[phase][None,None],a,s)

def smoke(rows,stats):
    if (OUT/'smoke_summary.json').exists():return
    torch.manual_seed(713);np.random.seed(713)
    def batch(split):
        selected=[r for r in rows if r['split']==split][:2]
        assert len(selected)==2
        items=[]
        for row in selected:
            base=torch.load(DATA/row['file'],map_location='cpu',weights_only=False,mmap=True)
            labels=torch.load(OUT/row['target_file'],map_location='cpu',weights_only=False,mmap=True)
            items.append((base['current_latent'][0].float(),base['raw_actions'][0,:8],base['state'][0],labels['targets'][0].float()))
        z,a,s,y=[torch.stack([x[k] for x in items]).cuda() for k in range(4)]
        def norm(x,key):return (x-x.new_tensor(stats[key]['mean']))/x.new_tensor(stats[key]['std']).clamp_min(1e-6)
        return z,norm(a,'action'),norm(s,'state'),y
    train=batch('train');val=batch('val')
    m=ARPredictor().cuda();enable_vectorized(m)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.04)
    config=dict(initialization='random; no Predictor weights loaded',encoder='frozen VJEPA2.1 ViT-L384',
        parameters=sum(p.numel() for p in m.parameters()),width=1024,depth=24,heads=16,
        batch=2,optimizer_steps=5,action_shape=[8,7],state_shape=[8],output_shape=[2,576,1024],
        loss='0.5*L1(pred1,target1)+0.5*L1(pred2,target2); pure two-step autoregression',
        teacher_forcing=False,detach_between_steps=False,compile=False)
    save('smoke_config.json',config)
    def forward(b):
        z,a,s,y=b
        with torch.autocast('cuda',dtype=torch.bfloat16):
            p1=m(z,a[:,:4],s,0)
            p2=m(p1,a[:,4:8],s,1)
        l1=(p1.float()-y[:,0]).abs().mean();l2=(p2.float()-y[:,1]).abs().mean()
        return (l1+l2)*.5,l1,l2,p1,p2
    m.eval()
    with torch.no_grad():before=float(forward(val)[0])
    m.train();records=[];tick=time.perf_counter()
    for step in range(5):
        opt.zero_grad(set_to_none=True);loss,l1,l2,p1,p2=forward(train)
        assert p1.shape==p2.shape==(2,576,1024) and torch.isfinite(loss)
        cross=torch.autograd.grad(l2,p1,retain_graph=True)[0]
        cross_norm=float(cross.float().norm());assert np.isfinite(cross_norm) and cross_norm>0
        loss.backward()
        grads={name:float(layer.weight.grad.float().norm()) for name,layer in [('action',m.backbone.action_encoder),('state',m.backbone.state_encoder),('output',m.backbone.predictor_proj)]}
        assert all(np.isfinite(g) and g>0 for g in grads.values())
        gnorm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True))
        opt.step();torch.cuda.synchronize()
        rec=dict(step=step+1,loss=float(loss),step1_l1=float(l1),step2_l1=float(l2),
            step2_loss_gradient_to_step1_output_norm=cross_norm,parameter_gradient_norms=grads,
            global_gradient_norm_before_clip=gnorm)
        records.append(rec);emit('SMOKE_STEP',**rec)
        if step==1:
            m.eval()
            with torch.no_grad():reference=forward(val)[4].float().cpu()
            torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),step=2,config=config),OUT/'smoke_step2.pt')
            ck=torch.load(OUT/'smoke_step2.pt',map_location='cpu',weights_only=False)
            m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);del ck
            with torch.no_grad():restored=forward(val)[4].float().cpu()
            restore_error=float((reference-restored).abs().max());assert restore_error==0; m.train()
    m.eval()
    with torch.no_grad():after=float(forward(val)[0])
    torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),step=5,config=config),OUT/'smoke_step5.pt')
    result=dict(status='passed',config=config,records=records,val_loss_before=before,val_loss_after=after,
        checkpoint_restore_max_abs=restore_error,elapsed_s=time.perf_counter()-tick,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        note='Five optimizer steps verify AR supervision and gradient reachability, not a convergence run.')
    save('smoke_summary.json',result);emit('SMOKE_PASSED',validation_before=before,validation_after=after)
    del m,opt,train,val;torch.cuda.empty_cache()

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'shards').mkdir(exist_ok=True)
    lock=open(OUT/'run.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'summary.json').exists():emit('ALREADY_COMPLETE');return
    torch.set_num_threads(4)
    rows=json.load(open(DATA/'index.json'));assert len(rows)==5000
    config=dict(base_cache=str(DATA),windows=34995,train_windows=31495,val_windows=3500,
        encoder=CKPT,encoder_frozen=True,targets=[[0,4],[4,8]],action_prefix=8,
        state='XYZ + rotvec + two finger qpos',actions='raw controls; no integration or Euler conversion',
        view='main only',rotation='HDF5 rot180 once; upright JEPA input',
        image_transform='native PIL bicubic384 + ImageNet normalization',target_shape=[2,576,1024],
        storage_dtype='BF16 native encoder output',batch_size=1,stride=20,compile=False)
    if (OUT/'config.json').exists():assert json.load(open(OUT/'config.json'))==config
    else:save('config.json',config)
    if (OUT/'failure.json').exists():(OUT/'failure.json').rename(OUT/f'failure_previous_{int(time.time())}.json')
    if not (OUT/'train_only_stats.json').exists():
        accum={k:dict(n=0,total=np.zeros(d),sq=np.zeros(d)) for k,d in [('action',7),('state',8)]}
        emit('TRAIN_ONLY_STATS_START')
        for row in rows:
            if row['split']!='train':continue
            d=torch.load(DATA/row['file'],map_location='cpu',weights_only=False,mmap=True)
            for key,vals in [('action',d['raw_actions'][:,:8].reshape(-1,7)),('state',d['state'])]:
                v=vals.numpy().astype(np.float64);q=accum[key];q['n']+=len(v);q['total']+=v.sum(0);q['sq']+=(v*v).sum(0)
        stats={}
        for key,q in accum.items():
            mean=q['total']/q['n'];std=np.sqrt(np.maximum(q['sq']/q['n']-mean*mean,0))
            stats[key]=dict(mean=mean.tolist(),std=std.tolist(),count=q['n'])
        save('train_only_stats.json',stats);emit('TRAIN_ONLY_STATS_COMPLETE')
    stats=json.load(open(OUT/'train_only_stats.json'))
    encoder=VJEPA21ViTBackbone('vjepa2_1-vit-l-384px','resize-naive',checkpoint_path=CKPT).cuda().eval().requires_grad_(False)
    assert all(not p.requires_grad for p in encoder.parameters())
    def transform(a):return encoder.image_transform(Image.fromarray(np.ascontiguousarray(a[::-1,::-1])).convert('RGB'))
    start=time.perf_counter();done=0;new=0;manifest=[]
    for i,row in enumerate(rows):
        target_file='shards/'+Path(row['file']).name;p=OUT/target_file
        if p.exists():
            cached=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
            assert cached['source_shard']==row['file'] and cached['targets'].shape==(row['windows'],2,576,1024)
            del cached
        else:
            base=torch.load(DATA/row['file'],map_location='cpu',weights_only=False,mmap=True)
            values=[]
            with h5py.File(base['source'],'r') as f:
                rgb=f['data'][base['demo']]['obs/agentview_rgb']
                needed=sorted({int(t)+o for t in base['starts'] for o in [0,4,8]})
                frames={j:transform(np.asarray(rgb[j])) for j in needed}
                for j,t in enumerate(base['starts']):
                    t=int(t);labels=[]
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                        if i==0 and j==0:
                            z=encoder(frames[t][None].cuda()).float().cpu()
                            mismatch=float((z-base['current_latent'][j:j+1].float()).abs().max())
                            assert mismatch<.001,mismatch
                            save('encoder_smoke.json',dict(status='passed',current_cache_max_abs=mismatch,frozen=True))
                        for lo,hi in [(0,4),(4,8)]:
                            pair=torch.stack([frames[t+lo],frames[t+hi]])[None].cuda()
                            y=encoder.encode_pair(pair).reshape(576,1024).cpu().contiguous()
                            assert torch.isfinite(y).all();labels.append(y)
                    values.append(torch.stack(labels))
            tmp=p.with_suffix('.tmp');torch.save(dict(targets=torch.stack(values),starts=base['starts'],
                source_shard=row['file'],source=base['source'],demo=base['demo'],split=base['split']),tmp);tmp.replace(p)
            new+=row['windows']
        done+=row['windows'];manifest.append(dict(**row,target_file=target_file))
        if not (OUT/'smoke_summary.json').exists() and all(sum(r['split']==split for r in manifest)>=2 for split in ['train','val']):
            smoke(manifest,stats)
            # Free the allocator after smoke's local autograd graph is gone.
            torch.cuda.empty_cache()
        elapsed=time.perf_counter()-start
        progress=dict(status='running',completed_windows=done,total_windows=34995,completed_trajectories=i+1,
            elapsed_s=elapsed,new_windows=new,estimated_remaining_s=(34995-done)*elapsed/max(new,1),
            gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),smoke_passed=(OUT/'smoke_summary.json').exists())
        save('progress.json',progress)
        if i<3 or i%20==0:emit('PROGRESS',**progress)
    assert done==34995
    save('index.json',manifest);save('summary.json',dict(status='complete',windows=done,
        bytes=sum(p.stat().st_size for p in (OUT/'shards').glob('*.pt')),elapsed_s=time.perf_counter()-start,
        smoke_passed=(OUT/'smoke_summary.json').exists()))
    save('progress.json',dict(status='complete',completed_windows=done,total_windows=34995,smoke_passed=True));emit('COMPLETE')

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True);save('failure.json',dict(traceback=traceback.format_exc()));raise
