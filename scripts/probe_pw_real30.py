"""Offline, frozen, native cached W vs final AR2 P vs true future. No rollout."""
import os,sys,json,time,fcntl,traceback,ast,math
from pathlib import Path
import numpy as np,torch,h5py
from PIL import Image
from scipy.stats import spearmanr
ROOT=Path('/data/pxchen/PACT-WAM')
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from fastwam_ar2_eval import ARRanker
OLD=ROOT/'artifacts/fastwam_native_oracle_future30_20261008'
CACHE=ROOT/'artifacts/libero10_predictor_action_oneepoch_20260930'
DATA=ROOT/'artifacts/libero10_guidance_backward_smoke_20260929/data/libero_10'
OUT=ROOT/'artifacts/ar2_pw_real30_alignment_20261015'
CKPT=ROOT/'artifacts/fastwam_predictor21_ar2_full10epoch_bs16_20261014/step19690.pt'

def emit(event,**kw):print(json.dumps(dict(event=event,**kw)),flush=True)
def save(name,x):
    p=OUT/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def load(p):return torch.load(p,map_location='cpu',weights_only=False,mmap=True)
def l1(p,q):return float((p-q).abs().mean())
def cos(p,q):return float(torch.nn.functional.cosine_similarity(p.flatten()[None],q.flatten()[None]))
def corr(x,y):
    if np.std(x)<1e-12 or np.std(y)<1e-12:return None
    return float(spearmanr(x,y).statistic)
def boot_corr(x,y):
    # One selected window per demo; resample tasks (10 clusters), not scalar nodes.
    rng=np.random.default_rng(15);groups=np.repeat(np.arange(10),3);values=[]
    for _ in range(2000):
        ids=np.concatenate([np.flatnonzero(groups==j) for j in rng.integers(10,size=10)])
        v=corr(np.asarray(x)[ids],np.asarray(y)[ids])
        if v is not None:values.append(v)
    return np.percentile(values,[2.5,97.5]).tolist()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=open(OUT/'run.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'summary.json').exists(),'already complete'
    torch.set_num_threads(4);torch.manual_seed(15)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Reuse the exact online quaternion conversion, without importing the simulator.
    tree=ast.parse((ROOT/'experiments/libero/libero_utils.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='quat2axisangle')
    ns={'np':np,'math':math};exec(compile(ast.Module(body=[fn],type_ignores=[]),'quat_fn','exec'),ns)
    ranker=ARRanker(str(CKPT),'cuda',torch.float32)
    pairs=sorted(OLD.glob('paired_*.pt'));assert len(pairs)==30
    conf=dict(checkpoint=str(CKPT),windows=30,precision='FP32; no compile; all 576 tokens',
        source=str(OLD),cached_generated_future=True,new_wam_generation=False,robot_rollout=False,training=False,
        offsets=[0,4,8],targets=['E([t,t+4])','E([t+4,t+8])'],camera='main only, upright',
        state='XYZ + exact online quat2axisangle + 2 gripper qpos',
        candidates='five offline convex interpolations from native executable action to expert, not executed',
        caution='Expert proximity is an offline proxy, not a true outcome label for candidate actions')
    save('config.json',conf);records=[];latents=[];previews=[];begin=time.monotonic()
    alphas=np.linspace(0,1,5,dtype=np.float32)
    for j,path in enumerate(pairs):
        pair=load(path);sample=pair['sample'];row=sample['row']
        b=load(CACHE/'wam'/f'{sample["index"]:04d}.pt');assert b['row']==row
        assert sample['frame_offsets'][:3]==[0,4,8] and len(b['future'])==3
        with h5py.File(DATA/row['task_file'],'r') as f:
            demo=f['data'][row['demo']];t=row['t']
            expert=torch.from_numpy(np.asarray(demo['actions'][t:t+8],dtype=np.float32))
            raw=[np.asarray(demo['obs']['agentview_rgb'][t+d]) for d in (0,4,8)]
        assert np.array_equal(raw[0],b['obs']['agentview_image']), 'current/time mismatch'
        real=[np.ascontiguousarray(x[::-1,::-1]) for x in raw]
        wam=[np.asarray(x)[:,:224].copy() for x in b['future']]
        assert all(x.shape==(224,224,3) for x in wam)
        # Current native cache is explicitly upright. Verify no accidental double rotation.
        assert np.array_equal(real[0],b['context'][-1])
        state=np.concatenate([b['obs']['robot0_eef_pos'],ns['quat2axisangle'](b['obs']['robot0_eef_quat'].copy()),b['obs']['robot0_gripper_qpos']]).astype(np.float32)
        assert state.shape==(8,)
        scale=pair['scale'].reshape(-1);offset=pair['offset'].reshape(-1)
        # Verify cached expert labels against raw HDF5 controls before normalization.
        assert torch.allclose(expert*scale+offset,pair['expert'],atol=1e-6,rtol=1e-5)
        native=(pair['actions']['wam_native'][:8]-offset)/scale
        native[:,6]=1-2*native[:,6]  # exact online differentiable guidance convention
        candidates=torch.stack([(1-float(alpha))*native+float(alpha)*expert for alpha in alphas]).cuda()
        def pix(x):return ranker.pixels(Image.fromarray(x)).cuda()
        def encode_video(frames):
            frames=[pix(x) for x in frames]
            inp=torch.stack([torch.stack(frames[:2]),torch.stack(frames[1:])])
            return ranker.encoder.encode_pair(inp).reshape(2,576,1024).float()
        with torch.inference_mode():
            z=ranker.encoder(pix(real[0])[None]).float()
            w=encode_video(wam);y=encode_video(real)
            resized=[np.asarray(Image.fromarray(x).resize((224,224),Image.Resampling.BICUBIC)) for x in real]
            yr=encode_video(resized)
            am,astd=ranker.stats['action'];sm,sstd=ranker.stats['state']
            actions=(candidates-am)/astd;states=(torch.tensor(state,device='cuda')[None]-sm)/sstd
            p1=ranker.predictor(z.expand(5,-1,-1),actions[:,:4],states.expand(5,-1),0)
            p2=ranker.predictor(p1,actions[:,4:],states.expand(5,-1),1)
            p=torch.stack([p1,p2],dim=1)
            ew=(p-w[None]).abs().mean((1,2,3)).cpu().numpy()
            ey=(p-y[None]).abs().mean((1,2,3)).cpu().numpy()
            mse=(candidates[:,:,:6]-expert.cuda()[None,:,:6]).square().mean((1,2)).cpu().numpy()
            desired=p[-1]-p[0]
            rec=dict(window=j,source_index=sample['index'],task=row['task_file'],demo=row['demo'],t=t,
                p_native_w=l1(p[0],w),p_expert_w=l1(p[-1],w),p_native_real=l1(p[0],y),p_expert_real=l1(p[-1],y),
                w_real=l1(w,y),resize_only_l1=l1(yr,y),
                copy_real=l1(z.expand(2,-1,-1),y),p_native_w_cosine=cos(p[0],w),w_real_cosine=cos(w,y),
                p_native_w_norm_ratio=float(p[0].norm()/w.norm()),w_real_norm_ratio=float(w.norm()/y.norm()),
                expert_latent_direction_cosine_w=cos(desired,w-p[0]),expert_latent_direction_cosine_real=cos(desired,y-p[0]),
                alphas=alphas.tolist(),candidate_w_l1=ew.tolist(),candidate_real_l1=ey.tolist(),candidate_action_mse=mse.tolist(),
                within_window_rho_w_mse=corr(ew,mse),within_window_rho_real_mse=corr(ey,mse))
            latents.append(torch.stack([p[0],p[-1],w,y,z[0].expand(2,-1,-1)]).cpu())
        records.append(rec)
        if j in (0,10,20):previews.append((row,real,wam))
        save('progress.json',dict(completed=j+1,total=30,elapsed_s=time.monotonic()-begin))
        with (OUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        emit('WINDOW',**rec)
    del ranker;torch.cuda.empty_cache()
    allz=torch.stack(latents);p=allz[:,0];w=allz[:,2];y=allz[:,3]
    wrong=np.roll(np.arange(30),3)
    per={k:float(np.mean([r[k] for r in records])) for k in ['p_native_w','p_expert_w','p_native_real','p_expert_real','w_real','resize_only_l1','copy_real','p_native_w_cosine','w_real_cosine','p_native_w_norm_ratio','w_real_norm_ratio','expert_latent_direction_cosine_w','expert_latent_direction_cosine_real']}
    ew=np.array([r['candidate_w_l1'] for r in records]);ey=np.array([r['candidate_real_l1'] for r in records]);mse=np.array([r['candidate_action_mse'] for r in records])
    rho=corr(ew[:,0],mse[:,0]);rho_y=corr(ew[:,0],ey[:,0])
    result=dict(status='complete',windows=30,tasks=10,demonstrations=30,means=per,
        expert_vs_native=dict(w_loss_decreased_windows=int((ew[:,-1]<ew[:,0]).sum()),real_loss_decreased_windows=int((ey[:,-1]<ey[:,0]).sum())),
        correlations=dict(across_window_w_loss_vs_action_mse=rho,task_bootstrap_ci95=boot_corr(ew[:,0],mse[:,0]),
            across_window_w_loss_vs_predictor_real_l1=rho_y,
            median_within_window_w_vs_mse=float(np.median([r['within_window_rho_w_mse'] for r in records])),
            median_within_window_real_vs_mse=float(np.median([r['within_window_rho_real_mse'] for r in records]))),
        wrong_task_controls=dict(p_native_wrong_w=l1(p,w[wrong]),w_wrong_real=l1(w,y[wrong])),
        seconds=time.monotonic()-begin,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,
        limits=['Only30 cached offline demo windows; candidates not executed.',
            'P(expert) vs true future is physically paired; P(native) vs expert future is not a true outcome error.',
            'No fitted latent mapping. Real images128px vs generated224px; resize control is not full VAE control.'])
    np.savez_compressed(OUT/'compact_metrics.npz',ew=ew,ey=ey,mse=mse)
    plot(records,result,previews)
    save('summary.json',result);emit('COMPLETE',**result)

def plot(records,summary,previews):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    keys=['p_expert_real','p_native_real','p_native_w','w_real','copy_real','resize_only_l1']
    vals=[summary['means'][k] for k in keys]
    bars=axs[0,0].barh(range(6),vals,color=['.2','.4','.55','.65','.75','.85'])
    axs[0,0].set_yticks(range(6),['P(expert) - Real','P(native) - Real','P(native) - W','W - Real','Copy current - Real','Resize only']);axs[0,0].invert_yaxis();axs[0,0].set_xlim(0,max(vals)*1.22)
    for b,v in zip(bars,vals):axs[0,0].text(v+.008,b.get_y()+b.get_height()/2,f'{v:.3f}',va='center')
    axs[0,0].set_title('A. Latent L1 (all 576 tokens)')
    x=[r['p_native_w'] for r in records];y=[r['candidate_action_mse'][0] for r in records]
    axs[0,1].scatter(x,y,color='.3');axs[0,1].set(xlabel='P(native) - W latent L1',ylabel='Native action MSE vs expert (first 6D)',title=f'B. Across windows: Spearman rho = {summary["correlations"]["across_window_w_loss_vs_action_mse"]:.2f}')
    for key,style,label in [('candidate_w_l1','-','Distance to W'),('candidate_real_l1','--','Distance to real future')]:
        d=np.array([r[key] for r in records]);change=d-d[:,0:1]
        axs[1,0].plot(np.linspace(0,1,5),change.mean(0),style,color='black',label=label)
        se=change.std(0)/np.sqrt(30);axs[1,0].fill_between(np.linspace(0,1,5),change.mean(0)-se,change.mean(0)+se,color='.7',alpha=.2)
    axs[1,0].axhline(0,color='.7',lw=1);axs[1,0].set(xlabel='Offline interpolation: native action (0) -> expert (1)',ylabel='Change in latent L1 vs native',title='C. Does moving toward expert lower the loss?');axs[1,0].legend(frameon=False)
    xx=np.array([r['expert_latent_direction_cosine_w'] for r in records]);yy=np.array([r['expert_latent_direction_cosine_real'] for r in records])
    axs[1,1].scatter(xx,yy,c='.3');axs[1,1].axvline(0,c='.7',lw=1);axs[1,1].axhline(0,c='.7',lw=1);axs[1,1].set(xlabel='Expert-induced latent change vs W target direction',ylabel='Expert-induced latent change vs real target direction',title='D. Direction agreement (not action gradient)')
    fig.suptitle('Final epoch-10 Predictor / cached WAM / real future | 30 Long windows',fontsize=13)
    fig.tight_layout();fig.savefig(OUT/'pw_real_overview.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(6,3,figsize=(8,13))
    for i,(row,real,wam) in enumerate(previews):
        for k in range(3):
            axs[2*i,k].imshow(real[k]);axs[2*i+1,k].imshow(wam[k]);axs[2*i,k].set_title(f'Real t+{k*4}');axs[2*i+1,k].set_title(f'WAM t+{k*4}')
            axs[2*i,k].axis('off');axs[2*i+1,k].axis('off')
    fig.tight_layout();fig.savefig(OUT/'paired_images.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    try:main()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True);save('failure.json',dict(traceback=traceback.format_exc()));raise
