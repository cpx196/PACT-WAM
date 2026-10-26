"""Frozen epoch10 Predictor vs cached real interval latents; no rollout/training."""
import os,sys,time,json,fcntl,traceback,hashlib,collections
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/data/pxchen/PACT-WAM')
sys.path[:0]=[str(ROOT/'artifacts/predictor2.1ac1009'),str(ROOT/'scripts'),'/data/pxchen/vjepa2']
from smoke_joint_predictor import JointPredictor
from vectorized_ac_attention import enable_vectorized
CACHE=ROOT/'artifacts/fastwam_predictor21_ar2_stride20_20261014'
BASE=ROOT/'artifacts/predictor21_joint20_stride20_20261011'
CHECKPOINT=ROOT/'artifacts/fastwam_predictor21_ar2_full10epoch_bs16_20261014/step19690.pt'
OUT=ROOT/'artifacts/ar2_long642_alignment_probe_20261015'

class ARPredictor(JointPredictor):
    def __init__(self):
        super().__init__();self.action_steps=4
        self.action_position=torch.nn.Parameter(self.action_position[:,:4].detach().clone())
        self.phase=torch.nn.Embedding(2,1024)
    def forward(self,z,a,s,phase):return super().forward(z+self.phase.weight[phase][None,None],a,s)

def write(name,d):
    p=OUT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def log(event,**kw):print(json.dumps(dict(event=event,**kw)),flush=True)
def load(path):return torch.load(path,map_location='cpu',weights_only=False,mmap=True)

def pair_metrics(pred,target):
    # target may be [2,576,1024] or [K,2,576,1024]; final axes always space/channel.
    delta=pred-target
    l1=delta.abs().mean((-2,-1));rmse=delta.square().mean((-2,-1)).sqrt()
    dot=(pred*target).sum((-2,-1))
    cosine=dot/(pred.square().sum((-2,-1))*target.square().sum((-2,-1))).sqrt().clamp_min(1e-12)
    return l1,rmse,cosine

def cka(x,y):
    x=x.double();y=y.double();x=x-x.mean(0);y=y-y.mean(0)
    k=x@x.T;l=y@y.T
    return float((k*l).sum()/(k.square().sum()*l.square().sum()).sqrt().clamp_min(1e-20))

def block_bootstrap(values,metadata):
    groups=collections.defaultdict(list)
    for v,m in zip(values,metadata):
        if np.isfinite(v):groups[m['task']+'/'+m['demo']].append(v)
    means=np.array([np.mean(v) for v in groups.values()]);rng=np.random.default_rng(625)
    boot=means[rng.integers(len(means),size=(2000,len(means)))].mean(1)
    return dict(trajectory_count=len(means),trajectory_balanced_mean=float(means.mean()),ci95=np.percentile(boot,[2.5,97.5]).tolist())

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=open(OUT/'run.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'summary.json').exists(),'Refuse duplicate completed probe'
    torch.set_num_threads(4);torch.set_num_interop_threads(2)
    torch.manual_seed(625);np.random.seed(625)
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.12)
    torch.backends.cuda.matmul.allow_tf32=False
    allrows=json.loads((CACHE/'index.json').read_text())
    rows=[r for r in allrows if r['suite']=='libero10' and r['split']=='val']
    n=sum(r['windows'] for r in rows);assert n==642
    tasks=sorted({r['task'] for r in rows});assert len(tasks)==10
    z=torch.empty(n,576,1024);y=torch.empty(n,2,576,1024)
    a=torch.empty(n,8,7);s=torch.empty(n,8);meta=[];offset=0
    for row in rows:
        b=load(BASE/row['file']);t=load(CACHE/row['target_file']);count=row['windows']
        assert torch.equal(b['starts'],t['starts'])
        assert b['current_latent'].dtype==t['targets'].dtype==torch.float32
        z[offset:offset+count]=b['current_latent'];y[offset:offset+count]=t['targets']
        a[offset:offset+count]=b['raw_actions'][:,:8];s[offset:offset+count]=b['state']
        for j in range(count):meta.append(dict(task=row['task'],demo=row['demo'],start=int(b['starts'][j]),source=b['source'],shard=row['file'],window=j))
        offset+=count
    del b,t
    log('CACHE_LOADED',windows=n,trajectories=len(rows),tasks=10)
    # Same-space task-mean baseline, using TRAIN targets only, no validation fit.
    rng=np.random.default_rng(625);task_means=[];train_selection=[]
    for task in tasks:
        items=[(r,j) for r in allrows if r['suite']=='libero10' and r['split']=='train' and r['task']==task for j in range(r['windows'])]
        selected=rng.choice(len(items),min(32,len(items)),replace=False)
        mean=torch.zeros(2,576,1024);last=None;data=None
        for ix in selected:
            row,j=items[int(ix)]
            if row['target_file']!=last:data=load(CACHE/row['target_file']);last=row['target_file']
            mean+=data['targets'][j]
            train_selection.append(dict(task=task,file=last,window=j))
        task_means.append(mean/len(selected))
    task_means=torch.stack(task_means);del data,mean
    stats=json.loads((CACHE/'train_only_stats.json').read_text())
    norms={k:(torch.tensor(stats[k]['mean'],device='cuda'),torch.tensor(stats[k]['std'],device='cuda').clamp_min(1e-6)) for k in ('action','state')}
    model=ARPredictor();ck=load(CHECKPOINT);assert ck['step']==19690
    model.load_state_dict(ck['model'],strict=True);del ck
    model=model.cuda().eval().requires_grad_(False);enable_vectorized(model)
    config=dict(checkpoint=str(CHECKPOINT),step=19690,precision='FP32, TF32 disabled',encoder_recomputed=False,
        data='642 Long validation windows; 50 held-out demonstrations',all_tokens_preserved=True,
        inference_batch=1,device_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),cuda_fraction_cap=.12,
        teacher_forcing=False,autoregressive=True,action_steps=[4,4],state_dim=8,
        target_intervals=[[0,4],[4,8]],train_mean_baseline_samples_per_task=32,
        negative_samples_per_type=8,far_time_minimum_control_steps=40,
        no_wam_future=True,no_robot_rollout=True,no_training=True,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cka='64 fixed spatial tokens x1024 channels; diagnostic subsample only, model uses all576; both raw and within-task centered')
    write('config.json',config);write('metadata.json',meta);write('train_baseline_selection.json',train_selection)
    # Spatially spread 8x8 token grid, for bounded-memory representation CKA.
    grid=np.linspace(0,23,8,dtype=int);token_ids=torch.tensor([i*24+j for i in grid for j in grid])
    pf=np.empty((n,2,64*1024),np.float32);yf=np.empty_like(pf)
    matched=np.empty((n,2),np.float32);cos=np.empty_like(matched);rms=np.empty_like(matched)
    pearson=np.empty_like(matched);center_cos=np.empty_like(matched);baseline=np.empty_like(matched)
    ratios=np.empty_like(matched);means_p=np.empty_like(matched);means_y=np.empty_like(matched)
    std_p=np.empty_like(matched);std_y=np.empty_like(matched)
    types=['same_demo_far_time','same_task_other_demo','different_task']
    neg={k:np.full((n,8,2),np.nan,np.float32) for k in types}
    task_id=np.array([tasks.index(m['task']) for m in meta]);demo=np.array([m['demo'] for m in meta]);starts=np.array([m['start'] for m in meta])
    heat=torch.zeros(2,24,24);snapshots={};timer=time.perf_counter()
    with torch.inference_mode():
        for i in range(n):
            actions=a[i:i+1].cuda();state=s[i:i+1].cuda();am,ast=norms['action'];sm,sst=norms['state']
            actions=(actions-am)/ast;state=(state-sm)/sst
            p1=model(z[i:i+1].cuda(),actions[:,:4],state,0)
            p2=model(p1,actions[:,4:8],state,1)
            pred=torch.stack((p1[0],p2[0])).float().cpu();truth=y[i]
            ml,mr,mc=pair_metrics(pred,truth);matched[i]=ml.numpy();rms[i]=mr.numpy();cos[i]=mc.numpy()
            pm=pred.mean((-2,-1),keepdim=True);ym=truth.mean((-2,-1),keepdim=True)
            pearson[i]=pair_metrics(pred-pm,truth-ym)[2].numpy()
            ratios[i]=(pred.flatten(1).norm(dim=1)/truth.flatten(1).norm(dim=1).clamp_min(1e-12)).numpy()
            means_p[i]=pm.flatten().numpy();means_y[i]=ym.flatten().numpy()
            std_p[i]=pred.std((-2,-1),unbiased=False).numpy();std_y[i]=truth.std((-2,-1),unbiased=False).numpy()
            mean=task_means[task_id[i]]
            baseline[i]=pair_metrics(mean,truth)[0].numpy()
            center_cos[i]=pair_metrics(pred-mean,truth-mean)[2].numpy()
            candidates=[np.flatnonzero((task_id==task_id[i])&(demo==demo[i])&(np.abs(starts-starts[i])>=40)),
                        np.flatnonzero((task_id==task_id[i])&(demo!=demo[i])),np.flatnonzero(task_id!=task_id[i])]
            for kind,ids in zip(types,candidates):
                if len(ids):
                    chosen=rng.choice(ids,size=8,replace=len(ids)<8)
                    neg[kind][i]=pair_metrics(pred[None],y[torch.tensor(chosen)])[0].numpy()
            pf[i]=pred[:,token_ids].flatten(1).numpy();yf[i]=truth[:,token_ids].flatten(1).numpy()
            local=(pred-truth).abs().mean(-1).reshape(2,24,24);heat+=local
            if i==n//2:snapshots['middle']=dict(index=i,heat=local.numpy())
            if i%25==0 or i==n-1:
                elapsed=time.perf_counter()-timer
                progress=dict(status='inference',completed=i+1,total=n,elapsed_s=elapsed,
                    estimated_remaining_s=elapsed/(i+1)*(n-i-1),matched_l1=float(matched[:i+1].mean()),
                    peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                write('progress.json',progress);log('PROGRESS',**progress)
    del model,z,a,s,p1,p2;torch.cuda.empty_cache()
    # CUDA released before CPU-only statistics/plots.
    log('GPU_RELEASED')
    task_center_p=pf.copy();task_center_y=yf.copy()
    for tid in range(10):
        ids=task_id==tid
        task_center_p[ids]-=task_center_p[ids].mean(0)
        task_center_y[ids]-=task_center_y[ids].mean(0)
    representation=[]
    for k in range(2):
        permutation=rng.permutation(n)
        representation.append(dict(interval=k+1,raw_cka=cka(torch.from_numpy(pf[:,k]),torch.from_numpy(yf[:,k])),
            within_task_centered_cka=cka(torch.from_numpy(task_center_p[:,k]),torch.from_numpy(task_center_y[:,k])),
            shuffled_within_task_centered_cka=cka(torch.from_numpy(task_center_p[:,k]),torch.from_numpy(task_center_y[permutation,k]))))
    negatives={}
    for kind,values in neg.items():
        wrong=values.mean(2);correct=matched.mean(1);valid=np.isfinite(wrong).all(1)
        gap=wrong.mean(1)-correct
        negatives[kind]=dict(windows=int(valid.sum()),matched_l1=float(correct[valid].mean()),wrong_l1=float(wrong[valid].mean()),
            matched_beats_wrong_pct=float(100*(correct[valid,None]<wrong[valid]).mean()),
            matched_ranks_first_of9_pct=float(100*(correct[valid]<wrong[valid].min(1)).mean()),
            trajectory_bootstrap_wrong_minus_correct=block_bootstrap(gap,meta))
    result=dict(status='complete',scope='Predictor vs REAL future only; no WAM latent and no online success inference',
        windows=n,trajectories=len(rows),tasks=10,checkpoint=str(CHECKPOINT),precision='FP32',
        overall=dict(matched_l1=float(matched.mean()),task_mean_baseline_l1=float(baseline.mean()),
            task_mean_baseline_improvement_pct=float(100*(baseline.mean()-matched.mean())/baseline.mean()),
            rmse=float(rms.mean()),cosine=float(cos.mean()),per_window_flattened_pearson=float(pearson.mean()),
            cosine_after_subtracting_train_task_mean=float(center_cos.mean()),
            norm_ratio_mean=float(ratios.mean()),norm_ratio_median=float(np.median(ratios)),
            prediction_mean=float(means_p.mean()),target_mean=float(means_y.mean()),
            prediction_std=float(std_p.mean()),target_std=float(std_y.mean())),
        by_interval=[dict(interval=k+1,l1=float(matched[:,k].mean()),cosine=float(cos[:,k].mean()),
            task_centered_cosine=float(center_cos[:,k].mean()),norm_ratio=float(ratios[:,k].mean())) for k in range(2)],
        negative_controls=negatives,representation=representation,
        resource=dict(peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            elapsed_inference_and_stats_s=time.perf_counter()-timer),
        limitations=['Held out for Predictor only; WAM pretraining overlap not established.',
            'Correct latent pairing does not validate action gradients or WAM goal correctness.',
            'CKA uses64 spatial positions; all576 used for inference and distance metrics.',
            'Window dependence addressed by trajectory-level bootstrap; 8 negative draws are not 8 independent trials.'])
    np.savez_compressed(OUT/'window_metrics.npz',matched=matched,cosine=cos,pearson=pearson,center_cos=center_cos,
        baseline=baseline,ratios=ratios,means_p=means_p,means_y=means_y,std_p=std_p,std_y=std_y,
        **{f'negative_{k}':v for k,v in neg.items()},mean_heatmap=(heat/n).numpy())
    plot(matched,neg,ratios,representation,(heat/n).numpy(),result)
    write('summary.json',result);write('progress.json',dict(status='complete',completed=n,total=n));log('COMPLETE',summary=result)

def plot(matched,neg,ratios,reps,heat,summary):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(12,9))
    labels=['Matched','Wrong time, same demo','Other demo, same task','Other task']
    data=[matched.mean(1)]+[v.mean((1,2)) for v in neg.values()]
    axs[0,0].boxplot([x[np.isfinite(x)] for x in data],labels=labels,showfliers=False,
        medianprops={'color':'black'},boxprops={'color':'black'})
    axs[0,0].tick_params(axis='x',labelrotation=12);axs[0,0].set_ylabel('Latent L1 (lower is better)');axs[0,0].set_title('A. Correct vs mismatched future')
    xx=matched.mean(1);yy=neg['same_demo_far_time'].mean((1,2))
    axs[0,1].scatter(xx,yy,s=10,c='0.35',alpha=.45,rasterized=True)
    lo=float(np.nanmin([xx,yy]));hi=float(np.nanmax([xx,yy]));axs[0,1].plot([lo,hi],[lo,hi],'--',c='0.65')
    axs[0,1].set(xlabel='Matched L1',ylabel='Wrong-time L1',title='B. Above diagonal = correct pairing is closer')
    for k,style in enumerate(['-','--']):
        axs[1,0].hist(ratios[:,k],bins=35,histtype='step',color='black',linestyle=style,label=f'Interval {k+1}')
    axs[1,0].axvline(1,c='0.65',ls=':');axs[1,0].set(xlabel='Prediction / target latent norm',ylabel='Windows',title='C. Output scale');axs[1,0].legend(frameon=False)
    x=np.arange(2);width=.24
    for j,(name,key,color) in enumerate([('Raw','raw_cka','0.25'),('Task-centered','within_task_centered_cka','0.55'),('Shuffled centered','shuffled_within_task_centered_cka','0.8')]):
        axs[1,1].bar(x+(j-1)*width,[r[key] for r in reps],width,label=name,color=color)
    axs[1,1].set_xticks(x,['Interval 1','Interval 2']);axs[1,1].set(ylabel='Linear CKA',ylim=(0,1.05),title='D. Representation correspondence (64 spatial tokens)');axs[1,1].legend(frameon=False)
    fig.suptitle('Frozen AR2 Predictor — 642 Long validation windows / 50 demonstrations',fontsize=14)
    fig.tight_layout();fig.savefig(OUT/'alignment_overview.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,4))
    vmax=float(heat.max())
    for k,ax in enumerate(axs):
        im=ax.imshow(heat[k],cmap='Greys',vmin=0,vmax=vmax,origin='upper');ax.set_title(f'Interval {k+1}: mean token L1');ax.set_xticks([0,8,16,23]);ax.set_yticks([0,8,16,23])
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.8,label='L1');fig.savefig(OUT/'spatial_error.png',dpi=180,bbox_inches='tight');plt.close(fig)
    log('PLOTS_READY')

if __name__=='__main__':
    try:main()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True);write('failure.json',dict(traceback=traceback.format_exc()));raise
