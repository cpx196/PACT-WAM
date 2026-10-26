"""Bounded 100-window cache benchmark. No training or simulator."""
import os,sys,json,time,traceback,fcntl
from pathlib import Path
import h5py,numpy as np,torch
from PIL import Image
sys.path.insert(0,'/data/pxchen/JEPA_WAM')
sys.path.insert(0,'/data/pxchen/PACT-WAM/artifacts/predictor2.1ac1009')
from prismatic.models import load_vla
from smoke_joint_predictor import JointPredictor

ROOT=Path('/data/pxchen/PACT-WAM')
DATA=ROOT/'artifacts/predictor21_joint20_stride20_20261011'
OUT=Path(os.environ.get('CACHE_BENCH_OUT',str(ROOT/'artifacts/jepawam_pw_cache_benchmark100_20261012')))
ASSETS=ROOT/'checkpoints/jepa_wam_assets_20261009'
PCKPT=ROOT/'artifacts/predictor21_joint20_train10epoch_bs16_gpu3_20261011/step19690.pt'

def emit(event,**kw):print(json.dumps(dict(event=event,time=time.time(),**kw)),flush=True)
def savejson(name,obj):
    p=OUT/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,indent=2));q.replace(p)
def main():
    OUT.mkdir(exist_ok=True);lock=open(OUT/'run.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'summary.json').exists():raise RuntimeError('Already complete')
    (OUT/'pid').write_text(str(os.getpid()))
    torch.set_num_threads(2);torch.manual_seed(7);np.random.seed(7)
    assert torch.cuda.is_available()
    rows=json.load(open(DATA/'index.json'))
    # One window from each task, stratified between suites; original trajectory split retained.
    selected=[];seen=set()
    for r in rows:
        key=(r['suite'],r['task'])
        if key not in seen:seen.add(key);selected.append(r)
    assert len(selected)==100
    policy=next((ASSETS/'JEPA_WAM/checkpoints/libero').rglob('step-040000-epoch-37-loss=0.0262.pt'))
    started=time.perf_counter()
    m=load_vla(policy,load_for_training=False,
        base_vlm=str(ASSETS/'JEPA_WAM/checkpoints/pretrained_vlm/prism-qwen25-vjepa21-vitl-384px+0_5b+stage-finetune+x7'),
        llm_checkpoint_path=str(ASSETS/'Qwen2.5-0.5B'),
        vjepa_checkpoint_path=str(ASSETS/'vjepa2/vjepa2_1_vitl_dist_vitG_384.pt'),
        load_visual_token_cosine_head=True).eval().cuda().requires_grad_(False)
    raw=torch.load(policy,map_location='cpu',weights_only=False,mmap=True)['model']['visual_token_cosine_head']
    m.visual_token_cosine_head.load_state_dict(raw,strict=True)
    assert all(torch.equal(v.detach().cpu(),raw[k]) for k,v in m.visual_token_cosine_head.state_dict().items())
    del raw
    pred=JointPredictor().eval().requires_grad_(False)
    q=torch.load(PCKPT,map_location='cpu',weights_only=False,mmap=True)
    pred.load_state_dict(q['model'],strict=True);del q;pred.cuda()
    vectorized=os.environ.get('VECTORIZED_AC','0')=='1'
    if vectorized:
        from vectorized_ac_attention import enable_vectorized
        enable_vectorized(pred)
    stats=json.load(open(DATA/'train_only_stats.json'))
    norm={k:(torch.tensor(v['mean'],device='cuda'),torch.tensor(v['std'],device='cuda').clamp_min(1e-6)) for k,v in stats.items()}
    # Use the target online LIBERO10 policy contract for all training tasks.
    key=m._check_unnorm_key(m.norm_stats,'libero_10_no_noops')
    captured={}
    def hook(module,args,output):
        h=output['llm_hidden'];assert h.shape[1]>=1153
        with torch.autocast('cuda',dtype=torch.bfloat16):
            captured['W']=m.visual_token_cosine_head.align_dimension(h[:,1:577]).float()
    handle=m.register_forward_hook(hook)
    torch.cuda.synchronize();load_s=time.perf_counter()-started
    savejson('config.json',dict(windows=100,gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),policy=str(policy),predictor=str(PCKPT),
        source=str(DATA),precision='native BF16 autocast; FP32 cache',wam_views=['primary','wrist'],W='first576 primary visual positions; trained alignment head strict-loaded',
        action='native output unnormalization followed by native gripper binarization/inversion',predictor_input='WAM executable actions; train-only stats',
        vectorized_ac=vectorized,compile=False,unnorm_key=key,sampling='one window per task, trajectories read sequentially; not a warmed trajectory-prefetch throughput estimate',seed=7))
    emit('MODELS_READY',load_s=load_s,allocated_mib=torch.cuda.memory_allocated()/2**20)
    times=[];totalbytes=0;runstart=time.perf_counter()
    for i,r in enumerate(selected):
        tick=time.perf_counter()
        d=torch.load(DATA/r['file'],map_location='cpu',weights_only=False,mmap=True);j=0;t=int(d['starts'][j])
        with h5py.File(d['source'],'r') as f:
            g=f['data'][d['demo']];obs=g['obs'];instruction=json.loads(f['data'].attrs['problem_info'])['language_instruction']
            images=[Image.fromarray(np.ascontiguousarray(obs[k][t][::-1,::-1])) for k in ('agentview_rgb','eye_in_hand_rgb')]
            state=np.concatenate((obs['ee_states'][t],obs['gripper_states'][t])).astype(np.float32)
        assert np.allclose(state,d['state'][j].numpy(),atol=1e-6)
        z=d['current_latent'][j:j+1].cuda();s=torch.from_numpy(state).cuda()[None];expert=d['raw_actions'][j].clone()
        read_s=time.perf_counter()-tick;torch.cuda.synchronize();begin=time.perf_counter()
        with torch.inference_mode():
            action=m.predict_action(image=images,instruction=instruction,unnorm_key=key,proprio=state)
        assert action.shape==(20,7)
        action=action.astype(np.float32);action[:,-1]=-np.sign(2*action[:,-1]-1)
        torch.cuda.synchronize();wam_s=time.perf_counter()-begin;begin=time.perf_counter()
        a=torch.from_numpy(action).cuda()[None];am,asd=norm['action'];sm,ssd=norm['state']
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):p=pred(z,(a-am)/asd,(s-sm)/ssd)
        torch.cuda.synchronize();pred_s=time.perf_counter()-begin;begin=time.perf_counter()
        w=captured.pop('W');assert w.shape==p.shape==(1,576,1024)
        assert all(bool(torch.isfinite(v).all()) for v in (w,p,a))
        out=dict(P=p[0].float().cpu(),W=w[0].float().cpu(),wam_action=torch.from_numpy(action),state=torch.from_numpy(state),
            expert_action=expert,source_shard=r['file'],window_index=j,start=t,split=r['split'],instruction=instruction)
        path=OUT/f'{i:03d}.pt';tmp=path.with_suffix('.tmp');torch.save(out,tmp);tmp.replace(path)
        record=dict(index=i,read_s=read_s,wam_s=wam_s,predictor_s=pred_s,write_s=time.perf_counter()-begin,total_s=time.perf_counter()-tick)
        times.append(record);totalbytes+=path.stat().st_size
        with open(OUT/'metrics.jsonl','a') as log:log.write(json.dumps(record)+'\n')
        if (i+1)%10==0 or i==0:emit('PROGRESS',completed=i+1,total=100,last=record)
    handle.remove();elapsed=time.perf_counter()-runstart
    warm=times[5:];mean=float(np.mean([x['total_s'] for x in warm]))
    result=dict(status='complete',windows=100,load_s=load_s,elapsed_s=elapsed,mean_s_per_window=mean,
        median_s_per_window=float(np.median([x['total_s'] for x in warm])),estimated_full_hours=mean*34995/3600,
        bytes=totalbytes,estimated_full_bytes=totalbytes/100*34995,
        stage_mean_s={k:float(np.mean([x[k] for x in warm])) for k in ('read_s','wam_s','predictor_s','write_s')},
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        caveat='Shared GPU, batch1; one cold-ish trajectory read per window. Full sequential trajectory reuse may improve IO.')
    savejson('summary.json',result)
    if (OUT/'failure.json').exists():(OUT/'failure.json').unlink()
    emit('COMPLETE',**result)
if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);savejson('failure.json',dict(status='failed',traceback=traceback.format_exc()));raise
