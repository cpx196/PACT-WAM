"""Resume-safe, trajectory-sharded P/W cache. Single GPU, no compile or training."""
import sys,os,json,time,fcntl,traceback
from pathlib import Path
import h5py,numpy as np,torch
from PIL import Image
sys.path.insert(0,'/data/pxchen/JEPA_WAM')
sys.path.insert(0,'/data/pxchen/PACT-WAM/artifacts/predictor2.1ac1009')
from prismatic.models import load_vla
from smoke_joint_predictor import JointPredictor,DATA
from vectorized_ac_attention import enable_vectorized
ROOT=Path('/data/pxchen/PACT-WAM')
OUT=ROOT/'artifacts/jepawam_pw_joint20_stride20_full_20261012'
ASSETS=ROOT/'checkpoints/jepa_wam_assets_20261009'
PCKPT=ROOT/'artifacts/predictor21_joint20_train10epoch_bs16_gpu3_20261011/step19690.pt'
def emit(event,**kwargs):print(json.dumps(dict(event=event,time=time.time(),**kwargs)),flush=True)
def savejson(name,d):
    p=OUT/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def main():
    OUT.mkdir(exist_ok=True);(OUT/'shards').mkdir(exist_ok=True)
    lock=open(OUT/'run.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'summary.json').exists():emit('ALREADY_COMPLETE');return
    (OUT/'pid').write_text(str(os.getpid()))
    torch.set_num_threads(2);torch.manual_seed(7);np.random.seed(7)
    rows=json.load(open(DATA/'index.json'));assert len(rows)==5000
    total=sum(r['windows'] for r in rows);assert total==34995
    policy=next((ASSETS/'JEPA_WAM/checkpoints/libero').rglob('step-040000-epoch-37-loss=0.0262.pt'))
    config=dict(source=str(DATA),policy=str(policy),predictor=str(PCKPT),windows=total,trajectories=len(rows),
        train_windows=31495,val_windows=3500,stride=20,action_length=20,target_offset=31,
        wam_views=['primary','wrist'],W='main-view visual positions 1:577 through strict-loaded trained alignment head',
        unnorm_key='libero_10_no_noops',action='raw executable LIBERO command; native gripper binarization and inversion',
        predictor_mode='frozen eval; vectorized AC QKV; native RoPE; train-only normalization; step19690',
        precision='BF16 autocast FP32 weights and P/W cache',batch_size=1,compile=False,
        sampling='all original windows, trajectory-disjoint original train/val split',
        seed='7 + global manifest window index',real_future_input=False,reference_base_cache=True)
    if (OUT/'config.json').exists():assert json.load(open(OUT/'config.json'))==config,'Cache contract mismatch'
    else:savejson('config.json',config)
    emit('LOADING_MODELS',gpu=int(os.environ['CUDA_VISIBLE_DEVICES']))
    started=time.perf_counter()
    m=load_vla(policy,load_for_training=False,
        base_vlm=str(ASSETS/'JEPA_WAM/checkpoints/pretrained_vlm/prism-qwen25-vjepa21-vitl-384px+0_5b+stage-finetune+x7'),
        llm_checkpoint_path=str(ASSETS/'Qwen2.5-0.5B'),vjepa_checkpoint_path=str(ASSETS/'vjepa2/vjepa2_1_vitl_dist_vitG_384.pt'),
        load_visual_token_cosine_head=True).eval().cuda().requires_grad_(False)
    raw=torch.load(policy,map_location='cpu',weights_only=False,mmap=True)['model']['visual_token_cosine_head']
    m.visual_token_cosine_head.load_state_dict(raw,strict=True)
    assert all(torch.equal(v.detach().cpu(),raw[k]) for k,v in m.visual_token_cosine_head.state_dict().items());del raw
    pred=JointPredictor().eval().requires_grad_(False)
    q=torch.load(PCKPT,map_location='cpu',weights_only=False,mmap=True);pred.load_state_dict(q['model'],strict=True);del q
    pred.cuda();enable_vectorized(pred)
    stats=json.load(open(DATA/'train_only_stats.json'))
    norm={k:(torch.tensor(v['mean'],device='cuda'),torch.tensor(v['std'],device='cuda').clamp_min(1e-6)) for k,v in stats.items()}
    key=m._check_unnorm_key(m.norm_stats,config['unnorm_key']);captured={}
    def hook(module,args,output):
        hidden=output['llm_hidden'];assert hidden.shape[1]>=1153
        with torch.autocast('cuda',dtype=torch.bfloat16):captured['W']=m.visual_token_cosine_head.align_dimension(hidden[:,1:577]).float()
    handle=m.register_forward_hook(hook);torch.cuda.synchronize()
    emit('MODELS_READY',load_s=time.perf_counter()-started,allocated_mib=torch.cuda.memory_allocated()/2**20)
    runstart=time.perf_counter();processed=0;done=0;global_start=0;new_windows=0;manifest=[];bytes_total=0
    for ri,row in enumerate(rows):
        path=OUT/'shards'/Path(row['file']).name
        if path.exists():
            cached=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            assert cached['source_shard']==row['file'] and cached['split']==row['split']
            assert cached['P'].shape==cached['W'].shape==(row['windows'],576,1024)
            del cached;done+=row['windows'];global_start+=row['windows'];bytes_total+=path.stat().st_size
            manifest.append(dict(**row,cache_file=str(path.relative_to(OUT))));continue
        tick=time.perf_counter();d=torch.load(DATA/row['file'],map_location='cpu',weights_only=False,mmap=True)
        assert d['split']==row['split'];assert len(d['starts'])==row['windows']
        ps=[];ws=[];actions=[]
        with h5py.File(d['source'],'r') as f:
            g=f['data'][d['demo']];obs=g['obs'];instruction=json.loads(f['data'].attrs['problem_info'])['language_instruction']
            for j,tensor_t in enumerate(d['starts']):
                t=int(tensor_t)
                images=[Image.fromarray(np.ascontiguousarray(obs[k][t][::-1,::-1])) for k in ('agentview_rgb','eye_in_hand_rgb')]
                state=np.concatenate((obs['ee_states'][t],obs['gripper_states'][t])).astype(np.float32)
                assert np.allclose(state,d['state'][j].numpy(),atol=1e-6)
                torch.manual_seed(7+global_start+j)
                with torch.inference_mode():a=m.predict_action(image=images,instruction=instruction,unnorm_key=key,proprio=state)
                assert a.shape==(20,7)
                a=a.astype(np.float32);a[:,-1]=-np.sign(2*a[:,-1]-1)
                ag=torch.from_numpy(a).cuda()[None];sg=torch.from_numpy(state).cuda()[None];z=d['current_latent'][j:j+1].cuda()
                am,asd=norm['action'];sm,ssd=norm['state']
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):p=pred(z,(ag-am)/asd,(sg-sm)/ssd)
                w=captured.pop('W');assert p.shape==w.shape==(1,576,1024)
                assert all(bool(torch.isfinite(v).all()) for v in (p,w,ag))
                ps.append(p[0].float().cpu());ws.append(w[0].float().cpu());actions.append(torch.from_numpy(a))
        output=dict(P=torch.stack(ps),W=torch.stack(ws),wam_actions=torch.stack(actions),
            starts=d['starts'].clone(),source_shard=row['file'],source=d['source'],demo=d['demo'],suite=d['suite'],split=d['split'],
            instruction=instruction,window_seed_base=7+global_start)
        tmp=path.with_suffix('.tmp');torch.save(output,tmp);tmp.replace(path)
        done+=row['windows'];new_windows+=row['windows'];global_start+=row['windows'];processed+=1;bytes_total+=path.stat().st_size
        manifest.append(dict(**row,cache_file=str(path.relative_to(OUT))))
        elapsed=time.perf_counter()-runstart;mean=elapsed/max(new_windows,1)
        progress=dict(status='running',completed_windows=done,total_windows=total,completed_trajectories=ri+1,total_trajectories=len(rows),
            new_windows=new_windows,elapsed_s=elapsed,s_per_window=mean,estimated_remaining_s=(total-done)*mean,
            bytes=bytes_total,gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,last_shard=str(path),last_trajectory_s=time.perf_counter()-tick)
        savejson('progress.json',progress)
        if processed<=3 or processed%10==0:emit('PROGRESS',**progress)
        if processed%100==0:savejson('index_partial.json',manifest)
        del d,output,ps,ws,actions
    handle.remove();savejson('index.json',manifest)
    savejson('summary.json',dict(status='complete',windows=done,trajectories=len(rows),bytes=bytes_total,elapsed_s=time.perf_counter()-runstart,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
    emit('COMPLETE',windows=done)
if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);savejson('failure.json',dict(status='failed',time=time.time(),traceback=traceback.format_exc()));raise
