"""Frozen predictor diagnostic: expert proximity versus real-future error.

No training, no environment rollouts, no generated ranking labels for training.
"""
import fcntl
import json
import os
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from scripts.train_libero10_future_correction_reconstruct10 import CACHE, MANIFEST, emit

ROOT=Path('/data/pxchen/PACT-WAM')
OUT=ROOT/'artifacts/libero10_jepa_action_ordering_gripper_fixed_20261007'
SOURCE=ROOT/'checkpoints/vjepa2-ac-libero90-euler-fixed-rot180-step2000-inference.pt'


def main():
    from experiments.libero.vjepa2_ac_ranker import VJEPA2ACRanker,LiberoACAdapter
    torch.set_num_threads(4);torch.manual_seed(239)
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'metrics.jsonl').exists() and not (OUT/'summary.json').exists()
    (OUT/'pid.txt').write_text(str(os.getpid())+'\n')
    rows=json.loads(MANIFEST.read_text());groups=defaultdict(list)
    for i,r in enumerate(rows):
        if r['split']=='val':groups[(r['task_file'],r['demo'])].append(i)
    assert len(groups)==50
    ids=[]
    for key,indices in sorted(groups.items()):
        # Up to two separated windows from each held-out demonstration.
        pick=np.linspace(0,len(indices)-1,min(2,len(indices)),dtype=int)
        ids.extend(indices[j] for j in pick)
    config=dict(checkpoint=str(SOURCE),manifest=str(MANIFEST),cache=str(CACHE),windows=len(ids),
                held_out_demos=len(groups),selection='up to two evenly separated stride32 windows per held-out demo',
                dtype='float32',frozen=True,training=False,rollouts=False,
                target='cached encoded REAL demonstration future, two autoregressive AC steps',
                primary_metric='mean latent L1',secondary_metric='latent MSE',
                candidates='5 joint interpolation +3 position-only +3 rotation-only +4 local expert perturbations +native WAM',
                interpolation='expert + alpha*(WAM-expert), alpha=0,.25,.5,.75,1',
                gripper='all diagnostic candidates retain expert gripper; native WAM separately verifies cache',
                local_perturbation_rms=.02,coordinate_system='first6 normalized WAM actions; executable conversion matches cache',
                batch=1,seed=239,ids=ids,
                caveats=['expert proximity is a surrogate, not simulator success',
                         'adapter integrates OSC commands approximately, not measured executed poses',
                         'held-out demos share LIBERO10 tasks; not OOD test',
                         'windows within demo correlated; bootstrap grouped by demo'])
    (OUT/'config.json').write_text(json.dumps(config,indent=2));emit('CONFIG',**config)
    ranker=VJEPA2ACRanker(checkpoint_path=str(SOURCE),vjepa2_repo='/data/pxchen/vjepa2',
                          device='cpu',dtype=torch.float32,max_ac_steps=2)
    del ranker.encoder
    predictor=ranker.predictor.cuda().eval().requires_grad_(False)
    adapter=LiberoACAdapter(low_level_steps_per_ac_step=4)
    emit('PREDICTOR_READY',parameters=sum(p.numel() for p in predictor.parameters()))
    results=[];started=time.monotonic()
    for n,index in enumerate(ids,1):
        item=torch.load(CACHE/f'{index:05d}.pt',map_location='cpu',weights_only=False)
        assert item['index']==index and item['row']==rows[index]
        expert=item['expert_executable']*item['scale']+item['offset']
        assert torch.allclose(expert[:,:6],item['expert_normalized_first6'],atol=1e-6,rtol=1e-6)
        # WAM's decoded gripper is open-probability (0 close / 1 open),
        # while expert_executable is -1 open / +1 close. Encode it before
        # the common WAM->executable conversion; never flip executable twice.
        expert[:,6]=((1-item['expert_executable'][:,6])/2)*item['scale'][6]+item['offset'][6]
        decoded=(expert-item['offset'])/item['scale']
        assert torch.allclose(1-2*decoded[:,6],item['expert_executable'][:,6],atol=1e-6)
        wam=item['action'];delta=wam[:,:6]-expert[:,:6]
        candidates=[];descriptions=[]
        for mode,alphas in [('joint',[0,.25,.5,.75,1]),('position',[0,.5,1]),('rotation',[0,.5,1])]:
            for alpha in alphas:
                a=expert.clone()
                dims=slice(0,6) if mode=='joint' else slice(0,3) if mode=='position' else slice(3,6)
                a[:,dims]+=alpha*delta[:,dims]
                candidates.append(a);descriptions.append(dict(group=mode,alpha=alpha))
        rng=np.random.default_rng(239+index)
        for mode,dims in [('position',slice(0,3)),('rotation',slice(3,6))]:
            direction=torch.tensor(rng.normal(size=(8,3)),dtype=torch.float32)
            direction*=.02/direction.square().mean().sqrt()
            for sign in [-1,1]:
                a=expert.clone();a[:,dims]+=sign*direction
                candidates.append(a);descriptions.append(dict(group='perturb_'+mode,sign=sign,rms=.02))
        candidates.append(wam.clone());descriptions.append(dict(group='native_wam'))
        candidates=torch.stack(candidates)
        assert torch.isfinite(candidates).all()
        errors_l1=[];errors_mse=[];predictions=[]
        for start in range(0,len(candidates)):
            a=candidates[start:start+1].cuda();scale=item['scale'].cuda();offset=item['offset'].cuda()
            raw=(a-offset)/scale
            executable=torch.cat((raw[:,:,:6],1-2*raw[:,:,6:7]),-1)
            ac=adapter.convert_torch(executable,item['state'].cuda())
            if n==1 and start==0:
                ref=adapter.convert(executable.cpu().numpy(),item['state'].numpy())
                assert np.allclose(ac.actions.cpu().numpy(),ref.actions,atol=2e-5,rtol=2e-5)
                assert np.allclose(ac.states.cpu().numpy(),ref.states,atol=2e-5,rtol=2e-5)
            with torch.no_grad():
                history=item['current'].cuda()[None].expand(len(a),-1,-1).contiguous()
                future=[]
                for k in range(2):
                    p=predictor(history,ac.actions[:,:k+1],ac.states[:,:k+1])
                    p=F.layer_norm(p[:,-256:],(1408,));future.append(p)
                    history=torch.cat((history,p),1)
                prediction=torch.stack(future,1)
                diff=prediction-item['real_future'].cuda()[None]
                errors_l1.append(diff.abs().mean((2,3)).cpu())
                errors_mse.append(diff.square().mean((2,3)).cpu())
                predictions.append(prediction.cpu())
        prediction=torch.cat(predictions)
        # Cache was produced with batch1. Verify that path independently;
        # different GEMM batch shapes can change FP32 rounding slightly.
        with torch.no_grad():
            raw=(wam.cuda()[None]-item['offset'].cuda())/item['scale'].cuda()
            executable=torch.cat((raw[:,:,:6],1-2*raw[:,:,6:7]),-1)
            ac=adapter.convert_torch(executable,item['state'].cuda())
            assert torch.allclose(ac.actions.cpu(),item['ac_actions'],atol=1e-6,rtol=1e-6)
            assert torch.allclose(ac.states.cpu(),item['ac_states'],atol=1e-6,rtol=1e-6)
            history=item['current'].cuda()[None];single=[]
            for k in range(2):
                p=predictor(history,ac.actions[:,:k+1],ac.states[:,:k+1])
                p=F.layer_norm(p[:,-256:],(1408,));single.append(p[0])
                history=torch.cat((history,p),1)
            single=torch.stack(single).cpu()
        cache_diff=float((single-item['predicted_future']).abs().max())
        cache_mean=float((single-item['predicted_future']).abs().mean())
        batch_diff=(prediction[-1]-single).abs()
        emit('CACHE_REPRODUCTION',index=index,max_abs=cache_diff,mean_abs=cache_mean)
        assert cache_diff<1e-3 and cache_mean<2e-5,f'Cached prediction mismatch {cache_diff}, mean {cache_mean}'
        assert float(batch_diff.mean())<1e-5 and float(batch_diff.max())<1e-3
        e1=torch.cat(errors_l1);e2=torch.cat(errors_mse)
        distances=(candidates[:,:,:6]-expert[None,:,:6]).square().mean((1,2))
        record=dict(index=index,row=rows[index],cache_max_abs_diff=cache_diff,
                    cache_mean_abs_diff=cache_mean,
                    batch_shape_rounding_max=float(batch_diff.max()),
                    batch_shape_rounding_mean=float(batch_diff.mean()),candidates=[])
        for j,description in enumerate(descriptions):
            record['candidates'].append(dict(**description,action_mse=float(distances[j]),
                position_mse=float((candidates[j,:,:3]-expert[:,:3]).square().mean()),
                rotation_mse=float((candidates[j,:,3:6]-expert[:,3:6]).square().mean()),
                latent_l1=float(e1[j].mean()),latent_mse=float(e2[j].mean()),
                latent_l1_per_time=e1[j].tolist(),latent_mse_per_time=e2[j].tolist()))
        with (OUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        results.append(record)
        if n==1:
            emit('SMOKE_PASSED',cache_max_abs_diff=cache_diff,candidate_count=len(candidates),
                 numpy_torch_adapter_match=True,peak_gib=torch.cuda.max_memory_allocated()/2**30)
        if n==1 or n%10==0 or n==len(ids):
            emit('PROGRESS',completed=n,total=len(ids),seconds=time.monotonic()-started,
                 seconds_per_window=(time.monotonic()-started)/n)
    def summarize(subset):
        stats={}
        for mode in ['joint','position','rotation']:
            for metric in ['latent_l1','latent_mse']:
                endpoint=[];rho=[];monotonic=[];per_time=[];curves=[];dist=[];noise=[]
                bydemo=defaultdict(list)
                for r in subset:
                    c=[x for x in r['candidates'] if x['group']==mode]
                    e=np.array([x[metric] for x in c]);d=np.array([x['action_mse'] for x in c])
                    curves.append(e);dist.append(d)
                    advantage=e[-1]-e[0];endpoint.append(advantage)
                    noise.append(abs(advantage)<=2*r['cache_mean_abs_diff'])
                    bydemo[(r['row']['task_file'],r['row']['demo'])].append(advantage)
                    rho.append(float(spearmanr(d,e).statistic) if np.ptp(d)>1e-12 and np.ptp(e)>1e-12 else np.nan)
                    monotonic.append(float(np.mean(np.diff(e)>0)))
                    per_time.append(np.array(c[-1][metric+'_per_time'])-np.array(c[0][metric+'_per_time']))
                demo=np.array([np.mean(v) for v in bydemo.values()])
                boot=np.random.default_rng(239).choice(demo,size=(2000,len(demo)),replace=True).mean(1)
                stats[mode+'_'+metric]=dict(windows=len(subset),expert_beats_endpoint_rate=float(np.mean(np.array(endpoint)>0)),
                    endpoint_error_minus_expert_mean=float(np.mean(endpoint)),
                    endpoint_margin_near_cache_rounding_fraction=float(np.mean(noise)),
                    demo_mean_advantage_95ci=np.quantile(boot,[.025,.975]).tolist(),
                    median_within_window_spearman=float(np.nanmedian(rho)),
                    positive_spearman_rate=float(np.mean(np.array(rho)>0)),valid_spearman_count=int(np.isfinite(rho).sum()),
                    adjacent_order_accuracy=float(np.mean(monotonic)),
                    mean_latent_error_curve=np.mean(curves,0).tolist(),mean_action_mse_curve=np.mean(dist,0).tolist(),
                    expert_beats_endpoint_rate_per_time=(np.array(per_time)>0).mean(0).tolist())
        for mode in ['position','rotation']:
            differences=[]
            for r in subset:
                expert=next(c for c in r['candidates'] if c['group']=='joint' and c['alpha']==0)
                differences.extend(c['latent_l1']-expert['latent_l1'] for c in r['candidates'] if c['group']=='perturb_'+mode)
            stats['local_'+mode]=dict(perturbation_increases_error_rate=float(np.mean(np.array(differences)>0)),
                                      mean_error_increase=float(np.mean(differences)))
        return stats
    summary=dict(config=config,overall=summarize(results),
                 per_task={task:summarize([r for r in results if r['row']['task_file']==task])
                           for task in sorted({r['row']['task_file'] for r in results})},
                 seconds=time.monotonic()-started,peak_gib=torch.cuda.max_memory_allocated()/2**30,
                 max_cache_reproduction_error=max(r['cache_max_abs_diff'] for r in results))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));emit('ALL_COMPLETED',overall=summary['overall'],seconds=summary['seconds'])

if __name__=='__main__':main()
