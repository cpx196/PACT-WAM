import sys,json,time,fcntl
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/pxchen/PACT-WAM/scripts')
import train_jepawam_arb_pilot100 as t

def main():
    torch.set_num_threads(2)
    roots={k:t.ROOT/f'jepawam_arb_{k}_15epoch_clip5_20261013' for k in ['pw','no_pw']}
    out=roots['pw']/'error_distribution.json'
    lock=open(roots['pw']/'error_distribution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if out.exists():raise RuntimeError('Result already exists')
    stats=json.load(open(t.BASE/'train_only_stats.json'))
    models={}
    for k,root in roots.items():
        ck=torch.load(root/'epoch15_step59055.pt',map_location='cpu',weights_only=False)
        assert ck['epoch']==15 and ck['config']['no_pw']==(k=='no_pw')
        m=t.ARB(stats).cuda().eval();m.load_state_dict(ck['model'],strict=True);models[k]=m;del ck
    ds=t.Windows('val');dl=t.loader(ds,list(range(len(ds))))
    names=['wam','pw','no_pw'];records={k:[] for k in names};started=time.time()
    with torch.inference_mode():
        for i,cpu in enumerate(dl):
            b=t.device(cpu);pred={'wam':b['action'][...,:6]}
            for k,m in models.items():
                t.NO_PW=(k=='no_pw')
                with torch.autocast('cuda',dtype=torch.bfloat16):pred[k]=m(b)[0]
            for k,p in pred.items():
                error=p-b['target'][...,:6];norm=error/models['pw'].action_std[:6]
                delta=p-b['action'][...,:6]
                records[k].append(torch.stack((norm.square().mean(1),error.square().mean(1),delta.square().mean(1)),dim=1).cpu().numpy())
            if i%100==0:print(json.dumps({'windows':min((i+1)*8,len(ds)),'seconds':time.time()-started}),flush=True)
    arrays={k:np.concatenate(v,axis=0) for k,v in records.items()}
    assert all(v.shape==(3500,3,6) for v in arrays.values())
    summary={'windows':3500,'epoch':15,'dimensions':['dx','dy','dz','rx','ry','rz'],
        'units':'raw values are executable control commands, not meters/radians; normalized values divide by train action std',
        'window_definition':'20 controls x6 dimensions; gripper not supervised','results':{},'seconds':time.time()-started}
    wmse=arrays['wam'][:,0,:].mean(1)
    for k,a in arrays.items():
        mse=a[:,0,:].mean(1);rmse=np.sqrt(mse)
        r={'normalized_mse':float(mse.mean()),'normalized_rmse':float(np.sqrt(mse.mean())),
           'normalized_rmse_per_dim':np.sqrt(a[:,0,:].mean(0)).tolist(),
           'raw_rmse_per_dim':np.sqrt(a[:,1,:].mean(0)).tolist(),
           'window_normalized_rmse_percentiles':{str(q):float(np.percentile(rmse,q)) for q in [10,50,90,95,99,100]},
           'window_normalized_mse_percentiles':{str(q):float(np.percentile(mse,q)) for q in [50,90,95,99]},
           'raw_correction_rms_per_dim':np.sqrt(a[:,2,:].mean(0)).tolist()}
        if k!='wam':
            diff=mse-wmse;rel=diff/np.maximum(wmse,1e-12)
            r.update(improved_count=int((diff< -1e-6).sum()),worsened_count=int((diff>1e-6).sum()),
                tied_count=int((np.abs(diff)<=1e-6).sum()),worsened_more_than_10pct_count=int((rel>.1).sum()),
                worsened_more_than_50pct_count=int((rel>.5).sum()),
                mean_mse_improved_subset=float(mse[diff< -1e-6].mean()),
                mean_mse_worsened_subset=float(mse[diff>1e-6].mean()))
            expected=json.load(open(roots[k]/'summary.json'))['validation'][-1]['normalized_mse']
            assert abs(r['normalized_mse']-expected)<1e-6
        summary['results'][k]=r
    diff=arrays['pw'][:,0,:].mean(1)-arrays['no_pw'][:,0,:].mean(1)
    summary['pw_vs_no_pw']={'pw_better_count':int((diff< -1e-6).sum()),'pw_worse_count':int((diff>1e-6).sum())}
    temp=out.with_suffix('.tmp');temp.write_text(json.dumps(summary,indent=2));temp.replace(out)
    np.savez_compressed(roots['pw']/'error_distribution_windows.npz',**arrays)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
