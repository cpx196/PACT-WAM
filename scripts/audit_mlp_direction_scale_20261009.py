"""Read-only CPU inference: separate residual direction from correction scale."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/data/pxchen/PACT-WAM')
sys.path.insert(0, str(ROOT))
from scripts.train_fulltoken_mlp80m_pilot5_20261008 import FullTokenMLP, CACHE, PCACHE, MANIFEST

OUT = ROOT/'artifacts/mlp80m_direction_scale_audit_20261009'
SOURCE = ROOT/'artifacts/libero10_fulltoken_mlp80m_pilot5_20261008'


def evaluate_arrays(a, y, p):
    r, c = y-a, p-a
    dot = (r*c).mean((1, 2))
    c2 = (c*c).mean((1, 2))
    r2 = (r*r).mean((1, 2))
    cosine = dot/np.maximum(np.sqrt(c2*r2), 1e-15)
    before, after = r2, ((p-y)**2).mean((1, 2))
    oracle_alpha = max(0., float(dot.mean()/max(c2.mean(), 1e-15)))
    window_alpha = np.maximum(0., dot/np.maximum(c2, 1e-15))
    return dict(windows=len(a), raw_mse=float(before.mean()), corrected_mse=float(after.mean()),
                positive_direction_fraction=float((dot>0).mean()), mean_cosine=float(cosine.mean()),
                improved_fraction=float((after<before).mean()),
                positive_direction_but_worse_fraction=float(((dot>0)&(after>before)).mean()),
                nonpositive_direction_fraction=float((dot<=0).mean()),
                global_oracle_scale=oracle_alpha,
                global_oracle_scaled_mse=float(((a+oracle_alpha*c-y)**2).mean()),
                per_window_oracle_scaled_mse=float(((a+window_alpha[:,None,None]*c-y)**2).mean()),
                scale_sweep={str(s):float(((a+s*c-y)**2).mean()) for s in (0.,.1,.25,.5,.75,1.)},
                per_dimension_mse=((p-y)**2).mean((0,1)).tolist(),
                note='Oracle scales use evaluated labels and are diagnostic bounds, not deployment estimates.')


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'summary.json').exists()
    rows = json.loads(MANIFEST.read_text())
    train = [i for i,r in enumerate(rows) if r['split']=='train']
    val = [i for i,r in enumerate(rows) if r['split']=='val']
    calibration = np.array(train)[np.linspace(0,len(train)-1,256,dtype=int)].tolist()
    ids = calibration+val
    models = {}
    for arm in ('full_tokens','zero_jepa'):
        ck = torch.load(SOURCE/arm/'epoch005.pt',map_location='cpu',weights_only=False,mmap=True)
        model = FullTokenMLP(ck['model']['mean'],ck['model']['std'])
        model.load_state_dict(ck['model']); model.eval().requires_grad_(False)
        models[arm]=model
    saved = {'a':[], 'y':[], 'full_tokens':[], 'zero_jepa':[]}
    start=time.monotonic()
    for start_index in range(0,len(ids),16):
        batch=ids[start_index:start_index+16]
        aa,ss,yy,zz=[],[],[],[]
        for i in batch:
            old=torch.load(CACHE/f'{i:05d}.pt',map_location='cpu',weights_only=False,mmap=True)
            new=torch.load(PCACHE/'tokens'/f'{i:05d}.pt',map_location='cpu',weights_only=False,mmap=True)
            assert old['row']==new['row']==rows[i]
            aa.append(old['action'].float()); ss.append(old['state'].float())
            yy.append(old['expert_normalized_first6'].float())
            zz.append(torch.stack((new['predicted_future'],old['wam_future']),1).float())
        a,s,y,z=torch.stack(aa),torch.stack(ss),torch.stack(yy),torch.stack(zz)
        with torch.no_grad():
            for arm,model in models.items():
                saved[arm].append(model(a,s,z if arm=='full_tokens' else None).numpy())
        saved['a'].append(a[:,:,:6].numpy()); saved['y'].append(y.numpy())
        if start_index%128==0:
            print(json.dumps(dict(done=start_index+len(batch),total=len(ids),seconds=time.monotonic()-start)),flush=True)
    saved={k:np.concatenate(v) for k,v in saved.items()}
    n=len(calibration)
    result={}
    expected=json.loads((SOURCE/'summary.json').read_text())['results']
    for arm in models:
        train_metrics=evaluate_arrays(saved['a'][:n],saved['y'][:n],saved[arm][:n])
        val_metrics=evaluate_arrays(saved['a'][n:],saved['y'][n:],saved[arm][n:])
        assert abs(val_metrics['corrected_mse']-expected[arm]['final']['val']['mse'])<2e-6
        scale=train_metrics['global_oracle_scale']
        a,y,p=saved['a'][n:],saved['y'][n:],saved[arm][n:]
        val_metrics['train_fitted_scale']=scale
        val_metrics['train_fitted_scale_val_mse']=float(((a+scale*(p-a)-y)**2).mean())
        result[arm]=dict(train_calibration=train_metrics,val=val_metrics)
    result['metadata']=dict(source=str(SOURCE),checkpoint_epoch=5,train_calibration_windows=n,
                            val_windows=len(val),updated_weights=False,device='cpu',seconds=time.monotonic()-start)
    np.savez_compressed(OUT/'predictions.npz',**saved,ids=np.array(ids),calibration_count=n)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
