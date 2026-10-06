"""CPU-only read-only diagnostics of saved residual MLPs."""
import json,sys
from pathlib import Path
import torch
import numpy as np
ROOT=Path('/data/pxchen/PACT-WAM');sys.path.insert(0,str(ROOT))
from scripts.train_libero10_residual_mlp import ResidualMLP,OUT
torch.set_num_threads(4)
d=torch.load(OUT/'features.pt',map_location='cpu',weights_only=False)
tr=d['train_ids'];va=d['val_ids'];rows=d['rows'];records={}
@torch.no_grad()
def evaluate(m,ids,mode='normal'):
    a,s,z,y=[d[k][ids].clone() for k in ('action','state','difference','expert')]
    if mode=='mean_jepa':z[:]=m.mean[63:]
    elif mode=='within_task_shuffle':
        for name in sorted(set(rows[i]['task_file'] for i in ids)):
            ix=[j for j,i in enumerate(ids) if rows[i]['task_file']==name]
            z[ix]=z[ix].roll(1,0)
    pred,c=m(a,s,z);base=(a[:,:,:6]-y).square().mean((1,2));err=(pred-y).square().mean((1,2))
    residual=y-a[:,:,:6]
    cosine=torch.nn.functional.cosine_similarity(c.flatten(1),residual.flatten(1),dim=1)
    norm=c.square().mean((1,2)).sqrt()
    return dict(mse=float(err.mean()),better=int((err<base).sum()),cosine_mean=float(cosine.mean()),
        rms_mean=float(norm.mean()),rms_p95=float(torch.quantile(norm,.95)),per_window=err.tolist(),
        baseline_per_window=base.tolist(),per_dimension_mse=(pred-y).square().mean((0,1)).tolist(),
        baseline_per_dimension=(a[:,:,:6]-y).square().mean((0,1)).tolist())
for variant in ['jepa','no_jepa']:
    records[variant]={}
    for name in ['best','last']:
        ck=torch.load(OUT/'train'/variant/f'{name}.pt',map_location='cpu',weights_only=False)
        m=ResidualMLP(ck['input_mean'],ck['input_std'],variant=='jepa').eval();m.load_state_dict(ck['model'])
        out=dict(epoch=ck['epoch'],step=ck['step'],train=evaluate(m,tr),val=evaluate(m,va))
        if variant=='jepa':
            out['train_mean_jepa']=evaluate(m,tr,'mean_jepa');out['val_mean_jepa']=evaluate(m,va,'mean_jepa')
            out['train_shuffled_jepa']=evaluate(m,tr,'within_task_shuffle');out['val_shuffled_jepa']=evaluate(m,va,'within_task_shuffle')
        per=out['val']['per_window'];base=out['val']['baseline_per_window'];task={}
        for i,err,b in zip(va,per,base):task.setdefault(rows[i]['task_file'],[]).append((err,b))
        out['per_task_val']={k:dict(mse=float(np.mean([v[0] for v in vs])),baseline=float(np.mean([v[1] for v in vs]))) for k,vs in task.items()}
        out['worse_tasks']=sum(v['mse']>v['baseline'] for v in out['per_task_val'].values())
        records[variant][name]=out
        print(json.dumps(dict(variant=variant,checkpoint=name,epoch=out['epoch'],train=out['train']['mse'],val=out['val']['mse'],
            val_cos=out['val']['cosine_mean'],worse_tasks=out['worse_tasks'],
            mean_jepa_val=out.get('val_mean_jepa',{}).get('mse'),shuffle_jepa_val=out.get('val_shuffled_jepa',{}).get('mse'))),flush=True)
ck=torch.load(OUT/'train/jepa/last.pt',map_location='cpu',weights_only=False)
flat=torch.cat((d['action'].flatten(1),d['state'],d['difference']),-1);norm=(flat-ck['input_mean'])/ck['input_std']
stats={}
for name,lo,hi in [('action_state',0,63),('jepa',63,1471)]:
    a,b=norm[tr,lo:hi],norm[va,lo:hi]
    singular=torch.linalg.svdvals(a);var=singular.square();cum=var.cumsum(0)/var.sum()
    stats[name]=dict(train_rms=float(a.square().mean().sqrt()),val_rms=float(b.square().mean().sqrt()),
        val_fraction_abs_over5=float((b.abs()>5).float().mean()),
        pca_dims_90=int((cum<.9).sum()+1),pca_dims_95=int((cum<.95).sum()+1),
        median_train_nearest_distance=float(torch.cdist(a,a).fill_diagonal_(float('inf')).min(1).values.median()),
        median_val_nearest_distance=float(torch.cdist(b,a).min(1).values.median()))
print(json.dumps(dict(feature_distribution=stats)),flush=True)
report=dict(models=records,feature_distribution=stats)
path=OUT/'generalization_audit.json';path.write_text(json.dumps(report,indent=2));print(str(path))
