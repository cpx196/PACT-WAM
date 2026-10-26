import json,sys,time
from pathlib import Path
import torch,numpy as np
ROOT=Path('/data/pxchen/PACT-WAM')
OUT=ROOT/'artifacts/ar2_long642_alignment_probe_20261015'
BASE=ROOT/'artifacts/predictor21_joint20_stride20_20261011'
CACHE=ROOT/'artifacts/fastwam_predictor21_ar2_stride20_20261014'
torch.set_num_threads(4)
rows=[r for r in json.loads((CACHE/'index.json').read_text()) if r['suite']=='libero10' and r['split']=='val']
values=[]
for r in rows:
    b=torch.load(BASE/r['file'],map_location='cpu',weights_only=False,mmap=True)
    y=torch.load(CACHE/r['target_file'],map_location='cpu',weights_only=False,mmap=True)
    assert torch.equal(b['starts'],y['starts'])
    values.append((b['current_latent'][:,None]-y['targets']).abs().mean((-2,-1)).numpy())
copy=np.concatenate(values)
metrics=dict(np.load(OUT/'window_metrics.npz'));metrics['copy_current_l1']=copy
np.savez_compressed(OUT/'window_metrics.npz',**metrics)
report=json.loads((OUT/'summary.json').read_text())
report['overall']['copy_current_l1']=float(copy.mean())
report['overall']['improvement_vs_copy_current_pct']=float(100*(copy.mean()-metrics['matched'].mean())/copy.mean())
for k in range(2):report['by_interval'][k]['copy_current_l1']=float(copy[:,k].mean())
q=OUT/'summary.tmp';q.write_text(json.dumps(report,indent=2));q.replace(OUT/'summary.json')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(10,4.3))
names=['Predictor','Copy current','Train task mean','Wrong time\n(same demo)']
vals=[metrics['matched'].mean(),copy.mean(),metrics['baseline'].mean(),metrics['negative_same_demo_far_time'].mean()]
bars=axs[0].bar(range(4),vals,color=['0.2','0.55','0.7','0.85'],width=.58)
axs[0].set_xticks(range(4),names);axs[0].set_ylabel('Latent L1 (lower is better)');axs[0].set_ylim(0,max(vals)*1.23)
for b,v in zip(bars,vals):axs[0].text(b.get_x()+b.get_width()/2,v+.012,f'{v:.3f}',ha='center')
axs[0].set_title('A. Does prediction beat a static copy?')
xx=metrics['matched'].mean(1);yy=copy.mean(1)
axs[1].scatter(xx,yy,s=11,color='0.3',alpha=.45)
lo=0;hi=max(xx.max(),yy.max())*1.05
axs[1].plot([lo,hi],[lo,hi],'--',color='0.65',linewidth=1)
axs[1].set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Predictor L1',ylabel='Copy-current L1',title='B. Above diagonal = Predictor improves')
fig.suptitle('Frozen epoch-10 Predictor | 642 Long validation windows',fontsize=14)
fig.tight_layout();fig.savefig(OUT/'alignment_baselines.png',dpi=180);plt.close(fig)
print(json.dumps(report['overall']),flush=True)
