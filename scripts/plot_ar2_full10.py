"""Match the earlier Predictor's black/gray two-panel scientific style."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
parser=argparse.ArgumentParser()
parser.add_argument('--root',default='/Users/ocbrain/Documents/ChatGPT/PACT/artifacts/fastwam_ar2_full10_curves_20261014')
ROOT=Path(parser.parse_args().root)
train={};val={}
lines=(ROOT/'metrics.jsonl').read_text().splitlines()
for i,line in enumerate(lines):
    if not line.strip():continue
    try:r=json.loads(line)
    except json.JSONDecodeError:
        if i==len(lines)-1:continue
        raise
    (train if r['kind']=='train' else val)[r['step']]=r
assert train and val
steps=np.array(sorted(train));loss=np.array([train[s]['loss'] for s in steps]);v=sorted(val)
assert list(steps)==list(range(1,int(steps[-1])+1))
scale=1969;latest=val[v[-1]]
fonts={f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({'font.family':'Calibri' if 'Calibri' in fonts else 'DejaVu Sans','font.size':12,
    'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(11.5,4.4),layout='constrained')
ax=axes[0]
ax.plot(steps/scale,loss,color='black',alpha=.18,lw=.7,label='Train (per batch)')
window=min(100,len(steps))
ax.plot(steps[window-1:]/scale,np.convolve(loss,np.ones(window)/window,'valid'),color='black',lw=1.6,label=f'Train ({window}-step mean)')
ax.plot(np.array(v)/scale,[val[s]['normal'] for s in v],color='black',ls='--',marker='o',ms=3,lw=1.5,label='Validation')
ax.set_title('Training and validation loss')
ax=axes[1]
for key,label,ls,marker in [('normal','Correct actions','-','o'),('shuffled_action','Shuffled actions','--','s'),('mean_action','Mean actions',':','^')]:
    ax.plot(np.array(v)/scale,[val[s][key] for s in v],color='black',ls=ls,marker=marker,ms=3,lw=1.5,label=label)
ax.set_title('Validation: action-input comparison')
for ax in axes:
    ax.axvline(296/scale,color='black',alpha=.35,lw=.8,ls=':')
    ax.set_xlabel('Epoch');ax.set_ylabel('Joint latent L1');ax.set_xlim(0,float(steps[-1])/scale*1.02)
    ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=10)
fig.suptitle('V-JEPA2.1 Predictor: two-step autoregressive training',fontsize=15)
for ext in ['png','svg','pdf']:fig.savefig(ROOT/f'fastwam_ar2_loss.{ext}',dpi=220)
summary=dict(latest_train_step=int(steps[-1]),epoch=float(steps[-1])/scale,latest_validation=latest,
    last100_train=float(loss[-100:].mean()),shuffle_increase_pct=100*(latest['shuffled_action']/latest['normal']-1))
(ROOT/'plot_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
