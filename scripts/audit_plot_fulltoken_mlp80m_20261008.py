"""CPU-only final checkpoint capacity audit and actual epoch curves."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from train_fulltoken_mlp80m_pilot5_20261008 import FullTokenMLP, write_json

OUT = Path('/data/pxchen/PACT-WAM/artifacts/libero10_fulltoken_mlp80m_pilot5_20261008')
torch.set_num_threads(4)
summary = json.loads((OUT/'summary.json').read_text())
audit = {}
for variant in ('full_tokens', 'zero_jepa'):
    ck = torch.load(OUT/variant/'epoch005.pt', map_location='cpu', mmap=True, weights_only=False)
    torch.manual_seed(240)
    fresh = FullTokenMLP(ck['model']['mean'], ck['model']['std'])
    details = {}
    for name, initial in fresh.named_parameters():
        final = ck['model'][name]
        delta = float((final-initial.detach()).square().mean().sqrt())
        details[name] = dict(elements=initial.numel(), delta_rms=delta, finite=bool(torch.isfinite(final).all()))
    assert sum(x['elements'] for x in details.values()) == 80_001_136
    assert all(v['finite'] for v in details.values())
    assert all(details[n]['delta_rms'] > 0 for n in ('net.0.weight','net.2.weight','net.4.weight','net.6.weight'))
    if variant == 'full_tokens':
        assert details['token_projection.weight']['delta_rms'] > 0
    else:
        assert details['token_projection.weight']['delta_rms'] == 0
    audit[variant] = dict(parameters=80_001_136, step=ck['step'], epoch=ck['epoch'], layers=details,
                          optimizer_parameter_entries=len(ck['optimizer']['state']))
    del fresh, ck
write_json(audit, OUT/'capacity_audit.json')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':'black',
                     'axes.labelcolor':'black','xtick.color':'black','ytick.color':'black'})
fig, axes = plt.subplots(1,2,figsize=(10.4,4.1),layout='constrained')
for ax, field, title in zip(axes, ('train','val'), ('Full training set','Held-out demonstrations')):
    for variant, label, style, marker in (
        ('full_tokens','MLP + full P/W','-','o'),('zero_jepa','MLP + zero JEPA','--','s')):
        result = summary['results'][variant]
        x = [0]+[r['epoch'] for r in result['epochs']]
        y = [result['initial'][field]['mse']]+[r[field]['mse'] for r in result['epochs']]
        ax.plot(x,[v*1000 for v in y],style,color='black',marker=marker,markersize=4,
                linewidth=1.7,label=label)
    baseline = summary['results']['full_tokens']['initial'][field]['baseline_mse']
    ax.axhline(baseline*1000,color='black',linestyle=':',linewidth=1.1,label='Raw WAM')
    ax.set_title(title);ax.set_xlabel('Epoch');ax.set_ylabel('Normalized action MSE (x1000)')
    ax.set_xticks(range(6));ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,loc='best',fontsize=10)
fig.savefig(OUT/'loss_curves.png',dpi=200,facecolor='white')
fig.savefig(OUT/'loss_curves.svg',facecolor='white')
print(json.dumps({'capacity_audit_passed':True,'plots':['loss_curves.png','loss_curves.svg']}))
