"""Deterministic local energy-ranking pilot for the LIBERO-10 predictor."""
import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/data/pxchen/PACT-WAM')
BASE_CACHE = ROOT/'artifacts/libero10_predictor_action_oneepoch_20260930'
OUT = ROOT/'artifacts/libero10_predictor_energy_ranking_pilot100_20260930'
SOURCE = ROOT/'checkpoints/vjepa2-ac-libero90-euler-fixed-rot180-step2000-inference.pt'
DATA = ROOT/'artifacts/libero10_guidance_backward_smoke_20260929/data/libero_10'
sys.path.insert(0, str(ROOT))


def emit(event, **kw):
    print(json.dumps(dict(time=time.strftime('%F %T'), event=event, **kw)), flush=True)


def selected_ids():
    from scripts.train_guidance_libero10_one_epoch import tasks
    rows = tasks()
    train = [i for i, r in enumerate(rows) if r['split'] == 'train']
    val = [i for i, r in enumerate(rows) if r['split'] == 'val']
    order = np.random.default_rng(239).permutation(train).tolist()[:100]
    assert len(order) == 100 and len(val) == 50
    return rows, order, val


def cache_real_latents():
    from experiments.libero.vjepa2_ac_ranker import VJEPA2ACRanker
    rows, order, val = selected_ids()
    destroot = OUT/'real_latent'; destroot.mkdir(parents=True, exist_ok=True)
    ranker = VJEPA2ACRanker(checkpoint_path=str(SOURCE), vjepa2_repo='/data/pxchen/vjepa2',
        device='cuda', dtype=torch.float32, max_ac_steps=2)
    del ranker.predictor
    ids = order+val
    for n, index in enumerate(ids, 1):
        dest = destroot/f'{index:04d}.pt'
        if dest.exists():
            continue
        row = rows[index]; t = row['t']
        cached = torch.load(BASE_CACHE/'wam'/f'{index:04d}.pt', map_location='cpu', weights_only=False)
        with h5py.File(DATA/row['task_file'], 'r') as f:
            rgb = f['data'][row['demo']]['obs']['agentview_rgb']
            v = [np.rot90(np.asarray(rgb[i]), 2).copy() for i in (t, t+4, t+8)]
        with torch.no_grad():
            reps = ranker.encode_clips([cached['context'], [v[0],v[1]], [v[1],v[2]]]).cpu()
        tmp = dest.with_suffix('.tmp'); torch.save(reps, tmp); os.replace(tmp, dest)
        if n == 1 or n % 25 == 0:
            emit('REAL_LATENT_CACHE_PROGRESS', completed=n, total=len(ids))
    emit('REAL_LATENT_CACHE_COMPLETED', total=len(ids))


def cache_wam_actions():
    from scripts.train_guidance_libero10_one_epoch import make_wam
    from scripts.smoke_guidance_three_step_joint import make_train_method
    _, order, val = selected_ids()
    destroot = OUT/'wam_action'; destroot.mkdir(parents=True, exist_ok=True)
    model, _, _ = make_wam(need_prompts=False)
    infer = make_train_method(model)
    ids = order+val
    for n, index in enumerate(ids, 1):
        dest = destroot/f'{index:04d}.pt'
        if dest.exists():
            continue
        r = torch.load(BASE_CACHE/'wam'/f'{index:04d}.pt', map_location='cpu', weights_only=False)
        kw = r['kw'].copy(); kw['precomputed_video_latents'] = kw.pop('video_latents')
        assert kw.pop('action_infer_mode', 'idm') == 'idm'; kw.pop('compile_action_infer', None)
        kw = {k:(v.cuda() if torch.is_tensor(v) else v) for k,v in kw.items()}
        with torch.no_grad():
            result = infer(**kw)
        action = result['action']
        if action.ndim == 3: action = action[0]
        tmp = dest.with_suffix('.tmp'); torch.save(action[:8].detach().float().cpu(), tmp); os.replace(tmp, dest)
        if n == 1 or n % 25 == 0:
            emit('WAM_ACTION_CACHE_PROGRESS', completed=n, total=len(ids))
        del result, action, kw; gc.collect(); torch.cuda.empty_cache()
    emit('WAM_ACTION_CACHE_COMPLETED', total=len(ids))


def configure_predictor(predictor):
    predictor.requires_grad_(False)
    for name in ('action_encoder','state_encoder','predictor_proj','predictor_norm'):
        getattr(predictor, name).requires_grad_(True)
    for block in predictor.predictor_blocks[-4:]: block.requires_grad_(True)
    predictor.extrinsics_encoder.requires_grad_(False)
    predictor.use_activation_checkpointing = True
    heads, blocks = [], []
    for name, p in predictor.named_parameters():
        if p.requires_grad:
            (blocks if name.startswith('predictor_blocks.') else heads).append(p)
    assert heads and blocks
    return heads, blocks


def train(smoke=False):
    from experiments.libero.vjepa2_ac_ranker import VJEPA2ACRanker, LiberoACAdapter
    rows, order, valids = selected_ids()
    ranker = VJEPA2ACRanker(checkpoint_path=str(SOURCE), vjepa2_repo='/data/pxchen/vjepa2',
        device='cpu', dtype=torch.float32, max_ac_steps=2)
    del ranker.encoder; ranker.device = torch.device('cuda')
    predictor = ranker.predictor.cuda().eval()
    heads, blocks = configure_predictor(predictor); params = heads+blocks
    optimizer = torch.optim.AdamW([dict(params=heads,lr=1e-5),dict(params=blocks,lr=3e-6)],weight_decay=0.04)
    adapter = LiberoACAdapter(low_level_steps_per_ac_step=4)
    runroot = OUT/('smoke' if smoke else 'train100'); runroot.mkdir(parents=True,exist_ok=False)
    metric_path = runroot/'metrics.jsonl'
    delta_rms = 0.01; margin = 0.02; positive_weight = 0.1

    def load(index):
        r = torch.load(BASE_CACHE/'wam'/f'{index:04d}.pt',map_location='cpu',weights_only=False)
        real = torch.load(OUT/'real_latent'/f'{index:04d}.pt',map_location='cpu',weights_only=True).cuda()
        online = torch.load(BASE_CACHE/'latent'/f'{index:04d}.pt',map_location='cpu',weights_only=True).cuda()
        wam = torch.load(OUT/'wam_action'/f'{index:04d}.pt',map_location='cpu',weights_only=True).cuda()
        return r,real,online,wam

    def one(index, update):
        r, real, online, wam = load(index)
        expert = r['expert'].cuda()[:8]
        direction = wam[:8,:6]-expert[:8,:6]
        norm = direction.float().square().mean().sqrt()
        if norm < 1e-6:
            direction = torch.zeros_like(direction); direction[:,index%6] = 1.0
            norm = direction.float().square().mean().sqrt()
        direction = direction/(norm+1e-8)
        plus = expert.clone(); minus = expert.clone()
        plus[:,:6] += delta_rms*direction
        minus[:,:6] -= delta_rms*direction
        state = torch.as_tensor(adapter.state_from_observation(r['obs']),device='cuda')
        scale, offset = r['scale'].cuda(),r['offset'].cuda()

        def energy(action, reps):
            raw = (action[None]-offset)/scale
            ac = adapter.convert_torch(raw,state)
            return ranker.differentiable_energy(current_rep=reps[0],target_reps=reps[1:],
                ac_actions=ac.actions,ac_states=ac.states).mean()

        optimizer.zero_grad(set_to_none=True); started=time.monotonic()
        e_good=energy(expert,real); e_plus=energy(plus,real); e_minus=energy(minus,real)
        rank_plus=F.softplus(e_good-e_plus+margin)
        rank_minus=F.softplus(e_good-e_minus+margin)
        loss=rank_plus+rank_minus+positive_weight*e_good
        if update: loss.backward()

        # Deployment-matched diagnostic: WAM action against WAM imagined-future latent.
        probe=wam.detach().clone().requires_grad_(True)
        e_online=energy(probe,online)
        g=torch.autograd.grad(e_online,probe,create_graph=False)[0]
        predicted=-4.0*g[:,:6]
        pred_rms=predicted.float().square().mean().sqrt()
        predicted=predicted*torch.clamp(torch.tensor(0.02,device='cuda')/(pred_rms+1e-8),max=1.0)
        desired=expert[:,:6]-wam[:,:6]
        cosine=F.cosine_similarity(predicted.flatten(),desired.flatten(),dim=0,eps=1e-8)
        clean_mse=F.mse_loss(wam[:,:6],expert[:,:6])
        corrected_mse=F.mse_loss(wam[:,:6]+predicted,expert[:,:6])
        row=dict(window=index,loss=float(loss.detach()),e_good=float(e_good.detach()),
            e_plus=float(e_plus.detach()),e_minus=float(e_minus.detach()),
            margin_plus=float((e_plus-e_good).detach()),margin_minus=float((e_minus-e_good).detach()),
            both_ranked=bool((e_good<e_plus).detach() and (e_good<e_minus).detach()),
            online_cosine=float(cosine.detach()),online_correction_rms=float(predicted.float().square().mean().sqrt()),
            clean_action_mse=float(clean_mse),corrected_action_mse=float(corrected_mse),
            seconds=time.monotonic()-started)
        if update:
            grads=[p.grad for p in params if p.grad is not None]
            assert grads and all(torch.isfinite(x).all() for x in grads)
            gradnorm=float(torch.nn.utils.clip_grad_norm_(params,1.0))
            row.update(predictor_grad_norm=gradnorm,global_clip_active=bool(gradnorm>1.0))
            optimizer.step()
        del r,real,online,wam,expert,plus,minus,e_good,e_plus,e_minus,loss,probe,e_online,g,predicted
        gc.collect(); torch.cuda.empty_cache()
        return row

    def evaluate(label):
        out=[]
        for j,index in enumerate(valids,1):
            row=one(index,False); row.update(split='val',eval_at=label,val_index=j);out.append(row)
            with metric_path.open('a') as f:f.write(json.dumps(row)+'\n')
            if j%10==0: emit('VAL_PROGRESS',eval_at=label,completed=j,
                ranking_accuracy=float(np.mean([x['both_ranked'] for x in out])),
                online_cosine=float(np.mean([x['online_cosine'] for x in out])))
        return out

    baseline=None if smoke else evaluate('step_0000')
    training=[]
    for step,index in enumerate(order[:2] if smoke else order,1):
        row=one(index,True);row.update(split='train',step=step);training.append(row)
        with metric_path.open('a') as f:f.write(json.dumps(row)+'\n')
        if step<=3 or step%10==0:emit('TRAIN_PROGRESS',step=step,**{k:v for k,v in row.items() if k not in ('split','step')})
    if smoke:
        summary=dict(passed=True,steps=2,source_checkpoint=str(SOURCE),metrics=training,
            peak_gib=torch.cuda.max_memory_allocated()/2**30)
    else:
        final=evaluate('step_0100')
        keys=('e_good','margin_plus','margin_minus','both_ranked','online_cosine','online_correction_rms',
              'clean_action_mse','corrected_action_mse')
        def aggregate(items):return {k:float(np.mean([x[k] for x in items])) for k in keys}
        b,f=aggregate(baseline),aggregate(final)
        summary=dict(passed=True,steps=100,source_checkpoint=str(SOURCE),
            objective='deterministic two-sided local energy ranking on real future latents',
            delta_rms=delta_rms,margin=margin,positive_weight=positive_weight,
            baseline=b,final=f,change={k:f[k]-b[k] for k in keys},
            train_first10_loss=float(np.mean([x['loss'] for x in training[:10]])),
            train_last10_loss=float(np.mean([x['loss'] for x in training[-10:]])),
            clip_rate=float(np.mean([x['global_clip_active'] for x in training])),
            peak_gib=torch.cuda.max_memory_allocated()/2**30)
        payload=dict(predictor=predictor.state_dict(),optimizer=optimizer.state_dict(),step=100,
            source_checkpoint=str(SOURCE),objective=summary['objective'])
        tmp=runroot/'step_0100.tmp';torch.save(payload,tmp);os.replace(tmp,runroot/'step_0100.pt')
    (runroot/'summary.json').write_text(json.dumps(summary,indent=2))
    emit('SMOKE_COMPLETED' if smoke else 'PILOT_COMPLETED',**summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cache-real','cache-wam','smoke','train'],required=True)
    args=p.parse_args()
    try:
        dict(**{'cache-real':cache_real_latents,'cache-wam':cache_wam_actions},
             smoke=lambda:train(True),train=lambda:train(False))[args.phase]()
    except Exception:
        traceback.print_exc();sys.exit(1)
