"""Read-only checkpoint/input ablations. No optimization or online evaluation."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from scripts.train_libero10_idm_residual_pilot import OUT, Corrector


def main():
    torch.set_num_threads(4)
    data = torch.load(OUT/'dataset.pt', map_location='cpu', weights_only=False)
    train, val = data['train_ids'], data['val_ids']
    a, z, y = (data[k].cuda() for k in ('action', 'tokens', 'expert'))
    mean = data['action'][train].flatten(1).mean(0)
    std = data['action'][train].flatten(1).std(0).clamp_min(1e-4)
    report = {}
    tasknames = [r['task_file'] for r in data['rows']]
    def permutation(ids, within_task=False):
        out = ids.copy()
        groups = sorted(set(tasknames[i] for i in ids)) if within_task else ['all']
        for group in groups:
            positions = [j for j, i in enumerate(ids) if group == 'all' or tasknames[i] == group]
            for j, position in enumerate(positions):
                out[position] = ids[positions[(j+1) % len(positions)]]
        return out
    def encode(model, ids, mode='normal'):
        raws, las = [], []
        with torch.no_grad():
            for start in range(0, len(ids), 8):
                zz = z[ids[start:start+8]]
                if mode == 'current_repeated':
                    zz = zz[:, :1].expand(-1, 3, -1, -1)
                elif mode == 'time_reversed':
                    zz = zz.flip(1)
                elif mode == 'future_swapped':
                    zz = zz[:, [0, 2, 1]]
                pairs = torch.stack((zz[:, :2], zz[:, 1:3]), 1)
                raw = model.idm(pairs.flatten(0, 1)).reshape(zz.shape[0], 2, 128)
                raws.append(raw)
                las.append(model.la_norm(raw))
        return torch.cat(raws), torch.cat(las)
    def stats(values, ids):
        xx = values[ids].flatten(1)
        centered = xx-xx.mean(0)
        norm = xx.norm(dim=1)
        cos = F.cosine_similarity(xx, xx.roll(1, 0), dim=1)
        return dict(mean_norm=float(norm.mean()), std_norm=float(norm.std()),
                    mean_channel_std=float(xx.std(0).mean()),
                    variation_energy_fraction=float(centered.square().mean()/xx.square().mean()),
                    adjacent_sample_cosine_mean=float(cos.mean()),
                    la1_la2_cosine=float(F.cosine_similarity(values[ids, 0], values[ids, 1], dim=1).mean()))
    def predict_metrics(model, ids, la):
        with torch.no_grad():
            aa, yy = a[ids], y[ids]
            flat = (aa.flatten(1)-model.action_mean)/model.action_std
            correction = model.mlp(torch.cat((flat, la.flatten(1)), -1)).reshape(-1, 8, 6)
            residual = yy-aa[:, :, :6]
            error = (correction-residual).square().mean((1, 2))
            base = residual.square().mean((1, 2))
            return dict(mse=float(error.mean()), baseline=float(base.mean()),
                        correction_rms=float(correction.square().mean((1, 2)).sqrt().mean()),
                        cosine=float(F.cosine_similarity(correction.flatten(1), residual.flatten(1), dim=1).mean()),
                        better=int((error<base).sum()), worse=int((error>base).sum()))
    torch.manual_seed(239)
    initial = Corrector(mean, std, True).cuda().eval()
    init_raw, init_la = encode(initial, list(range(500)))
    initial_weights = {k: v.detach().cpu().clone() for k, v in initial.idm.state_dict().items()}
    report['initial_la'] = dict(train=stats(init_la, train), val=stats(init_la, val))
    del initial
    for checkpoint in ('best', 'last'):
        ck = torch.load(OUT/'train/idm'/f'{checkpoint}.pt', map_location='cpu', weights_only=False)
        model = Corrector(mean, std, True).cuda().eval()
        model.load_state_dict(ck['model'])
        raw, la = encode(model, list(range(500)))
        changes = {}
        for k in ('project.weight', 'transformer.layers.0.self_attn.in_proj_weight',
                  'transformer.layers.5.self_attn.in_proj_weight', 'output.weight'):
            old = initial_weights[k]
            new = model.idm.state_dict()[k].cpu()
            changes[k] = float((new-old).norm()/old.norm())
        entry = dict(epoch=ck['epoch'], steps=ck['step'], weight_relative_changes=changes,
                     la_stats=dict(train=stats(la, train), val=stats(la, val)), ablations={})
        avg = la[train].mean(0, keepdim=True)
        for split, ids in (('train', train), ('val', val)):
            conditions = dict(normal=la[ids], zero=torch.zeros_like(la[ids]),
                              train_mean=avg.expand(len(ids), -1, -1),
                              global_shuffle=la[permutation(ids)],
                              within_task_shuffle=la[permutation(ids, True)],
                              la_order_swapped=la[ids].flip(1))
            entry['ablations'][split] = {name: predict_metrics(model, ids, value)
                                         for name, value in conditions.items()}
        entry['temporal_ablations'] = {}
        for mode in ('current_repeated', 'time_reversed', 'future_swapped'):
            _, alternative = encode(model, val, mode)
            entry['temporal_ablations'][mode] = dict(metrics=predict_metrics(model, val, alternative),
                la_change_rms=float((alternative-la[val]).square().mean().sqrt()),
                la_change_relative=float((alternative-la[val]).norm()/la[val].norm()))
        # Inspect gradients without updating any checkpoint parameters.
        model.zero_grad(set_to_none=True)
        aa = a[train[:8]].detach().clone().requires_grad_(True)
        zz = z[train[:8]].detach().clone().requires_grad_(True)
        prediction, delta = model(aa, zz)
        loss = F.mse_loss(prediction, y[train[:8]])
        loss.backward()
        entry['gradient_probe'] = dict(loss=float(loss), token_gradient_norm=float(zz.grad.norm()),
            idm_parameter_gradient_norm=float(torch.stack([p.grad.square().sum() for p in model.idm.parameters() if p.grad is not None]).sum().sqrt()),
            mlp_parameter_gradient_norm=float(torch.stack([p.grad.square().sum() for p in model.mlp.parameters() if p.grad is not None]).sum().sqrt()))
        report[checkpoint] = entry
        del model, ck
        torch.cuda.empty_cache()
        print(json.dumps(dict(checkpoint=checkpoint, epoch=entry['epoch'], la_stats=entry['la_stats'],
                              val_ablations=entry['ablations']['val'], temporal=entry['temporal_ablations'])), flush=True)
    path = OUT/'diagnosis.json'
    path.write_text(json.dumps(report, indent=2))
    print('DIAGNOSIS_COMPLETED '+str(path), flush=True)


if __name__ == '__main__':
    main()
