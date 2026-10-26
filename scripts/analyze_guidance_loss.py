import ast,json,pathlib,re,statistics,collections
r=pathlib.Path('/data/pxchen/PACT-WAM/evaluate_results/plus_short315_paired_rot180_20260926')
groups=[]
for group in ['control_noguidance','experiment_rot180_guidance4x']:
    d={}
    for p in (r/group).glob('*/*results.json'):
        x=json.loads(p.read_text());d[(x['task_suite'],x['task_id'])]=x
    groups.append(d)
control,experiment=groups
records={}
logs=list(r.glob('restart_archive_1234/experiment_worker_pool/logs/worker*.log'))+list(r.glob('restart_archive_numpy2_1348/experiment_worker_pool/logs/worker*.log'))+list((r/'experiment_rot180_guidance4x/.worker_pool/logs').glob('worker*.log'))
for p in logs:
    pending=[]
    for line in p.read_text(errors='replace').splitlines():
        if 'action guidance diagnostics=' in line:
            s=line.split('action guidance diagnostics=',1)[1]
            pending.append(ast.literal_eval(s))
        m=re.search(r'worker \d+ completed (libero_\w+),(\d+):',line)
        if m:
            key=(m[1],int(m[2]));records[key]=pending;pending=[]
def stats(keys,early=False):
    rows=[x for k in keys for batch in (records.get(k,[])[:10] if early else records.get(k,[])) for x in batch]
    if not rows:return {}
    rel=[100*(x['loss']-x['loss_after_clean_step'])/x['loss'] for x in rows]
    return dict(tasks=len(keys),covered=sum(bool(records.get(k)) for k in keys),updates=len(rows),loss_median=statistics.median(x['loss'] for x in rows),drop_pct_median=statistics.median(rel),rise=sum(v<0 for v in rel),rise_pct=100*sum(v<0 for v in rel)/len(rows),correction_rms_median=statistics.median(x['correction_rms'] for x in rows),correction_rms_max=max(x['correction_rms'] for x in rows),clipped=sum(x['unclipped_delta_rms']>0.0200001 for x in rows))
report={}
for suite in ['libero_spatial','libero_object','libero_goal']:
    keys=[k for k in experiment if k[0]==suite]
    for subset in ['all','harm','rescue','both_success','both_failure']:
        ks=[k for k in keys if subset=='all' or (control[k]['successes'],experiment[k]['successes'])=={'harm':(1,0),'rescue':(0,1),'both_success':(1,1),'both_failure':(0,0)}[subset]]
        report[f'{suite}/{subset}']={'all_replans':stats(ks),'first10_replans':stats(ks,True)}
for k in sorted(experiment):
    if k[0]=='libero_goal' and control[k]['successes'] and not experiment[k]['successes']:
        report[f'harm_task_{k[1]}']=stats([k])
print(json.dumps(report,indent=2));print('MISSING',sorted(set(experiment)-set(records)))
out=r/'harm_inspection/guidance_loss_analysis.json';out.write_text(json.dumps(report,indent=2))
