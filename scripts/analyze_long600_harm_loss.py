import ast,collections,json,pathlib,re,statistics
r=pathlib.Path('/data/pxchen/PACT-WAM/evaluate_results/long800_paired_seed7_20260927')
keys=[(t['suite'],t['task_id']) for t in json.loads((r/'task_manifest.json').read_text())['tasks'][:600]]
groups=[]
for g in ['control_noguidance','experiment_rot180_guidance4x']:
    d={}
    for p in (r/g).glob('*/*results.json'):
        x=json.loads(p.read_text());d[(x['task_suite'],x['task_id'])]=x
    assert all(k in d for k in keys)
    groups.append(d)
a,b=groups
records={}
for p in (r/'experiment_rot180_guidance4x/.worker_pool/logs').glob('worker*.log'):
    if p.name not in ['worker_2.log','worker_5.log']:continue
    pending=[]
    for line in p.read_text(errors='replace').splitlines():
        if 'action guidance diagnostics=' in line:pending.append(ast.literal_eval(line.split('action guidance diagnostics=',1)[1]))
        m=re.search(r'worker \d+ completed (libero_\w+),(\d+):',line)
        if m:records[(m[1],int(m[2]))]=pending;pending=[]
assert all(records.get(k) for k in keys),[k for k in keys if not records.get(k)]
def stat(ks):
    rows=[x for k in ks for batch in records[k] for x in batch]
    rises=[(k,i,x) for k in ks for i,batch in enumerate(records[k]) for x in batch if x['loss_after_clean_step']>x['loss']]
    rel=[100*(x['loss_after_clean_step']-x['loss'])/x['loss'] for x in rows]
    worst=max(rel)
    return dict(tasks=len(ks),updates=len(rows),rises=len(rises),rise_rate_pct=100*len(rises)/len(rows),tasks_with_rise=len({k for k,i,x in rises}),median_drop_pct=-statistics.median(rel),max_rise_pct=max(0,worst),max_absolute_rise=max(0,max(x['loss_after_clean_step']-x['loss'] for x in rows)),correction_rms_median=statistics.median(x['correction_rms'] for x in rows),rises_details=[dict(task_id=k[1],replan=i,flow_step=x['flow_step'],before=x['loss'],after=x['loss_after_clean_step'],rise_pct=100*(x['loss_after_clean_step']-x['loss'])/x['loss'],correction_rms=x['correction_rms']) for k,i,x in rises])
report={}
for name,pair in [('harm',(1,0)),('rescue',(0,1)),('both_success',(1,1)),('both_failure',(0,0))]:
    ks=[k for k in keys if (a[k]['successes'],b[k]['successes'])==pair];report[name]=stat(ks)
report['harm_per_task']={str(k[1]):dict(description=a[k]['task_description'],**stat([k])) for k in keys if a[k]['successes'] and not b[k]['successes']}
out=r/'analysis';out.mkdir(exist_ok=True);(out/'first600_harm_loss.json').write_text(json.dumps(report,indent=2))
for name,x in report.items():
    if name=='harm_per_task':
        for k,v in x.items():print('HARM_TASK',k,v['updates'],v['rises'],v['median_drop_pct'],v['max_rise_pct'])
    else:print(name,json.dumps(x))
