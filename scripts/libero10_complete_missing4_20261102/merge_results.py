from pathlib import Path
import json
R=Path('/data/pxchen/PACT-WAM/evaluate_results');O=R/'libero10_missing4_robotpose_seed7_20261102';rows=[]
def load(p):
 d=json.loads(p.read_text());assert not d['partial'] and d['completed']==d['total'];return d
for task in sorted((R/'libero10_remaining7_robotpose_three_arm_seed7_20261031').glob('task[0-9][0-9]')):
 m=json.loads((task/'manifest.json').read_text());rows.append(dict(task=m['base_task'],groups={a:load(task/a/'summary.json') for a in ['control','unweighted','weighted']}))
white=R/'white_mug_robot42_seed7_sameT5_20261025'
rows.append(dict(task='white_mug_and_chocolate_pudding',groups={'control':load(white/'control/summary.json'),'weighted':load(white/'weighted/summary.json'),'unweighted':load(white/'unweighted_20261026/summary.json')}))
rows.append(dict(task='alphabet_soup_and_tomato_sauce',groups={'control':load(R/'soup_tomato_robot41_baseline_repair_20261029/control/summary.json'),'weighted':load(R/'soup_tomato_robot41_seed7_sameT5_20261028/weighted/summary.json'),'unweighted':load(O/'task01/unweighted/summary.json')}))
rows.append(dict(task='alphabet_soup_and_cream_cheese',groups={a:load(O/'task02'/a/'summary.json') for a in ['control','unweighted','weighted']}))
assert len(rows)==10
for row in rows:
 ds=list(row['groups'].values());ids=[{x['task_id'] for x in d['rollouts']} for d in ds];assert all(x==ids[0] for x in ids)
totals={a:dict(successes=sum(x['groups'][a]['successes'] for x in rows),episodes=sum(x['groups'][a]['completed'] for x in rows)) for a in ['control','unweighted','weighted']}
assert all(x['episodes']==393 for x in totals.values())
(O/'complete_libero10_table.json').write_text(json.dumps(dict(complete=True,seed=7,base_tasks=10,variants=393,total_episodes=1179,totals=totals,rows=rows),indent=2))
lines=['# LIBERO-10 initial robot pose: complete three-arm comparison','', 'seed7, one episode per version, paired T5 contexts.','', '|Task|Versions|Control|Unweighted|Weighted|','|---|---:|---:|---:|---:|']
for x in rows:
 g=x['groups'];lines.append('|'+x['task']+'|'+str(g['control']['completed'])+'|'+ '|'.join(str(g[a]['successes']) for a in ['control','unweighted','weighted'])+'|')
lines.append('|TOTAL|393|'+'|'.join(str(totals[a]['successes']) for a in ['control','unweighted','weighted'])+'|')
(O/'complete_libero10_table.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(totals))
