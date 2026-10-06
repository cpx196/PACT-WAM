import argparse, collections, contextlib, hashlib, io, json, random, re
from pathlib import Path
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from libero.libero import benchmark

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
if (args.output/'tasks.txt').exists():raise FileExistsError('Immutable task list already exists')
source=Path('/data/pxchen/LIBERO-plus/libero/libero/benchmark/task_classification.json')
data=json.loads(source.read_text())['libero_10']
categories=['Objects Layout','Camera Viewpoints','Robot Initial States','Language Instructions','Light Conditions','Background Textures','Sensor Noise']
bases=sorted(b for b,n in collections.Counter(re.split(r'_(?:table|tb|light|view|moved|add|language|instruction)_',t['name'],maxsplit=1)[0] for t in data).most_common(10))
cells=collections.defaultdict(list)
for t in data:
    matches=[b for b in bases if t['name']==b or t['name'].startswith(b+'_')];assert len(matches)==1,t
    cells[(bases.index(matches[0]),categories.index(t['category']))].append(t)
# Integer transportation: 80 per base; 114/115 per OOD; minimize deviation from 800/70 per cell.
A=[];lo=[];hi=[]
for i in range(10):
    row=np.zeros(140);row[i*7:(i+1)*7]=1;A.append(row);lo.append(80);hi.append(80)
for j in range(7):
    row=np.zeros(140);row[j:70:7]=1;A.append(row);lo.append(115 if j<2 else 114);hi.append(lo[-1])
for k in range(70):
    row=np.zeros(140);row[k]=1;row[70+k]=-1;A.append(row);lo.append(-np.inf);hi.append(800/70)
    row=np.zeros(140);row[k]=-1;row[70+k]=-1;A.append(row);lo.append(-np.inf);hi.append(-800/70)
capacity=[len(cells[(i,j)]) for i in range(10) for j in range(7)]
res=milp(c=np.r_[np.zeros(70),np.ones(70)],integrality=np.r_[np.ones(70),np.zeros(70)],bounds=Bounds(np.zeros(140),np.r_[capacity,np.full(70,np.inf)]),constraints=LinearConstraint(np.array(A),lo,hi))
assert res.success,res.message
quotas=np.rint(res.x[:70]).astype(int).reshape(10,7)
selected=[]
for i in range(10):
    for j in range(7):
        candidates=cells[(i,j)];q=int(quotas[i,j]);by_level=collections.defaultdict(list)
        for t in candidates:by_level[t['difficulty_level']].append(t)
        exact={level:q*len(items)/len(candidates) for level,items in by_level.items()}
        allocation={level:int(v) for level,v in exact.items()}
        for level in sorted(exact,key=lambda l:(-(exact[l]-allocation[l]),l))[:q-sum(allocation.values())]:allocation[level]+=1
        rng=random.Random(239+i*100+j)
        for level in sorted(by_level):
            for t in rng.sample(sorted(by_level[level],key=lambda t:t['id']),allocation[level]):selected.append((t,bases[i]))
with contextlib.redirect_stdout(io.StringIO()):bench=benchmark.get_benchmark_dict()['libero_10'](task_order_index=0)
mapping={bench.get_task(i).name:i for i in range(bench.n_tasks)}
rows=[]
for t,base in selected:
    tid=mapping[t['name']];assert tid==t['id']-1
    task=bench.get_task(tid);bddl=Path(bench.get_task_bddl_file_path(tid))
    resolved=Path(str(bddl).split('_view_')[0]+'.bddl') if '_view_' in str(bddl) else bddl
    assert resolved.is_file(),resolved
    states=bench.get_task_init_states(tid);assert len(states)>0
    rows.append(dict(suite='libero_10',task_id=tid,classification_id=t['id'],base_task=base,category=t['category'],official_difficulty_level=t['difficulty_level'],name=task.name,instruction=task.language,bddl_path=str(bddl),resolved_bddl_path=str(resolved),init_states_filename=task.init_states_file,initial_state_index=0,available_initial_states=len(states)))
# Round-robin stratified order, so interrupted runs have broad category coverage.
rng=random.Random(239);buckets=collections.defaultdict(list)
for row in rows:buckets[(row['base_task'],row['category'])].append(row)
for bucket in buckets.values():rng.shuffle(bucket)
ordered=[]
while any(buckets.values()):
    keys=[k for k,v in buckets.items() if v];rng.shuffle(keys)
    for key in keys:ordered.append(buckets[key].pop())
assert len(ordered)==800 and len({r['task_id'] for r in ordered})==800
text=''.join(f"libero_10,{r['task_id']}\n" for r in ordered)
summary=dict(source_tasks=len(data),selected_tasks=800,rollout_seed=7,sampling_seed=239,trials_per_task=1,initial_state_index=0,episodes_per_group=800,total_paired_episodes=1600,status='task list only; not launched',classification_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),task_list_sha256=hashlib.sha256(text.encode()).hexdigest(),category_counts=dict(collections.Counter(r['category'] for r in ordered)),base_task_counts=dict(collections.Counter(r['base_task'] for r in ordered)),difficulty_counts=dict(collections.Counter(r['official_difficulty_level'] for r in ordered)),cell_counts={bases[i]:{categories[j]:int(quotas[i,j]) for j in range(7)} for i in range(10)},sampling_policy='exactly 80 per base task; 114/115 per OOD; cell balance constrained by availability; proportional difficulty within cell; stratified round-robin ordering')
args.output.mkdir(parents=True,exist_ok=True)
(args.output/'tasks.txt').write_text(text)
(args.output/'task_manifest.json').write_text(json.dumps(dict(settings=summary,tasks=ordered),indent=2,ensure_ascii=False)+'\n')
(args.output/'sampling_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
md=['# Long-horizon paired evaluation: fixed 800-task list','','Single rollout seed 7; initial-state index 0; 800 distinct tasks per group. Not launched.','','## Sampling balance','', '| Base task | '+' | '.join(categories)+' | Total |','|---|'+'---:|'*8]
for i,b in enumerate(bases):md.append('| '+b+' | '+' | '.join(str(int(v)) for v in quotas[i])+' | 80 |')
md+=['',f"Task-list SHA256: {summary['task_list_sha256']}",'','## Complete list','','| Order | Task ID (0-based) | OOD | Difficulty | Name |','|---:|---:|---|---:|---|']
for i,row in enumerate(ordered,1):md.append(f"| {i} | {row['task_id']} | {row['category']} | {row['official_difficulty_level']} | {row['name']} |")
(args.output/'TASK_LIST.md').write_text('\n'.join(md)+'\n')
print(json.dumps(summary,indent=2))
