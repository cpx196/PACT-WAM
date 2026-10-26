"""Freeze 200 task IDs BEFORE inference, without consulting outcomes."""
import json,random,collections,hashlib,contextlib,io
from pathlib import Path
from wand.api import library as _wand_library
from libero.libero import benchmark
ROOT=Path('/data/pxchen/PACT-WAM')
OUT=ROOT/'evaluate_results/fastwam_noguidance_long200_fullvideo_20261015'
SOURCE=Path('/data/pxchen/LIBERO-plus/libero/libero/benchmark/task_classification.json')
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'task_manifest.json').exists():
        print((OUT/'sampling_summary.json').read_text());return
    data=json.loads(SOURCE.read_text())['libero_10'];assert len(data)==2519
    bases=sorted(json.loads((ROOT/'evaluate_results/long800_paired_seed7_20260927/task_manifest.json').read_text())['settings']['base_task_counts'])
    cats=['Objects Layout','Camera Viewpoints','Robot Initial States','Language Instructions','Light Conditions','Background Textures','Sensor Noise']
    cells=collections.defaultdict(list)
    for t in data:
        matched=[b for b in bases if t['name']==b or t['name'].startswith(b+'_')];assert len(matched)==1
        cells[bases.index(matched[0]),cats.index(t['category'])].append(t)
    with contextlib.redirect_stdout(io.StringIO()):bench=benchmark.get_benchmark_dict()['libero_10'](task_order_index=0)
    mapping={bench.get_task(i).name:i for i in range(bench.n_tasks)}
    rng=random.Random(20261015);rows=[]
    for i,b in enumerate(bases):
        for j,cat in enumerate(cats):
            # 6 categories x3 + 1 category x2 =20 per original task.
            count=2 if j==i%7 else 3
            candidates=sorted(cells[i,j],key=lambda x:x['id']);assert len(candidates)>=count
            levels=collections.defaultdict(list)
            for t in candidates:levels[t['difficulty_level']].append(t)
            exact={k:count*len(v)/len(candidates) for k,v in levels.items()};quota={k:int(v) for k,v in exact.items()}
            for k in sorted(exact,key=lambda k:(-(exact[k]-quota[k]),k))[:count-sum(quota.values())]:quota[k]+=1
            for k in sorted(levels):
                for t in rng.sample(levels[k],quota[k]):
                    tid=mapping[t['name']];assert tid==t['id']-1
                    states=bench.get_task_init_states(tid);assert len(states)>0
                    rows.append(dict(suite='libero_10',task_id=tid,base_task=b,category=cat,difficulty_level=k,
                        name=t['name'],instruction=bench.get_task(tid).language,seed=7,initial_state_index=0))
    rng.shuffle(rows);assert len(rows)==len({r['task_id'] for r in rows})==200
    text=''.join(f'libero_10,{r["task_id"]}\n' for r in rows)
    (OUT/'tasks200.txt').write_text(text);(OUT/'smoke_task.txt').write_text(f'libero_10,{rows[0]["task_id"]}\n')
    summary=dict(episodes=200,source_variants=2519,seed=7,sampling_seed=20261015,trials=1,
        base_task_counts=dict(collections.Counter(r['base_task'] for r in rows)),
        category_counts=dict(collections.Counter(r['category'] for r in rows)),
        task_sha256=hashlib.sha256(text.encode()).hexdigest(),classification_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        selection='20 per original Long task; 28/29 per OOD category; difficulty proportional inside each cell; outcomes not consulted')
    (OUT/'task_manifest.json').write_text(json.dumps(dict(settings=summary,tasks=rows),indent=2))
    (OUT/'sampling_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
