"""Read-only CPU analysis of completed online guidance diagnostics."""
import ast,glob,json,re,statistics
from pathlib import Path
ROOT=Path('/data/pxchen/PACT-WAM/evaluate_results')
NEW=ROOT/'fastwam_guidance789_ar2_epoch10_full144_20261015'
OLD=ROOT/'guidance789_rawgrad4x_long36_full144_libero10_calibrated_step2000_20261007'

def outcomes(root):
    result={}
    for seed in (7,17,27,37):
        for file in (root/f'seed_{seed}/libero_10').glob('gpu*_task*_results.json'):
            d=json.loads(file.read_text());result[seed,d['task_id']]=d
    return result

def summarize(rows):
    b=[r['loss'] for r in rows];a=[r['loss_after_clean_step'] for r in rows]
    return dict(nodes=len(rows),replans=len(rows)//3,loss_before=statistics.mean(b),loss_after=statistics.mean(a),
        relative_drop_pct=100*(sum(b)-sum(a))/sum(b),
        decrease_rate_pct=100*sum(y<x for x,y in zip(b,a))/len(rows),
        increase_rate_pct=100*sum(y>x for x,y in zip(b,a))/len(rows),
        cap_rate_pct=100*sum(r['unclipped_delta_rms']>.0200001 for r in rows)/len(rows),
        correction_rms=statistics.mean(r['correction_rms'] for r in rows))

def main():
    new,old=outcomes(NEW),outcomes(OLD);assert len(new)==len(old)==144 and new.keys()==old.keys()
    nodes={}
    for seed in (7,17,27,37):
        pending=[]
        for path in (NEW/f'seed_{seed}/.worker_pool/logs').glob('worker_*.log'):
            for line in path.read_text(errors='replace').splitlines():
                marker='V-JEPA2-AC action guidance diagnostics='
                if marker in line:pending.extend(ast.literal_eval(line.split(marker,1)[1]))
                found=re.search(r'worker \S+ completed libero_10,(\d+):',line)
                if found:
                    key=seed,int(found[1]);assert key not in nodes and pending
                    nodes[key]=pending;pending=[]
        assert not pending
    assert nodes.keys()==new.keys()
    groups={k:[] for k in ('new_harm_vs_old_guidance','new_rescue_vs_old_guidance','both_success','both_failure')}
    cases=[]
    for key,data in new.items():
        a=bool(data['successes']);b=bool(old[key]['successes'])
        group=('both_success' if a else 'both_failure') if a==b else ('new_rescue_vs_old_guidance' if a else 'new_harm_vs_old_guidance')
        groups[group].append(key)
        if a!=b:cases.append(dict(seed=key[0],task_id=key[1],group=group,description=data['task_description'],**summarize(nodes[key])))
    result=dict(baseline='old JEPA calibrated full-guidance, NOT no-guidance',
        total_episodes=144,groups={g:dict(episodes=len(keys),**summarize([r for k in keys for r in nodes[k]]),
            episode_mean_drop_pct=statistics.mean(summarize(nodes[k])['relative_drop_pct'] for k in keys),
            first4=summarize([r for k in keys for r in nodes[k][:12]]),
            first20=summarize([r for k in keys for r in nodes[k][:60]])) for g,keys in groups.items()},
        changed_cases=sorted(cases,key=lambda d:(d['group'],d['seed'],d['task_id'])))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
