"""Vanilla LIBERO Long, seed-major order 7/17/27/37, official init0, frozen T5 cache."""
from wand.api import library as _wand_library
import hashlib, json, os, sys, time
from pathlib import Path
ROOT = Path('/data/pxchen/PACT-WAM')
OUT = Path(os.environ['ORIGINAL40_OUT'])
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import libero.libero.benchmark as benchmark
assert '/libero-suite/LIBERO/' in benchmark.__file__, benchmark.__file__
suite = benchmark.get_benchmark_dict()['libero_10']()
assert suite.get_num_tasks() == 10

def save(name, value):
    p = OUT/name
    q = p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    q.replace(p)

rows = []
for i in range(10):
    task = suite.get_task(i)
    from libero.libero import get_libero_path
    state_path = Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file
    states = torch.load(state_path, map_location='cpu', weights_only=False)
    assert len(states) >= 4
    rows.append(dict(task_id=i, instruction=task.language, bddl=task.bddl_file,
        initial_state_file=str(state_path), initial_state_file_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),
        initial_state_indices=[0], initial_state_hashes=[hashlib.sha256(np.asarray(states[0]).tobytes()).hexdigest()]))
save('task_manifest.json', dict(benchmark_source=benchmark.__file__, tasks=rows, total_episodes=40))
if '--metadata-only' in sys.argv:
    print('VANILLA_METADATA_PASSED', flush=True)
    sys.exit(0)

source = ROOT/'experiments/libero/eval_libero_single.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == os.environ['ORIGINAL40_NATIVE_SHA256']
ns = dict(__name__='__main__', __file__=str(source))
code = source.read_text()
entry = 'if __name__ == "__main__":\n    eval_single_process()'
assert code.count(entry) == 1
code = code.replace(entry, 'if __name__ == "__main__":\n    pass')
exec(compile(code, str(source), 'exec'), ns)
ns['_evaluation_prompts'] = lambda cfg: [ns['DEFAULT_PROMPT'].format(task=r['instruction']) for r in rows]
original_preencode = ns['_preencode_prompt_contexts']

def preencode(cfg, **kw):
    cache = original_preencode(cfg, **kw)
    from experiments.libero.text_context_cache import save_cache, load_cache
    audits = list(OUT.glob('text_context_fingerprints_*.json'))
    assert len(audits) == 1
    identity = json.loads(audits[0].read_text())['identity']
    path = OUT/'frozen_t5_contexts.pt'
    checksum = save_cache(cache, path, identity)
    loaded, digest = load_cache(path, identity, list(cache), checksum)
    assert digest == checksum and len(loaded) == 10
    for key in cache:
        assert all(torch.equal(a,b) for a,b in zip(cache[key],loaded[key]))
    save('t5_reference.json',dict(path=str(path),sha256=checksum,prompts=10,encoding_batch_size=1,
        policy='All subsequent vanilla LIBERO10 comparisons must load this cache and verify its hash.'))
    return loaded

ns['_preencode_prompt_contexts'] = preencode
original_run = ns['_run_task_to_file']

def run_all(**kwargs):
    cfg = kwargs['cfg']
    assert cfg.seed == 7 and cfg.EVALUATION.num_trials == 1 and cfg.EVALUATION.replan_steps == 8
    assert not cfg.EVALUATION.vjepa2_ac.enabled
    ns['OmegaConf'].save(cfg, OUT/'runtime_config.yaml')
    results = []
    for seed in (7,17,27,37):
        for i in range(10):
            cfg.seed = seed
            ns['set_global_seed'](seed, get_worker_init_fn=False)
            save('progress.json',dict(state='RUNNING',seed=seed,task_id=i,completed_episodes=len(results),total_episodes=40,updated=time.strftime('%F %T')))
            kwargs['suite_name']='libero_10'; kwargs['task_id']=i
            kwargs['output_root']=OUT/f'seed_{seed}'
            path, result = original_run(**kwargs)
            results.append(dict(seed=seed,task_id=i,trial=0,initial_state_index=0,instruction=rows[i]['instruction'],successes=result['successes'],trials=1,
                duration_s=result['duration'],result_path=str(path)))
            save('summary.json',dict(partial=len(results)<40,seeds=[7,17,27,37],order='seed-major; task0..9 within each seed',
                completed_episodes=len(results),total_episodes=40,successes=sum(r['successes'] for r in results),replan_steps=8,rollouts=results))
    save('progress.json',dict(state='COMPLETED',completed_episodes=40,total_episodes=40,updated=time.strftime('%F %T')))
    return path, result

ns['_run_task_to_file'] = run_all
ns['eval_single_process']()
