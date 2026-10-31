import json,pathlib,importlib.util,sys
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('controller',HERE/'control.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert m.health(),'vLLM health failed'
ps=m.subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert ps=='GPU-d152c6a7-1dcf-0ff2-1802-80647045e446'
assert pathlib.Path('/data/pxchen/envs/jepa-wam/bin/python').exists()
assert pathlib.Path('/opt/nvidia/nsight-compute/2024.3.2/ncu').exists()
for q in (HERE/'worker').glob('*.py'):compile(q.read_text(),str(q),'exec')
assert "target=torch.device('cuda:0')" in (HERE/'worker/cpu_model_ready.py').read_text()
assert 'cudaProfilerStart()' in (HERE/'worker/bench_async_timing.py').read_text()
assert 'torch.profiler.profile(' not in (HERE/'worker/bench_async_timing.py').read_text()
# Check tracked-identity logic with a harmless short-lived child, without touching vLLM.
p=m.subprocess.Popen(['/bin/sleep','0.1']);item=m.ident(p.pid);assert m.owned_alive(item)
p.wait();assert not m.owned_alive(item)
print('PASS: GPU4 identity, vLLM health, tools, script syntax, capture boundaries, and process identity guard. No GPU workload or vLLM stop performed.')
