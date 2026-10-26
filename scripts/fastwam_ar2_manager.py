"""Use the original persistent manager with a run-local JEPA worker."""
from pathlib import Path
source = Path('/data/pxchen/PACT-WAM/experiments/libero/run_libero_manager.py')
code = source.read_text()
needle = '"experiments/libero/eval_libero_single.py"'
assert code.count(needle) == 1
code = code.replace(needle, '"scripts/fastwam_ar2_eval.py"')
exec(compile(code, str(source), 'exec'), {'__name__':'__main__','__file__':str(source)})
