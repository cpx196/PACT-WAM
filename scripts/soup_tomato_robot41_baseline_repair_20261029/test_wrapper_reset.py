import ast,pathlib
p=pathlib.Path(__file__).with_name('eval_soup41.py');tree=ast.parse(p.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='native_predictor');ns={};exec(compile(ast.Module(body=[f],type_ignores=[]),str(p),'exec'),ns)
def native():return 'native'
def stale():raise AssertionError('Stale weighted wrapper must not execute')
state={'_soup41_native_predict':native,'_predict_action_chunk':stale}
for _ in range(2):
 state['_predict_action_chunk']=stale
 assert ns['native_predictor'](state) is native and state['_predict_action_chunk']()=='native'
assert "for arm in ['control']:" in p.read_text()
assert "assert path.exists(), 'Baseline repair must reuse existing T5" in p.read_text()
print('Regression passed: stale arm wrapper is reset; baseline only; immutable T5 required')
