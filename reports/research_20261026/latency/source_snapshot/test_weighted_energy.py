"""CPU contract checks: original energy/gradient, broadcasting, estimator."""
import ast
from pathlib import Path
import torch
path=Path(__file__).with_name('weighted_ar2_ranker.py')
tree=ast.parse(path.read_text())
selected=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('variance_weights','weighted_l1')],type_ignores=[])
ns={'torch':torch};exec(compile(selected,str(path),'exec'),ns)
torch.manual_seed(7)
p1=torch.randn(2,576,1024,requires_grad=True)
p2=torch.randn(2,576,1024,requires_grad=True)
target=torch.randn(2,576,1024)
one=torch.ones(2,576,1024)
old=torch.stack([(p1-target[0]).abs().mean((1,2)),(p2-target[1]).abs().mean((1,2))],1).mean(1)
new=ns['weighted_l1'](p1,p2,target,one)
assert torch.equal(old,new)
g0=torch.autograd.grad(old.sum(),(p1,p2),retain_graph=True)
g1=torch.autograd.grad(new.sum(),(p1,p2),retain_graph=True)
assert all(torch.equal(a,b) for a,b in zip(g0,g1))
w=one.clone();w[0,0,0]=9
g2=torch.autograd.grad(ns['weighted_l1'](p1,p2,target,w).sum(),(p1,p2))
assert torch.allclose(g2[0][:,0,0],g1[0][:,0,0]*9)
assert torch.equal(g2[0][:,1:],g1[0][:,1:]) and torch.equal(g2[1],g1[1])
v=torch.ones(2,576,1024);v[0,0,0]=4
weights=ns['variance_weights'](v)
assert torch.allclose(weights.mean((1,2)),torch.ones(2),atol=1e-6)
assert torch.isclose(weights[0,0,0]/weights[0,0,1],torch.tensor(4.))
assert torch.equal(weights[1],torch.ones_like(weights[1]))
try:ns['variance_weights'](torch.zeros_like(v))
except AssertionError:pass
else:raise AssertionError('Zero action-response must not silently produce valid weights')
print('PASS: all-one weights reproduce original L1 and gradients; individual element affects only its gradient; phase normalization and zero-response rejection.')
