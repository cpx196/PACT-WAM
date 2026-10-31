import json,torch
from action_energy import execution_gripper,make_action_energy

checks=[]
for dtype in [torch.float32,torch.float64]:
 s=torch.tensor([2.],dtype=dtype);o=torch.tensor([-1.],dtype=dtype)
 x=torch.zeros(2048,8,7);x[...,6]=torch.linspace(-1e-5,1e-5,2048)[:,None]
 expected=-torch.sign(((x[...,6:7]-o)/s)*2-1).float()
 actual=execution_gripper(x,s,o)
 assert torch.equal(expected,actual)
 compiled=torch.compile(lambda a:execution_gripper(a,s,o),backend='aot_eager',fullgraph=True)
 assert torch.equal(expected,compiled(x))
 checks.append(dict(dtype=str(dtype),gripper_max_abs=0))
energy=lambda z,t,a,s:((a-z)**2).mean((1,2))+s.square().mean(1)
fn=make_action_energy(energy,torch.ones(7),torch.zeros(7),torch.tensor([2.]),torch.tensor([-1.]),torch.zeros(8),True)
x=torch.randn(1,32,7,requires_grad=True);z=torch.zeros(1,8,7)
ref=fn(x,z,z);g=torch.autograd.grad(ref,x)[0]
compiled=torch.compile(fn,backend='aot_eager',fullgraph=True);y=x.detach().clone().requires_grad_(True);v=compiled(y,z,z);cg=torch.autograd.grad(v,y)[0]
assert torch.equal(v,ref) and torch.equal(cg,g) and cg[...,6].abs().max()==0
print(json.dumps(dict(passed=True,boundary_checks=checks,compiled_gradient_max_abs=float((cg-g).abs().max()))))
