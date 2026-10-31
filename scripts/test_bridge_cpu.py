"""Check the auxiliary gradient bridge against native autograd, without CUDA."""
import json,torch
from async_guidance_bridge import PrecomputedActionGradient

def objective(x):
 # Match pose-only guidance and retain a discrete detached gripper input.
 pose=x[:,:8,:6];grip=torch.sign(x[:,:8,6:7].detach())
 return (pose.square()+.03*pose.sin()).mean()+.2*grip.mean()
def evaluate(x):
 with torch.enable_grad():
  separate=x.detach().clone().requires_grad_(True);loss=objective(separate);gradient=torch.autograd.grad(loss,separate)[0]
 return loss,gradient
results=[]
for multiplier in [1.,.37]:
 torch.manual_seed(7);x=torch.randn(1,32,7,dtype=torch.float64,requires_grad=True);y=x.detach().clone().requires_grad_(True)
 original=objective(x);bridge=PrecomputedActionGradient.apply(y,evaluate)
 a=torch.autograd.grad(original*multiplier,x)[0];b=torch.autograd.grad(bridge*multiplier,y)[0]
 assert torch.equal(original,bridge) and torch.allclose(a,b,rtol=1e-12,atol=1e-14)
 assert torch.count_nonzero(b[...,6])==0 and torch.count_nonzero(b[:,8:])==0
 results.append(dict(multiplier=multiplier,loss_exact=True,gradient_max_error=float((a-b).abs().max()),gripper_gradient_zero=True,unexecuted_suffix_zero=True))
print(json.dumps(dict(passed=True,checks=results)))
