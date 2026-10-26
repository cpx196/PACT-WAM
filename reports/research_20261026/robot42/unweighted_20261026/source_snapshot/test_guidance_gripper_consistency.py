"""CPU-only checks of the real guidance closure; no model loads or rollouts."""
import ast,json,sys
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import Future
import numpy as np
import torch

ROOT=Path('/data/pxchen/PACT-WAM')
source=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'experiments/libero/eval_libero_single.py'
tree=ast.parse(source.read_text())
names={'_denormalize_action','_build_jepa_action_guidance'}
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
utils=ast.parse((ROOT/'experiments/libero/libero_utils.py').read_text())
invert=next(n for n in utils.body if isinstance(n,ast.FunctionDef) and n.name=='invert_gripper_action')
# Extract the execution postprocess itself, rather than duplicating its formula.
predict=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_predict_action_chunk')
start=next(i for i,n in enumerate(predict.body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='_denormalize_action')
postprocess=ast.parse('def execution_postprocess(action, processor, cfg):\n    pass').body[0]
postprocess.body=predict.body[start:start+4]+[ast.Return(value=ast.Name(id='action',ctx=ast.Load()))]
assert isinstance(postprocess.body[3],ast.If) and 'binarize_gripper' in ast.unparse(postprocess.body[3])
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*functions,invert,postprocess],type_ignores=[])
namespace=dict(torch=torch,np=np,Future=Future)
exec(compile(ast.fix_missing_locations(module),str(source),'exec'),namespace)
class Config(dict):
    __getattr__=dict.__getitem__
class Normalizer:
    # Preserve float64 normalizer arithmetic, like the dataset stats loader.
    scale=torch.tensor([1.25,.8,2.,1.5,.75,1.1,2.],dtype=torch.float64)
    offset=torch.tensor([.1,-.2,.3,-.4,.5,-.6,-1.],dtype=torch.float64)
    def backward(self,action):return (action-self.offset)/self.scale
normalizer=Normalizer()
processor=SimpleNamespace(shape_meta={'action':[{'key':'action'}]},normalizer=SimpleNamespace(normalizers={'action':{'action':normalizer}}))
class Adapter:
    low_level_steps_per_ac_step=4
    def state_from_observation(self,obs):return np.zeros(8,np.float32)
    def convert_torch(self,actions,state):return SimpleNamespace(actions=actions,states=state[None].expand(len(actions),-1))
class Ranker:
    device=torch.device('cpu');dtype=torch.float32
    seen=None
    def encode_guidance_context(self,**kwargs):return torch.zeros(1,1,1),torch.zeros(2,1,1)
    def differentiable_energy(self,**kwargs):
        self.seen=kwargs['ac_actions']
        # Deliberately make loss sensitive to grip: detachment must still give zero grip gradient.
        return self.seen.square().sum(dim=(1,2))+3*self.seen[...,6].sum(dim=1)
checks=[]
for binary in (True,False):
    ranker=Ranker();cfg=Config(EVALUATION=Config(binarize_gripper=binary,vjepa2_ac=Config(guidance=Config(ac_steps=2))))
    closure,horizon=namespace['_build_jepa_action_guidance'](obs={},previous_jepa_image=None,current_jepa_image=None,predicted_future_frames=[None]*3,processor=processor,ac_ranker=ranker,ac_adapter=Adapter(),cfg=cfg,model_device='cpu')
    assert horizon==8
    x=torch.linspace(-.9,.9,2*12*7).reshape(2,12,7).requires_grad_(True)
    with torch.no_grad():x[:,:8,6]=torch.tensor([-2.,-1.,-1e-7,0.,1e-7,.25,1.,2.])
    original=x.detach().clone();loss=closure(x);gradient=torch.autograd.grad(loss,x)[0]
    expected=namespace['execution_postprocess'](x.detach()[:,:8],processor,cfg)
    assert np.array_equal(ranker.seen[...,6].detach().numpy(),expected[...,6].astype(np.float32))
    assert torch.count_nonzero(gradient[...,6])==0
    assert torch.count_nonzero(gradient[:,8:])==0
    assert torch.count_nonzero(gradient[:,:8,:6])>0 and torch.isfinite(gradient).all()
    if binary:assert set(ranker.seen[...,6].detach().flatten().tolist())=={-1.,0.,1.}
    pose_reference=(x[:,:8,:6]-normalizer.offset[:6].float())/normalizer.scale[:6].float()
    assert torch.equal(ranker.seen[...,:6],pose_reference)
    expected_pose_gradient=2*pose_reference/normalizer.scale[:6].float()/x.shape[0]
    assert torch.allclose(gradient[:,:8,:6],expected_pose_gradient,atol=1e-6,rtol=1e-6)
    assert torch.equal(x.detach(),original),'Loss mutated model candidate'
    updated=x.detach()-.01*gradient
    assert torch.equal(updated[...,6],original[...,6])
    checks.append(dict(binarize_gripper=binary,executed_gripper_exact=True,zero_gripper_gradient=True,pose_forward_and_gradient_preserved=True,horizon_mask_preserved=True,direct_guidance_update_preserves_gripper=True))
# Precision boundary: float32 reimplementation rounds this case to zero, but
# the real float64 execution normalizer places it on the positive side.
normalizer.offset=normalizer.offset.clone();normalizer.offset[-1]=-1.-1e-8
ranker=Ranker();cfg.EVALUATION['binarize_gripper']=True
closure,_=namespace['_build_jepa_action_guidance'](obs={},previous_jepa_image=None,current_jepa_image=None,predicted_future_frames=[None]*3,processor=processor,ac_ranker=ranker,ac_adapter=Adapter(),cfg=cfg,model_device='cpu')
x=torch.zeros(1,8,7,requires_grad=True);closure(x)
expected=namespace['execution_postprocess'](x.detach(),processor,cfg)
assert np.array_equal(ranker.seen[...,6].detach().numpy(),expected[...,6].astype(np.float32))
assert torch.all(ranker.seen[...,6]==-1)
print(json.dumps(dict(passed=True,device='CPU',model_loaded=False,rollouts=0,checks=checks,float64_normalizer_threshold_boundary_preserved=True)))
