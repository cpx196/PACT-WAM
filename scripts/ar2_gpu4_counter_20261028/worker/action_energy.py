"""Pure tensor execution mapping; no CPU action copies in the energy graph."""
import torch

def execution_gripper(clean,scale,offset,binarize=True):
 # Reproduce CPU float32 input + normalizer promotion and rounding after each
 # operation. Double intermediates prevent fused GPU arithmetic from skipping
 # float32 rounding at the open/close boundary; gradients remain detached.
 dtype=torch.promote_types(torch.float32,torch.promote_types(scale.dtype,offset.dtype))
 x=clean.detach().float()[:,:8,6:7].double()
 x=(x-offset.double()).to(dtype).double()
 x=(x/scale.double()).to(dtype).double()
 x=(x*2.0).to(dtype).double()
 x=(x-1.0).to(dtype)
 return (-torch.sign(x) if binarize else -x).float()

def make_action_energy(energy,scale,offset,raw_scale,raw_offset,initial_state,binarize):
 def action_energy(clean,z,target):
  denorm=(clean.float()[:,:8]-offset)/scale
  grip=execution_gripper(clean,raw_scale,raw_offset,binarize)
  executable=torch.cat((denorm[...,:6],grip),-1)
  # RawAdapter.convert_torch is exactly these tensors for the current AR2.
  states=initial_state[None].expand(len(clean),-1)
  return energy(z,target,executable,states).mean()
 return action_energy
