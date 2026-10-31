"""Expose an explicitly computed auxiliary-stream gradient to the native caller."""
import torch
class PrecomputedActionGradient(torch.autograd.Function):
 @staticmethod
 def forward(ctx,clean,evaluate):
  loss,gradient=evaluate(clean)
  assert loss.numel()==1 and gradient.shape==clean.shape
  ctx.save_for_backward(gradient.detach())
  return loss.detach()
 @staticmethod
 def backward(ctx,incoming):
  (gradient,)=ctx.saved_tensors
  return gradient*incoming,None
