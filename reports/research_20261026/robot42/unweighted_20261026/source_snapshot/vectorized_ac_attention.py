"""Inference-only, opt-in vectorization; unchanged parameters and native RoPE."""
import types
import torch
import torch.nn.functional as F
from src.models.utils.modules import rotate_queries_or_keys

def forward_vectorized(self,x,mask=None,attn_mask=None,T=None,H=None,W=None,action_tokens=0):
    B,N,C=x.shape
    if mask is not None:
        mask=mask.unsqueeze(1).repeat(1,self.num_heads,1)
    else:
        mask=torch.arange(int(T*H*W),device=x.device)
    d_mask,h_mask,w_mask=self.separate_positions(mask,H,W)
    h_mask*=self.grid_size/H;w_mask*=self.grid_size/W
    if action_tokens>0:
        grid=x.view(B,-1,action_tokens+H*W,C)
        a=grid[:,:,:action_tokens,:].flatten(1,2)
        aqkv=self.qkv(a).unflatten(-1,(3,self.num_heads,-1)).permute(2,0,3,1,4)
        aq,ak,av=aqkv.unbind(0)
        # Original loop rotates each action index with the SAME temporal positions.
        # Flattening is time-major then action-major; do not use action index as RoPE time.
        pos=torch.arange(T,device=x.device).repeat_interleave(action_tokens)
        aq=torch.cat((rotate_queries_or_keys(aq[...,:self.d_dim],pos),aq[...,self.d_dim:]),-1)
        ak=torch.cat((rotate_queries_or_keys(ak[...,:self.d_dim],pos),ak[...,self.d_dim:]),-1)
        x=grid[:,:,action_tokens:,:].flatten(1,2)
    qkv=self.qkv(x).unflatten(-1,(3,self.num_heads,-1)).permute(2,0,3,1,4)
    q,k,v=qkv.unbind(0)
    def rotate(t):
        d=self.d_dim;h=self.h_dim;w=self.w_dim
        parts=[rotate_queries_or_keys(t[...,:d],d_mask),
               rotate_queries_or_keys(t[...,d:d+h],h_mask),
               rotate_queries_or_keys(t[...,d+h:d+h+w],w_mask)]
        if d+h+w<self.head_dim:parts.append(t[...,d+h+w:])
        return torch.cat(parts,-1)
    q=rotate(q);k=rotate(k)
    if action_tokens>0:
        def merge(t,a):
            return torch.cat((a.view(B,self.num_heads,T,action_tokens,-1),t.view(B,self.num_heads,T,H*W,-1)),dim=3).flatten(2,3)
        q=merge(q,aq);k=merge(k,ak);v=merge(v,av)
    if attn_mask is not None or self.use_sdpa:
        x=F.scaled_dot_product_attention(q,k,v,dropout_p=self.proj_drop_prob,is_causal=self.is_causal,attn_mask=attn_mask)
    else:
        attn=(q@k.transpose(-2,-1))*self.scale
        x=self.attn_drop(attn.softmax(-1))@v
    return self.proj_drop(self.proj(x.transpose(1,2).reshape(B,N,C)))

def enable_vectorized(model):
    for block in model.backbone.predictor_blocks:
        block.attn.forward=types.MethodType(forward_vectorized,block.attn)
