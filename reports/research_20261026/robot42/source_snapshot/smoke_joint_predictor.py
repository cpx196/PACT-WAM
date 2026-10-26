"""Three optimizer steps only; native AC backbone, joint-latent target."""
import sys, time, json, os
from functools import partial
from pathlib import Path
import torch
import numpy as np
from torch import nn
from torch.utils.checkpoint import checkpoint
sys.path.insert(0, '/data/pxchen/vjepa2')
from src.models.ac_predictor import vit_ac_predictor

DATA=Path('/data/pxchen/PACT-WAM/artifacts/predictor21_joint20_stride20_20261011')
OUT=Path('/data/pxchen/PACT-WAM/artifacts/predictor21_joint20_smoke_bs16_20261011')

class JointPredictor(nn.Module):
    def __init__(self,action_steps=20):
        super().__init__()
        self.action_steps=action_steps
        self.backbone=vit_ac_predictor(img_size=384,patch_size=16,num_frames=2,tubelet_size=2,
            embed_dim=1024,predictor_embed_dim=1024,depth=24,num_heads=16,
            action_embed_dim=7,use_rope=True,is_frame_causal=False,use_activation_checkpointing=True)
        self.backbone.state_encoder=nn.Linear(8,1024)
        self.backbone._init_weights(self.backbone.state_encoder)
        self.action_position=nn.Parameter(torch.empty(1,20,1024))
        nn.init.trunc_normal_(self.action_position,std=.02)
        if action_steps!=20:
            extra=torch.empty(1,action_steps-20,1024)
            nn.init.trunc_normal_(extra,std=.02)
            self.action_position=nn.Parameter(torch.cat([self.action_position.detach(),extra],dim=1))
    def forward(self,z,a,s):
        m=self.backbone
        x=torch.cat([m.action_encoder(a)+self.action_position,m.state_encoder(s).unsqueeze(1),m.predictor_embed(z)],dim=1)
        assert a.shape[1]==self.action_steps
        assert x.shape[1:]==(576+self.action_steps+1,1024)
        # One current visual block; all 21 known condition tokens visible.
        # Original AC spatial RoPE remains; action order has explicit embeddings.
        for block in m.predictor_blocks:
            kwargs=dict(mask=None,attn_mask=None,T=1,H=24,W=24,action_tokens=self.action_steps+1)
            if self.training:
                mode=os.environ.get('CHECKPOINT_MODE','nonreentrant')
                if mode=='reentrant':x=checkpoint(partial(block,**kwargs),x,use_reentrant=True)
                elif mode=='none':x=block(x,**kwargs)
                else:x=checkpoint(block,x,use_reentrant=False,**kwargs)
            else:x=block(x,**kwargs)
        return m.predictor_proj(m.predictor_norm(x[:,self.action_steps+1:]))

def main():
    if (OUT/'summary.json').exists():raise RuntimeError('Duplicate smoke output')
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(7);np.random.seed(7)
    assert json.loads((DATA/'summary.json').read_text())['status']=='complete'
    index=json.loads((DATA/'index.json').read_text())
    stats=json.loads((DATA/'train_only_stats.json').read_text())
    def load_batch(split):
        rows=[r for r in index if r['split']==split]
        selected=np.linspace(0,len(rows)-1,16,dtype=int)
        batch={k:[] for k in ('current_latent','joint_target','raw_actions','state')}
        for i in selected:
            p=torch.load(DATA/rows[i]['file'],map_location='cpu',weights_only=False)
            for k in batch:batch[k].append(p[k][0])
        batch={k:torch.stack(v).cuda() for k,v in batch.items()}
        for key,name in [('raw_actions','action'),('state','state')]:
            mu=batch[key].new_tensor(stats[name]['mean']);std=batch[key].new_tensor(stats[name]['std']).clamp_min(1e-6)
            batch[key]=(batch[key]-mu)/std
        assert batch['raw_actions'].shape==(16,20,7) and batch['state'].shape==(16,8)
        return batch
    train=load_batch('train');val=load_batch('val')
    model=JointPredictor().cuda();params=sum(p.numel() for p in model.parameters())
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.04)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1.,(step+1)/20))
    config=dict(batch_size=16,depth=24,width=1024,heads=16,parameters=params,
        tokens=597,state_dim=8,action_shape=[20,7],output_shape=[576,1024],
        activation_checkpointing=True,precision='BF16 autocast, FP32 parameters/cache',
        loss='joint latent L1 only',smoke_optimizer_steps=3,checkpoint_resume=True,
        mask='full attention within one known-input block; no future label input')
    (OUT/'config.json').write_text(json.dumps(config,indent=2));print('MODEL',json.dumps(config),flush=True)
    def predict(batch):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return model(batch['current_latent'],batch['raw_actions'],batch['state'])
    model.eval()
    with torch.no_grad():
        before=(predict(val).float()-val['joint_target']).abs().mean().item()
    model.train();initial=model.backbone.predictor_proj.weight.detach().clone()
    records=[];started=time.time()
    for step in range(3):
        ts=time.time();optimizer.zero_grad(set_to_none=True)
        pred=predict(train);assert pred.shape==train['joint_target'].shape==(16,576,1024)
        loss=(pred.float()-train['joint_target']).abs().mean();assert torch.isfinite(loss)
        loss.backward()
        grads={name:float(module.weight.grad.float().norm()) for name,module in [('action',model.backbone.action_encoder),('state',model.backbone.state_encoder),('output',model.backbone.predictor_proj)]}
        assert all(np.isfinite(v) and v>0 for v in grads.values()),grads
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True))
        optimizer.step();scheduler.step();torch.cuda.synchronize()
        record=dict(step=step+1,loss=float(loss),gradient_norm_before_clip=norm,
            clipped=norm>1,projection_gradient_norms=grads,elapsed_s=time.time()-ts)
        records.append(record);print('STEP',json.dumps(record),flush=True)
        if step==1:
            model.eval()
            with torch.no_grad():reference=predict(val).float().cpu()
            path=OUT/'step2.pt'
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                step=2,config=config,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                numpy_rng=np.random.get_state()),path)
            payload=torch.load(path,map_location='cpu',weights_only=False)
            model.load_state_dict(payload['model']);optimizer.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler'])
            torch.set_rng_state(payload['torch_rng']);torch.cuda.set_rng_state_all(payload['cuda_rng']);np.random.set_state(payload['numpy_rng'])
            del payload
            with torch.no_grad():restored=predict(val).float().cpu()
            resume_error=(reference-restored).abs().max().item();assert resume_error==0.,resume_error
            print('RESUME_PASSED',resume_error,flush=True);model.train()
    delta=float((model.backbone.predictor_proj.weight-initial).abs().max());assert delta>0
    model.eval()
    with torch.no_grad():
        after=(predict(val).float()-val['joint_target']).abs().mean().item()
        original=predict(val).float()
        altered=dict(val);altered['raw_actions']=torch.zeros_like(val['raw_actions'])
        action_difference=(predict(altered).float()-original).abs().mean().item()
    summary=dict(status='passed',config=config,records=records,validation_l1_before=before,
        validation_l1_after=after,weight_max_change=delta,resume_max_abs_difference=resume_error,
        zero_normalized_action_output_difference=action_difference,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        elapsed_s=time.time()-started,note='Smoke only; three steps do not establish convergence or useful action sensitivity.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print('SMOKE_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'failure.json').write_text(json.dumps(dict(error=repr(e))))
        raise
