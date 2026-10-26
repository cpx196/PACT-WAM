"""Paper-style per-latent-element variance weights; retain existing L1 energy."""
import hashlib,json,os
from pathlib import Path
import torch
from fastwam_ar2_eval import ARRanker as BaseRanker,RawAdapter

def variance_weights(variance):
    assert variance.shape==(2,576,1024)
    assert torch.isfinite(variance).all() and (variance>=0).all()
    means=variance.mean(dim=(1,2),keepdim=True)
    assert (means>0).all(), 'Predictor has no measured action response'
    # Paper Eq15, epsilon=1e-8, beta=1 (full modulation; fixed before evaluation).
    return variance/(means+1e-8)

def weighted_l1(p1,p2,target,weights):
    return torch.stack([((p1-target[0]).abs()*weights[0]).mean((1,2)),
                        ((p2-target[1]).abs()*weights[1]).mean((1,2))],dim=1).mean(1)

class ARRanker(BaseRanker):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        registry=json.loads(Path(os.environ['ACTION_WEIGHTS_REGISTRY']).read_text())
        self.task_weights={}
        for item in registry['tasks']:
            path=Path(item['path'])
            assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
            payload=torch.load(path,map_location='cpu',weights_only=True)
            assert payload['beta']==1.0 and payload['weights'].shape==(2,576,1024)
            assert payload['predictor_sha256']==registry['predictor_sha256']
            w=payload['weights'].to(self.device,self.dtype)
            for tid in item['case_ids']:self.task_weights[tid]=w
        self.active_task_id=None

    def differentiable_energy(self,current_rep,target_reps,ac_actions,ac_states):
        assert self.active_task_id in self.task_weights
        am,ast=self.stats['action'];sm,sst=self.stats['state']
        actions=(ac_actions-am)/ast;state=(ac_states-sm)/sst
        with torch.autocast('cuda',enabled=False):
            p1=self.predictor(current_rep.expand(len(actions),-1,-1),actions[:,:4],state,0)
            p2=self.predictor(p1,actions[:,4:8],state,1)
            loss=weighted_l1(p1,p2,target_reps,self.task_weights[self.active_task_id])
        assert torch.isfinite(loss).all()
        self.calls+=1
        return loss
