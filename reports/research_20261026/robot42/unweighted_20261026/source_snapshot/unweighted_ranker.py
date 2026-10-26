import json,os
from pathlib import Path
import torch
from weighted_ar2_ranker import ARRanker
class UnweightedRanker(ARRanker):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw)
  self.task_weights={tid:torch.ones_like(w) for tid,w in self.task_weights.items()}
  assert all(torch.equal(w,torch.ones_like(w)) for w in self.task_weights.values())
  Path(os.environ['GUIDANCE_SIX_OUT'],'unweighted_objective.json').write_text(json.dumps(dict(method='unweighted_guidance',all_feature_weights=1,effective_beta=0,loss='L1 averaged across both AR2 horizons',task_ids=sorted(self.task_weights))))
