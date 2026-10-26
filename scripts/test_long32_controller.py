"""CPU integration checks exercise the actual episode loop with synthetic observations/actions."""
import os,tempfile
os.environ['T4_PILOT_OUT']=tempfile.mkdtemp(prefix='long32_control_test_')
import numpy as np
import torch
import eval_t4_long32 as runtime
from t4_long32_core import selftest
# Exercise the pinned historical-cache loader, including deferred local asset paths.
import json
from pathlib import Path
from types import SimpleNamespace
from experiments.libero.text_context_cache import cache_identity,save_cache
fixture=Path(tempfile.mkdtemp(prefix='long32_cache_test_'))
(fixture/'weights').write_bytes(b'test');(fixture/'tokenizer').mkdir()
class Asset:
    path=None
    def __init__(self,target):self.target=target
    def download_if_necessary(self):self.path=str(self.target)
text,token=Asset(fixture/'weights'),Asset(fixture/'tokenizer')
identity=cache_identity(text.target,token.target,2,torch.float32)
identity={**identity,'encoding_batch_size':'historical_mixed_1_and_5','encoding_provenance':'fixture'}
cache={'test':(torch.zeros(1,2,4),torch.ones(1,2,dtype=torch.bool))}
cache_path=fixture/'cache.pt';checksum=save_cache(cache,cache_path,identity)
(fixture/'manifest.json').write_text(json.dumps(dict(text_cache_path=str(cache_path),text_cache_sha256=checksum,encoding_provenance=identity)))
os.environ['LONG32_BASELINE_CACHE_PATH']=str(cache_path);os.environ['LONG32_BASELINE_CACHE_SHA256']=checksum
runtime.REFERENCE=fixture
runtime.NS={'_resolve_configs':lambda **kw:(None,text,None,token),'_evaluation_prompts':lambda cfg:['test']}
class C(dict):
    __getattr__=dict.__getitem__
cfg=C(model=C(model_id='test',tokenizer_model_id='test',tokenizer_max_len=2),data=C(train=C(context_len=2)))
loaded=runtime.preencode(cfg,device='cpu',dtype=torch.float32)
assert text.path and token.path and torch.equal(loaded['test'][0],cache['test'][0])
class Env:
    def __init__(self,terminal=None):self.n=0;self.terminal=terminal;self.seen=[]
    def step(self,action):
        self.n+=1;self.seen.append(action)
        return self.n,0,self.n==self.terminal,{}
class Monitor:
    def __init__(self):self.inputs=[]
    def start(self,frame,state,commands):self.inputs.append(commands.copy());return commands.copy()
    def finish(self,cache,frame):return dict(error=2.,timing={})
runtime.NS={'_extract_sim_state':lambda obs:np.zeros(8,np.float32),'_agentview_for_jepa':lambda imgs,p:imgs,'get_libero_image':lambda obs,**kw:np.zeros((2,2,3),np.uint8)}
runtime.rng_hash=lambda:'unchanged'
def predict(obs,description,kw,pid,horizon=32):
    return np.repeat((pid*100+np.arange(32,dtype=np.float32))[:,None],7,axis=1),None,None,dict(total_s=0.)
runtime.predict=predict
kw={'processor':None}
def run(arm,limit=700,terminal=None,**opts):
    env=Env(terminal);runtime.MONITOR=Monitor()
    result=runtime.run_episode(env,'test',0,kw,tid=151,seed=7,arm=arm,limit=limit,video=False,**opts)
    return result,env,runtime.MONITOR
checks=['historical_cache_loader_resolves_local_assets_and_verifies_tensors']
r,e,m=run('fixed32');assert r['wam_calls']==22 and r['control_steps']==700 and r['actions'].shape==(700,7)
assert e.seen[31][0]==31 and e.seen[32][0]==100 and r['eligible_decisions']==153
checks.append('fixed32_700_steps_22_calls_exact_plan_boundaries')
r,e,m=run('adaptive32',limit=12,tau=1.,monitor=True);assert r['wam_calls']==3 and r['early']==2
assert [a[0] for a in e.seen]==[0,1,2,3,100,101,102,103,200,201,202,203]
assert all(np.array_equal(a,np.asarray(e.seen[i*4:i*4+4],np.float32)) for i,a in enumerate(m.inputs))
checks.append('adaptive_discards_suffix_predictor_commands_equal_env')
r,e,m=run('random32',limit=12,probability=1.);assert r['wam_calls']==3 and r['early']==2
checks.append('random_same_four_step_grid')
for terminal in (3,4,7,32,33):
 r,e,m=run('adaptive32',terminal=terminal,tau=3.,monitor=True)
 assert r['control_steps']==terminal and r['success'] and r['wam_calls']==(terminal+31)//32
 assert r['windows'][-1]['decision']=='terminal'
 assert r['windows'][-1]['censored']==bool(terminal%4)
checks.append('terminal_before_at_after_boundary_no_extra_action_or_replan')
r,e,m=run('adaptive32',limit=7,tau=3.,monitor=True)
assert r['control_steps']==7 and not r['success'] and r['wam_calls']==1 and r['windows'][-1]['censored']
assert len(m.inputs)==1
checks.append('partial_final_window_not_passed_to_predictor')
print(dict(passed=True,controller_checks=checks,pure_core=selftest()))
