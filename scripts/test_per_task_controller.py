"""CPU controller tests with explicit synthetic errors; never part of formal evaluation."""
import os,tempfile,types,json
from pathlib import Path
os.environ['T4_PILOT_OUT']=tempfile.mkdtemp(prefix='t4_controller_test_')
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
os.environ['T4_POLICY_SEED']='57'
import t4_per_task_runtime as pilot
class Node(types.SimpleNamespace): pass
cfg=Node(EVALUATION=Node(max_steps_override=16,task_id=260,num_steps_wait=30,visualize_future_video=False,use_action_ensembler=False,trigger_probe=Node(enabled=False),vjepa2_ac=Node(enabled=False,guidance=Node(enabled=False),camera={})))
class Env:
    def __init__(self,done_at=None):self.done_at=done_at
    def reset(self):self.n=-30;self.executed=[]
    def obs(self):return {'n':self.n,'agentview_image':np.zeros((2,2,3),np.uint8)}
    def set_init_state(self,s):return self.obs()
    def step(self,a):
        self.n+=1
        if self.n>0:self.executed.append(a)
        return self.obs(),0,self.done_at is not None and self.n>=self.done_at,{}
class FakeMonitor:
    error=0.
    def start(self,frame,state,commands):return {'commands':commands.copy()}
    def finish(self,c,frame):return dict(error=self.error,timing={})
pilot.MONITOR=FakeMonitor()
starts=[]
def predict(obs,**kw):
    starts.append(obs['n']);return np.tile(np.arange(7,dtype=np.float32),(32,1)),{'n':obs['n']},None,{'total_s':0.}
pilot.NS={'apply_libero_camera':lambda *a:None,'get_libero_dummy_action':lambda:[0]*7,'get_libero_image':lambda obs,**k:obs,'_predict_action_chunk':predict,'_agentview_for_jepa':lambda imgs,p:np.zeros((2,2,3),np.uint8),'_extract_sim_state':lambda obs:np.zeros(8,np.float32)}
def run(arm,error=0,limit=16,done=None):
    starts.clear();pilot.WINDOWS.clear();cfg.EVALUATION.max_steps_override=limit
    os.environ.update(T4_ARM=arm,T4_MONITOR='1',T4_TAU='1')
    pilot.MONITOR.error=error;env=Env(done)
    pilot.episode(env,None,'test',None,None,cfg,0,action_horizon=32,input_w=2,input_h=2,model_device='cpu')
    return list(starts),pilot.episode.last
assert run('adaptive48',2)[0]==[0,4,8,12]
assert run('adaptive48',0)[0]==[0,8]
assert run('adaptive48',1)[0]==[0,8]
assert run('fixed8',2)[0]==[0,8]
assert run('fixed4',0)[0]==[0,4,8,12]
for limit,done in ((3,None),(16,3),(16,4)):
    _,r=run('fixed8',0,limit,done)
    assert r['valid_decisions']==0 and r['windows'][0]['censored']
assert run('fixed8',0,9)[1]['windows'][-1]['censored']
try:run('adaptive48',float('nan'))
except ValueError:pass
else:raise AssertionError('NaN accepted')
print(json.dumps(dict(passed=True,checks=['adaptive_high_4','adaptive_low_8','strict_threshold','fixed4','fixed8','done_before4','done_at4','max_short_window','max_tail_window','NaN_stop'],synthetic_only=True)))
