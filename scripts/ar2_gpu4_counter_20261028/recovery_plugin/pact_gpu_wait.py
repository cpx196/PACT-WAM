"""Process-local vLLM startup gate; no installed packages or model settings changed."""
import os,sys,time,json,pathlib,importlib.abc,importlib.machinery
OUT=pathlib.Path('/data/pxchen/PACT-WAM/evaluate_results/ar2_gpu4_counter_20261028')
NAME='vllm.v1.worker.gpu_worker'
def patch(module):
 cls=module.Worker
 if getattr(cls.init_device,'_pact_wait',False):return
 original=cls.init_device
 def init_device(self):
  import pynvml
  pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByUUID('GPU-d152c6a7-1dcf-0ff2-1802-80647045e446')
  last=0
  while True:
   mem=pynvml.nvmlDeviceGetMemoryInfo(handle);now=time.monotonic()
   # Conservative guard: the original .97 cache budget and a small margin.
   if mem.free>=int(mem.total*.97)+32*2**20:break
   if now-last>=5:
    q=OUT/'vllm_waiting_gpu4.json.tmp';q.write_text(json.dumps(dict(state='CPU_READY_WAITING_GPU4',pid=os.getpid(),free_mib=mem.free/2**20,required_mib=(mem.total*.97+32*2**20)/2**20,time=time.strftime('%F %T'))));q.replace(OUT/'vllm_waiting_gpu4.json');last=now
   time.sleep(.1)
  (OUT/'vllm_waiting_gpu4.json').write_text(json.dumps(dict(state='GPU4_AVAILABLE_STARTING',pid=os.getpid(),time=time.strftime('%F %T'))))
  return original(self)
 init_device._pact_wait=True;cls.init_device=init_device
class Loader(importlib.abc.Loader):
 def __init__(self,base):self.base=base
 def create_module(self,spec):return self.base.create_module(spec) if hasattr(self.base,'create_module') else None
 def exec_module(self,module):self.base.exec_module(module);patch(module)
class Finder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname!=NAME:return None
  spec=importlib.machinery.PathFinder.find_spec(fullname,path)
  if spec:spec.loader=Loader(spec.loader)
  return spec
def register():
 if NAME in sys.modules:patch(sys.modules[NAME])
 elif not any(isinstance(f,Finder) for f in sys.meta_path):sys.meta_path.insert(0,Finder())
