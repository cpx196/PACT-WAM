import runpy
from pathlib import Path
import torch
torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
print('Predictor-only; allocator budget 4 GiB',flush=True)
runpy.run_path(str(Path(__file__).with_name('calibrate_remaining7.py')),run_name='__main__')
