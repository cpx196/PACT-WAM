"""Bounded CPU-only read-ahead; yields batches in exactly the sampler order."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import threading
from torch.utils.data._utils.collate import default_collate

class PrefetchBatches:
    def __init__(self,dataset,batches,workers=4,depth=8,pin=False):
        self.dataset=dataset;self.batches=batches;self.workers=workers;self.depth=depth;self.pin=pin
        self.local=threading.local()
    def __iter__(self):
        def load(indices):
            if not hasattr(self.local,'dataset'):
                # Metadata/LRU ownership stays local to each reader thread.
                self.local.dataset=type(self.dataset)(self.dataset.rows)
            return default_collate([self.local.dataset[i] for i in indices])
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            source=iter(self.batches);pending=deque()
            for _ in range(self.depth):
                try:pending.append(pool.submit(load,next(source)))
                except StopIteration:break
            while pending:
                batch=pending.popleft().result()
                try:pending.append(pool.submit(load,next(source)))
                except StopIteration:pass
                if self.pin:batch={k:v.pin_memory() for k,v in batch.items()}
                yield batch
