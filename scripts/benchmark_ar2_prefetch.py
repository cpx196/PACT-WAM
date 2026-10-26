import os,json,time
import numpy as np,torch
from torch.utils.data._utils.collate import default_collate
from train_fastwam_ar2_full10 import CacheWindows,CACHE,OUT
from ar2_prefetch import PrefetchBatches

def main():
    torch.set_num_threads(1)
    rows=[r for r in json.load(open(CACHE/'index.json')) if r['split']=='train'];ds=CacheWindows(rows)
    order=np.random.default_rng(7).permutation(len(ds)).tolist()
    batches=[order[i:i+16] for i in range(250*16,250*16+16*16,16)]
    t=time.perf_counter();reference=[default_collate([ds[i] for i in indices]) for indices in batches[:4]]
    direct_s=time.perf_counter()-t
    print(json.dumps(dict(event='DIRECT_TIMING',batches=4,seconds=direct_s)),flush=True)
    t=time.perf_counter();loaded=list(PrefetchBatches(ds,batches,workers=4,depth=8,pin=False));parallel_s=time.perf_counter()-t
    for i,r in enumerate(reference):
        assert all(torch.equal(r[k],loaded[i][k]) for k in r),'Prefetch mismatch'
    result=dict(status='passed',read_threads=4,prefetch_batches=8,exact_equal_batches=4,
        serial_s_per_batch=direct_s/4,parallel_s_per_batch=parallel_s/16,
        note='Timing is an indicative read benchmark; final speed must be measured during training.')
    (OUT/'prefetch_benchmark.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
