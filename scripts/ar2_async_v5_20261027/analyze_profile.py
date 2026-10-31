"""Correlate prefill kernel gaps with host submission and auxiliary kernels.

Usage: python analyze_profile.py /path/to/requested_async_trace.json
Region 3/0 is the prefill graph in the recorded v5 source/compile ordering.
"""
import json,bisect,sys
from pathlib import Path
p=Path(sys.argv[1]);es=json.loads(p.read_text())['traceEvents']
regions=[e for e in es if e.get('cat')=='cpu_op' and e.get('name')=='Torch-Compiled Region: 3/0']
assert len(regions)==2,'Check graph identifiers against the captured source before analysis'
ks=[e for e in es if e.get('cat')=='kernel']
launches=[e for e in es if e.get('cat') in ['cuda_runtime','cuda_driver'] and 'correlation' in e.get('args',{})]
by_corr={e['args']['correlation']:e for e in launches};result=[]
for r in regions:
 cs={e['args']['correlation'] for e in launches if e.get('tid')==r['tid'] and r['ts']<=e.get('ts',0)<=r['ts']+r['dur']}
 main=sorted([e for e in ks if e['args'].get('correlation') in cs],key=lambda e:e['ts']);assert main
 stream=main[0]['args']['stream'];device=main[0]['args']['device'];assert all(e['args']['stream']==stream for e in main)
 other=sorted([(e['ts'],e['ts']+e['dur']) for e in ks if e['args']['device']==device and e['args']['stream']!=stream]);merged=[]
 for a,b in other:
  if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
  else:merged.append([a,b])
 starts=[x[0] for x in merged]
 def covered(a,b):
  i=max(0,bisect.bisect_right(starts,a)-1);total=0
  while i<len(merged) and merged[i][0]<b:
   total+=max(0,min(b,merged[i][1])-max(a,merged[i][0]));i+=1
  return total
 gaps=queued=covered_all=covered_queued=0
 for prev,nxt in zip(main,main[1:]):
  a=prev['ts']+prev['dur'];b=nxt['ts']
  if b<=a:continue
  gaps+=b-a;overlap=covered(a,b);covered_all+=overlap;cpu=by_corr[nxt['args']['correlation']]
  if cpu['ts']+cpu['dur']<=a:queued+=b-a;covered_queued+=overlap
 result.append(dict(prefill_cpu_ms=r['dur']/1000,prefill_kernel_sum_ms=sum(e['dur'] for e in main)/1000,prefill_kernel_span_ms=(main[-1]['ts']+main[-1]['dur']-main[0]['ts'])/1000,prefill_between_kernel_gaps_ms=gaps/1000,gaps_covered_by_other_stream_kernels_ms=covered_all/1000,gaps_next_kernel_already_submitted_ms=queued/1000,queued_gaps_covered_by_other_stream_ms=covered_queued/1000,main_stream=stream,aux_streams=sorted({e['args']['stream'] for e in ks if e['args']['device']==device and e['args']['stream']!=stream})))
(p.parent/'profile_stream_contention.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
