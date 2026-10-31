import json,sys,bisect,collections
from pathlib import Path
p=Path(sys.argv[1]);es=json.load(open(p/'requested_async_trace.json'))['traceEvents']
iters=[e for e in es if e.get('cat')=='user_annotation' and e.get('name','').startswith('iteration/')]
regions=[e for e in es if e.get('cat')=='cpu_op' and e.get('name')=='Torch-Compiled Region: 4/0']
ks=[e for e in es if e.get('cat')=='kernel'];ls=[e for e in es if e.get('cat') in ['cuda_runtime','cuda_driver'] and 'correlation' in e.get('args',{})];corr={e['args']['correlation']:e for e in ls}
def union(xs):
 out=[]
 for a,b in sorted(xs):
  if out and a<=out[-1][1]:out[-1][1]=max(b,out[-1][1])
  else:out.append([a,b])
 return out
def cover(xs,a,b):return sum(max(0,min(b,y)-max(a,x)) for x,y in xs if x<b and y>a)
out=[]
for it in iters:
 rr=sorted([r for r in regions if r['tid']==it['tid'] and it['ts']<=r['ts']<it['ts']+it['dur']],key=lambda r:r['ts']);assert len(rr)==10
 for i,r in enumerate(rr):
  cs={e['args']['correlation'] for e in ls if e['tid']==r['tid'] and r['ts']<=e['ts']<r['ts']+r['dur']}
  main=sorted([e for e in ks if e['args'].get('correlation') in cs],key=lambda e:e['ts']);assert main
  stream=main[0]['args']['stream'];device=main[0]['args']['device'];assert all(k['args']['stream']==stream for k in main)
  others=[k for k in ks if k['args']['device']==device and k['args']['stream']!=stream and k['ts']<main[-1]['ts']+main[-1]['dur'] and k['ts']+k['dur']>main[0]['ts']]
  otherunion=union([(k['ts'],k['ts']+k['dur']) for k in others]);convunion=union([(k['ts'],k['ts']+k['dur']) for k in others if 'cudnn' in k['name'].lower() and 'bf16' in k['name'].lower()])
  gaps=queued=overlap=queuedoverlap=late=convoverlap=0
  for prev,nxt in zip(main,main[1:]):
   a=prev['ts']+prev['dur'];b=nxt['ts']
   if b<=a:continue
   launch=corr[nxt['args']['correlation']];submitted=launch['ts']+launch['dur'];q=max(a,submitted)
   gaps+=b-a;late+=max(0,min(b,submitted)-a);overlap+=cover(otherunion,a,b)
   if q<b:queued+=b-q;queuedoverlap+=cover(otherunion,q,b);convoverlap+=cover(convunion,q,b)
  d=dict(mode=it['name'].split('/')[-1],iteration=iters.index(it),step=i+1,kernels=len(main),cpu_region_ms=r['dur']/1000,kernel_sum_ms=sum(k['dur'] for k in main)/1000,kernel_span_ms=(main[-1]['ts']+main[-1]['dur']-main[0]['ts'])/1000,gaps_ms=gaps/1000,host_not_fully_submitted_gap_ms=late/1000,queued_gap_ms=queued/1000,queued_gap_aux_overlap_ms=queuedoverlap/1000,queued_gap_aux_conv_overlap_ms=convoverlap/1000,aux_overlap_entire_step_ms=cover(otherunion,main[0]['ts'],main[-1]['ts']+main[-1]['dur'])/1000)
  out.append(d)
(p/'action_contention_trace_audit_20261028.json').write_text(json.dumps(out,indent=2))
for r in out:
 if r['mode'].endswith('prefill_first') and r['step']<=8:print(json.dumps(r))
