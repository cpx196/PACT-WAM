import json,sqlite3,bisect,collections,statistics,sys,time
from pathlib import Path
import numpy as np
P=Path(sys.argv[1]);db=sqlite3.connect('file:'+str(P/'contention.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
uuid='d152c6a7-1dcf-0ff2-1802-80647045e446';gpu=dict(db.execute('select * from TARGET_INFO_GPU where uuid=?',(uuid,)).fetchone());assert gpu['busLocation']=='0000:81:00.0' and gpu['pwGpuId']==7
base=min(x[0] for x in db.execute('select distinct typeId from TARGET_INFO_GPU_METRICS'));typ=base+gpu['pwGpuId']
strings=dict(db.execute('select id,value from StringIds'))
nv=[dict(x) for x in db.execute('select start,end,text,globalTid from NVTX_EVENTS where end is not null')];its=sorted([r for r in nv if r['text'].startswith('iteration/')],key=lambda r:r['start']);assert len(its)==20
runtime={};bythread=collections.defaultdict(list)
for r in db.execute('select start,end,globalTid,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME'):
 r=dict(r);runtime[r['correlationId']]=r;bythread[r['globalTid']].append(r)
for t in bythread:bythread[t].sort(key=lambda r:r['start'])
starts={t:[r['start'] for r in rr] for t,rr in bythread.items()}
ks=[dict(x) for x in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL')];bycorr=collections.defaultdict(list)
for k in ks:bycorr[k['correlationId']].append(k)
assert len({k['globalPid'] for k in ks})==1

all_runtime=sorted(runtime.values(),key=lambda r:r['start'])
all_starts=[r['start'] for r in all_runtime]
def kernels(r):
 if r['text']=='guidance_backward_total':a=all_runtime;ss=all_starts
 else:a=bythread[r['globalTid']];ss=starts[r['globalTid']]
 out=[]
 for launch in a[bisect.bisect_left(ss,r['start']):bisect.bisect_right(ss,r['end'])]:out.extend(bycorr.get(launch['correlationId'],[]))
 return sorted(out,key=lambda k:k['start'])
def union(xs):
 out=[]
 for a,b in sorted(xs):
  if out and a<=out[-1][1]:out[-1][1]=max(b,out[-1][1])
  else:out.append([a,b])
 return out
def overlap(xs,a,b):return sum(max(0,min(b,y)-max(a,x)) for x,y in xs if x<b and y>a)
metrics=collections.defaultdict(lambda:[[],[]]);mids=[14,15,16,23,26,29,30]
cache=P/'gpu4_counter_series.npz'
print('Reading actual GPU4 metrics',typ,flush=True)
if cache.exists():
 c=np.load(cache)
 for mid in mids:metrics[mid]=[c['t_'+str(mid)].tolist(),c['v_'+str(mid)].tolist()]
else:
 for ts,mid,value in db.execute('select timestamp,metricId,value from GPU_METRICS where typeId=? and metricId in (14,15,16,23,26,29,30)',(typ,)):
  metrics[mid][0].append(ts);metrics[mid][1].append(value)
for mid,(ts,v) in list(metrics.items()):
 order=np.argsort(ts);metrics[mid]=(np.array(ts,dtype=np.int64)[order],np.array(v,dtype=float)[order])
if not cache.exists():np.savez_compressed(cache,**{name:values for mid,pair in metrics.items() for name,values in [('t_'+str(mid),pair[0]),('v_'+str(mid),pair[1])]})
names={x['metricId']:x['metricName'] for x in db.execute('select * from TARGET_INFO_GPU_METRICS where typeId=?',(typ,))}
def stats_windows(windows):
 out={}
 for mid,(ts,v) in metrics.items():
  slices=[]
  for a,b in union(windows):
   i=np.searchsorted(ts,a);j=np.searchsorted(ts,b);slices.append(v[i:j])
  z=np.concatenate(slices) if slices else np.array([]);z=z[(z>=0)&(z<=100)]
  if len(z):out[names[mid]]=dict(n=len(z),mean=float(z.mean()),p50=float(np.median(z)),p95=float(np.percentile(z,95)),fraction_ge80=float((z>=80).mean()))
 return out
stages={'video_total','video_cache_prefill','action_denoise_step','prefix_decode','current_jepa_encoder','future_jepa_encoder','guidance_forward_total','guidance_backward_total'}
rows=[];kernel_tags={}
for ii,it in enumerate(its):
 mode=it['text'].split('/')[-1];rr=sorted([r for r in nv if it['start']<=r['start']<=it['end'] and r['end']<=it['end'] and r['text'] in stages],key=lambda r:r['start']);counts=collections.Counter()
 for r in rr:
  counts[r['text']]+=1;kk=kernels(r)
  if not kk:continue
  assert len({k['streamId'] for k in kk})==1,(r['text'],{k['streamId'] for k in kk})
  row=dict(iteration=ii,mode=mode,stage=r['text'],step=counts[r['text']],host_start=r['start'],host_end=r['end'],gpu_start=kk[0]['start'],gpu_end=max(k['end'] for k in kk),host_ms=(r['end']-r['start'])/1e6,kernel_sum_ms=sum(k['end']-k['start'] for k in kk)/1e6,kernel_span_ms=(max(k['end'] for k in kk)-kk[0]['start'])/1e6,kernels=len(kk),stream=kk[0]['streamId'])
  row['metrics_span']=stats_windows([(row['gpu_start'],row['gpu_end'])]);row['metrics_busy']=stats_windows([(k['start'],k['end']) for k in kk]);row['_kernels']=kk;rows.append(row)
  if r['text']!='video_total':
   for k in kk:kernel_tags[k['correlationId']]=r['text']
print('Analyzing CUDA queue gaps',flush=True)
for row in rows:
 kk=row['_kernels'];other=[k for k in ks if k['streamId']!=row['stream'] and k['start']<row['gpu_end'] and k['end']>row['gpu_start']]
 other_union=union([(k['start'],k['end']) for k in other]);tags={tag:union([(k['start'],k['end']) for k in other if kernel_tags.get(k['correlationId'])==tag]) for tag in ['prefix_decode','current_jepa_encoder','future_jepa_encoder']}
 vals=collections.Counter();qwindows=[];hostwindows=[]
 for prev,nxt in zip(kk,kk[1:]):
  a=prev['end'];b=nxt['start']
  if b<=a:continue
  launch=runtime[nxt['correlationId']];q=max(a,launch['end']);hs=max(0,min(b,launch['start'])-a);api=max(0,min(b,launch['end'])-max(a,launch['start']))
  vals['gap_ms']+=(b-a)/1e6;vals['next_launch_not_entered_ms']+=hs/1e6;vals['launch_api_in_progress_ms']+=api/1e6
  if hs:hostwindows.append((a,a+hs))
  if q<b:
   vals['already_submitted_gap_ms']+=(b-q)/1e6;vals['queued_aux_overlap_ms']+=overlap(other_union,q,b)/1e6;qwindows.append((q,b))
   for tag,windows in tags.items():vals['queued_overlap_'+tag+'_ms']+=overlap(windows,q,b)/1e6
 row['gaps']=dict(vals);row['queued_metrics']=stats_windows(qwindows);row['host_gap_metrics']=stats_windows(hostwindows)
 del row['_kernels']
# Resource footprints, aggregated by exact kernel name and attributed component.
resources=collections.defaultdict(lambda:dict(calls=0,sum_ms=0,registers=set(),shared=set(),blocks=set(),threads=set()))
for k in ks:
 tag=kernel_tags.get(k['correlationId'],'other');name=strings[k['demangledName']];v=resources[(tag,name)];v['calls']+=1;v['sum_ms']+=(k['end']-k['start'])/1e6
 v['registers'].add(k['registersPerThread']);v['shared'].add(k['staticSharedMemory']+k['dynamicSharedMemory']);v['blocks'].add(k['gridX']*k['gridY']*k['gridZ']);v['threads'].add(k['blockX']*k['blockY']*k['blockZ'])
res=[]
for (tag,name),v in resources.items():res.append(dict(stage=tag,name=name,**{k:sorted(x) if isinstance(x,set) else x for k,x in v.items()}))
res.sort(key=lambda r:r['sum_ms'],reverse=True)
# Steady values are separate from profiled spans.
steady=[json.loads(l) for l in (P/'latency_samples.jsonl').read_text().splitlines()];ss={}
for mode in sorted({r['mode'] for r in steady}):
 rr=[r for r in steady if r['mode']==mode];act=[[i for i in r['intervals'] if i['name']=='action_denoise_step'] for r in rr]
 ss[mode]=dict(n=len(rr),wall_median_ms=statistics.median(r['wall_ms'] for r in rr),first8_sum_median_ms=statistics.median(sum(x['gpu_span_ms'] for x in a[:8]) for a in act),step_median_ms=[statistics.median(a[i]['gpu_span_ms'] for a in act) for i in range(10)])
# Stage group summary: medians across individual component calls, not pooled time.
groups=[]
for mode in sorted({r['mode'] for r in rows}):
 for stage in sorted(stages):
  rr=[r for r in rows if r['mode']==mode and r['stage']==stage]
  if not rr:continue
  groups.append(dict(mode=mode,stage=stage,n=len(rr),kernel_span_median_ms=statistics.median(r['kernel_span_ms'] for r in rr),kernel_sum_median_ms=statistics.median(r['kernel_sum_ms'] for r in rr),metrics_busy_mean_median={key:statistics.median(r['metrics_busy'][key]['mean'] for r in rr if key in r['metrics_busy']) for key in rr[0]['metrics_busy']}))
out=dict(gpu=gpu,metric_type_id=typ,metric_names=names,metric_samples={mid:len(v[0]) for mid,v in metrics.items()},steady=ss,stage_groups=groups,stage_rows=rows,resources=res)
# Export a compact representative trace for local plotting, excluding all other GPUs.
example=next(i for i,it in enumerate(its) if it['text'].endswith('prefill_first'))
it=its[example];series={}
for mid,(ts,v) in metrics.items():
 mask=(ts>=it['start'])&(ts<=it['end']);series['t_'+str(mid)]=(ts[mask]-it['start'])/1e6;series['v_'+str(mid)]=v[mask]
np.savez_compressed(P/'gpu4_example_metrics.npz',**series)
(P/'gpu4_example_timeline.json').write_text(json.dumps(dict(iteration=example,origin_ns=it['start'],end_ns=it['end'],rows=[r for r in rows if r['iteration']==example]),indent=2))
(P/'hardware_contention_analysis_20261028.json').write_text(json.dumps(out,indent=2))
for g in groups:
 if g['stage'] in ['prefix_decode','future_jepa_encoder','guidance_forward_total','guidance_backward_total','action_denoise_step']:print(json.dumps(g),flush=True)
print('Top Decode kernels')
for r in [r for r in res if r['stage']=='prefix_decode'][:6]:print(json.dumps(r),flush=True)
