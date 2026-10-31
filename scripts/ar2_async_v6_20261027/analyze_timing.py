"""Audit the single-variable prefill-priority counterfactual."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/'outputs/ar2_async_timing_v6_20261027'
def read(name):return json.loads((OUT/name).read_text())
def med(xs):return float(np.median(xs))
rows=[json.loads(x) for x in (OUT/'latency_samples.jsonl').read_text().splitlines()]
s=read('latency_summary.json');modes=s['variants'];control='full_gpu_verify0_async';treatment='full_gpu_verify0_prefill_first'
assert read('progress.json')['state']=='COMPLETED'
assert read('compiler_counters_before_measure.json')==read('compiler_counters_after_measure.json')
assert read('prefill_priority_action_validation.json')['max_abs']<1e-5
for k,h in read('manifest.json')['source_sha256'].items():assert hashlib.sha256((OUT/'source_snapshot'/k).read_bytes()).hexdigest()==h,k
hashes=read('measured_action_hashes.json');assert hashes[control]==hashes[treatment] and len(hashes[control])==1
audit={}
for mode in modes:
 rs=[r for r in rows if r['mode']==mode];assert len(rs)==20
 for r in rs:
  assert sum(x['name']=='video_cache_prefill' for x in r['intervals'])==1
  assert sum(x['name']=='action_denoise_step' for x in r['intervals'])==10
  if mode!='no_guidance':
   assert not r['verify_descent'] and r['gpu_images'] and r['full_action_energy']
   for stage in ['guidance_forward_total','guidance_backward_total']:assert sum(x['name']==stage for x in r['intervals'])==3
   assert not any(x['name']=='guidance_verification_total' for x in r['intervals'])
 def one(r,name):return next(x for x in r['intervals'] if x['name']==name)
 if mode!='no_guidance':
  overlap=[];offset=[]
  for r in rs:
   p=one(r,'video_cache_prefill');d=one(r,'prefix_decode');overlap.append(max(0,min(p['gpu_end_ms'],d['gpu_end_ms'])-max(p['gpu_start_ms'],d['gpu_start_ms'])));offset.append(d['gpu_start_ms']-p['gpu_end_ms'])
   if mode==treatment:assert d['gpu_start_ms']>=p['gpu_end_ms']-.01,'Decode started before prefill completed'
  audit[mode]=dict(prefill_decode_overlap_median_ms=med(overlap),decode_start_minus_prefill_end_median_ms=med(offset),first_guidance_encoder_wait_median_ms=med([r['first_guidance_encoder_wait_ms'] for r in rs]),first_guidance_encoder_wait_p95_ms=float(np.percentile([r['first_guidance_encoder_wait_ms'] for r in rs],95)))
before=modes[control];after=modes[treatment];deltas=[r['wall_ms'] for r in rows if r['mode']==control];deltas=np.array(deltas)-np.array([r['wall_ms'] for r in rows if r['mode']==treatment]);rng=np.random.default_rng(0);boot=np.median(rng.choice(deltas,(5000,len(deltas)),replace=True),axis=1)
audit['_checks']=dict(passed=True,steady_samples=len(rows),all_measured_guided_actions_identical=True,no_recompilation=True,wall_median_saving_ms=before['median_ms']-after['median_ms'],paired_median_saving_ms=med(deltas),paired_bootstrap_95_ms=np.percentile(boot,[2.5,97.5]).tolist())
resource=[json.loads(x) for x in (OUT/'competition_samples.jsonl').read_text().splitlines()];loads={}
for mode in modes:
 rs=[r for r in rows if r['mode']==mode];samples=[x for x in resource if any(r['started_unix_s']<=x['unix_s']<=r['finished_unix_s'] for r in rs)];foreign={};util=[];clocks=[];errors=[]
 for x in samples:
  if 'error' in x:errors.append(x['error']);continue
  f=[v.strip() for v in x['gpu_state'].split(',')];util.append(float(f[4]));clocks.append(float(f[8]))
  for p in x.get('compute_processes','').splitlines():
   q=[a.strip() for a in p.split(',')]
   if len(q)==3 and q[0]==f[0] and int(q[1])!=x['worker_pid']:foreign.setdefault(q[1],set()).add(q[2])
 loads[mode]=dict(samples=len(samples),gpu_util_pct=float(np.mean(util)) if util else None,sm_clock_mhz=float(np.mean(clocks)) if clocks else None,other_processes={k:sorted(v) for k,v in foreign.items()},errors=errors)
audit['resources']=loads
(OUT/'prefill_priority_audit.json').write_text(json.dumps(audit,indent=2))

fig,axes=plt.subplots(2,1,figsize=(13,6),sharex=True,layout='constrained')
colors={'video_total':'#a9bfd9','action_total':'#efe0a7','current_jepa_encoder':'#91c4b0','prefix_decode':'#83bbce','future_jepa_encoder':'#c8aa8c','guidance_forward_total':'#b4a0c6','guidance_backward_total':'#85709d','video_cache_prefill':'#dc9c78'}
for ax,mode,label in zip(axes,[control,treatment],['Control: Decode released at action input VAE','Treatment: Decode released after prefill']):
 rs=[r for r in rows if r['mode']==mode];r=min(rs,key=lambda r:abs(r['wall_ms']-modes[mode]['median_ms']));ax.set_title(f'{label} | median {modes[mode]["median_ms"]:.1f} ms',loc='left',fontsize=11)
 for x in r['intervals']:
  n=x['name']
  if n not in colors or 'gpu_start_ms' not in x:continue
  lane=0 if n in ['video_total','action_total','video_cache_prefill'] else 1;a=x['gpu_start_ms'];b=x['gpu_end_ms'];ax.broken_barh([(a,b-a)],(lane-.25,.5),facecolors=colors[n],edgecolors='white',linewidth=.4,zorder=3 if n=='video_cache_prefill' else 2)
  label={'video_total':'Video','action_total':'Action + guidance','video_cache_prefill':'Prefill','prefix_decode':'Decode','future_jepa_encoder':'Future','current_jepa_encoder':'Current'}.get(n)
  if label and b-a>25:ax.text((a+b)/2,lane,label,ha='center',va='center',fontsize=8)
 ax.set_yticks([0,1],['Main','Aux']);ax.set_ylim(1.5,-.5);ax.grid(axis='x',alpha=.2);ax.axvline(r['wall_ms'],ls='--',lw=.8,color='.4');ax.spines[['top','right','left']].set_visible(False)
axes[-1].set_xlabel('Elapsed ms; representative real iterations; Action spans include waits')
fig.savefig(OUT/'prefill_priority_pipeline.png',dpi=180);fig.savefig(OUT/'prefill_priority_pipeline.pdf')

labels={'no_guidance':'无 guidance','full_gpu_verify0_async':'原并行：Decode 与动作输入编码同时启动','full_gpu_verify0_prefill_first':'prefill 优先：完成后才放行 Decode'}
table='\n'.join(f"| {labels[m]} | {v['median_ms']:.2f} | {v['p95_ms']:.2f} | {v['gpu_span_median_ms']['video_cache_prefill']:.2f} | {v['gpu_span_median_ms']['action_denoise_step']:.2f} | {v['first_guidance_encoder_wait_median_ms']:.4f} |" for m,v in modes.items())
profile_note='新旧两个guided模式均另做2次profiler，不计入正式采样。`profile_iterations.json`保存逐阶段CUDA/host区间，`requested_async_trace.json`保存在服务器，可用于检查GPU排队。'
report=f'''# prefill优先单变量对照（2026-10-27）

完成时间：{read('launcher_completed.json')['completed']}；GPU {read('launcher_completed.json')['gpu']}。三组各20次，共60次固定输入计划，无新增环境成功率rollout。

## 改了什么

保持task559、seed7、初始观测、T5、权重、FP32 JEPA/AR2、BF16 WAM、GPU未来图处理、整段action-to-energy编译、guidance节点7/8/9及关闭下降核对全部相同。当前JEPA启动时机也保持相同。

唯一调度改变：对照在动作侧输入VAE入口提交Decode；处理组在prefill返回后记录主流CUDA事件，辅助流显式等待该事件，才开始Decode。随后Decode/未来JEPA与动作去噪重叠。

## 结果

| 模式 | 总中位数 ms | P95 ms | prefill跨度 ms | 10步动作去噪合计 ms | 首次guidance特征未就绪等待 ms |
|---|---:|---:|---:|---:|---:|
{table}

本轮总中位数变化：{audit['_checks']['wall_median_saving_ms']:.2f}ms（正数为prefill优先更快）。逐轮配对差的中位数为{audit['_checks']['paired_median_saving_ms']:.2f}ms；探索性bootstrap 95%区间为{audit['_checks']['paired_bootstrap_95_ms']}ms。以同轮对照为准，不与上一轮933ms直接拼接计算收益。

动作去噪合计用于识别等待是否从prefill转移到去噪计算；首次guidance的等待用“编码就绪CUDA事件－该节点动作就绪CUDA事件”的正部测量，用来识别延迟是否转移到guidance。

![实测编排](prefill_priority_pipeline.png)

## 校验

- 每一条正式采样均保存完整原始动作SHA256；两个guided模式全部40次动作只有同一个哈希，调度未改变动作。
- prefill优先模式每次采样都验证Decode在prefill结束之后开始，允许计时分辨率0.01ms以内误差。
- 两个guided模式prefill/Decode重叠中位数分别为{audit[control]['prefill_decode_overlap_median_ms']:.4f}/{audit[treatment]['prefill_decode_overlap_median_ms']:.4f}ms。
- loss/梯度及2080个夹爪边界案例检查通过，夹爪梯度为0，正式采样没有重编译，源码SHA检查通过。
- GPU图像处理沿用上一轮方案，相对PIL的动作数值差此前已记录；这里比较的是相同GPU处理之间的调度，不把两者说成PIL逐元素等价。

## 竞争记录

按正式采样的Unix起止时间过滤资源日志，结果为：

```json
{json.dumps(loads,ensure_ascii=False,indent=2)}
```

原始日志还记录显存控制器利用率、功耗、频率、主/辅助CPU线程活动及上下文切换。约1秒采样可能混合相邻模式，仅作背景参考。CUDA跨度包含排队与host发射空档，不等于纯kernel忙碌时间，也不能单凭它认定SM/带宽饱和。

{profile_note}

## 既有trace的补充定位

上一轮trace里，约68–70ms的已提交prefill计算等待期间，至少约45ms与辅助流BF16卷积重叠。这些卷积属于Decode；其余包括矩阵乘、attention及其他操作，未在缺少明确阶段标记时全部硬归为JEPA。原记录保存在`previous_profile_gap_kernel_names.json`。

关键入口：`latency_summary.json`、`latency_samples.jsonl`、`prefill_priority_audit.json`、`measured_action_hashes.json`、`competition_samples.jsonl`和`source_snapshot/`。这仍是模型流水线latency测试，不含传感器、最外层WAM观测处理、仿真执行或录像；末尾评估器0/1不是任务成功率。
'''
(OUT/'prefill优先对照结果_20261027.md').write_text(report)
print(json.dumps(audit,indent=2))
