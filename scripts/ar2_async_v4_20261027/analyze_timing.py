"""Audit fixed-input measurements and plot measured CUDA-event intervals."""
import json,hashlib,datetime,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/'outputs/ar2_async_timing_v4_20261027'
def read(name):return json.loads((OUT/name).read_text())
rows=[json.loads(s) for s in (OUT/'latency_samples.jsonl').read_text().splitlines()]
summary=read('latency_summary.json');modes=summary['variants'];audit={}
for name,expected in read('manifest.json')['source_sha256'].items():
 assert hashlib.sha256((OUT/'source_snapshot'/name).read_bytes()).hexdigest()==expected,name
def one(row,name):return next(x for x in row['intervals'] if x['name']==name)
def med(xs):return float(np.median(xs))
for mode in modes:
 rs=[r for r in rows if r['mode']==mode];assert len(rs)==20
 for row in rs:
  for name,count in [('video_denoise_step',10),('action_denoise_step',10)]:
   assert sum(x['name']==name for x in row['intervals'])==count,(mode,name)
  if mode!='no_guidance':
   for name in ['guidance_forward_total','guidance_backward_total','guidance_verification_total','scheduler_guidance_update']:
    assert sum(x['name']==name for x in row['intervals'])==3,(mode,name)
  assert np.isfinite(row['wall_ms'])
 result=dict(n=len(rs),wall_median_ms=med([r['wall_ms'] for r in rs]),
  video_action_gap_ms=med([one(r,'action_total')['gpu_start_ms']-one(r,'video_total')['gpu_end_ms'] for r in rs]),
  peak_allocated_mib=max(r['peak_allocated_mib'] for r in rs),peak_reserved_mib=max(r['peak_reserved_mib'] for r in rs))
 if mode!='no_guidance':
  overlaps=[];starts=[]
  for r in rs:
   a=one(r,'action_total');d=one(r,'prefix_decode')
   overlaps.append(max(0,min(a['gpu_end_ms'],d['gpu_end_ms'])-max(a['gpu_start_ms'],d['gpu_start_ms'])))
   starts.append(one(r,'wam_input_vae_encode_action')['gpu_start_ms']-d['gpu_start_ms'])
  result.update(decode_action_overlap_ms=med(overlaps),action_vae_start_minus_decode_start_ms=med(starts))
 audit[mode]=result
assert read('compiler_counters_before_measure.json')==read('compiler_counters_after_measure.json'),'Compilation during measurement'
assert all(v['max_abs']==0 for v in read('scheduling_validation.json').values())
assert read('aux_gradient_validation.json')['gradient']['max_abs']<1e-5
assert read('aux_gradient_validation.json')['loss']['max_abs']==0
assert read('aux_gradient_validation.json')['gripper_gradient_max']==0
pid=read('active_process.json')['pid'];foreign={}
measure_start=(OUT/'compiler_counters_before_measure.json').stat().st_mtime
measure_end=(OUT/'compiler_counters_after_measure.json').stat().st_mtime
for line in (OUT/'gpu_resource_samples.jsonl').read_text().splitlines():
 x=json.loads(line)
 if 'state' not in x:continue
 stamp=datetime.datetime.strptime(x['time'],'%Y-%m-%d %H:%M:%S').replace(tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
 if not measure_start<=stamp<=measure_end:continue
 uuid=x['state'].split(',')[0].strip()
 for p in x.get('processes','').splitlines():
  fields=[v.strip() for v in p.split(',')]
  if len(fields)==3 and fields[0]==uuid and int(fields[1])!=pid:
   foreign.setdefault(fields[1],set()).add(fields[2])
audit['_checks']=dict(no_steady_compilation=True,all_guided_action_max_abs=0,aux_gradient_max_abs=read('aux_gradient_validation.json')['gradient']['max_abs'],
 foreign_gpu_processes={k:sorted(v) for k,v in foreign.items()},worker_pid=pid)
(OUT/'timing_audit.json').write_text(json.dumps(audit,indent=2))

selected=['no_guidance','reference_joint_late_main','split_early_aux_prepared','split_early_aux_ready']
labels=['No guidance','Late setup / joint encoder','Guidance setup prepared','Decode submitted after native action inputs ready']
fig,axes=plt.subplots(4,1,figsize=(13,8),sharex=True,layout='constrained')
colors={'video_total':'#a9bfd9','action_total':'#efe0a7','current_jepa_encoder':'#91c4b0','prefix_decode':'#83bbce','future_jepa_encoder':'#c8aa8c','joint_jepa_encoder':'#b3b994','guidance_forward_total':'#b4a0c6','guidance_backward_total':'#85709d','guidance_verification_total':'#cfc3da','guidance_setup':'#888888'}
for ax,mode,label in zip(axes,selected,labels):
 rs=[r for r in rows if r['mode']==mode];target=med([r['wall_ms'] for r in rs]);r=min(rs,key=lambda r:abs(r['wall_ms']-target))
 ax.set_title(f'{label}  |  median {target:.1f} ms',loc='left',fontsize=11)
 for x in r['intervals']:
  name=x['name']
  if name not in colors:continue
  start=x.get('gpu_start_ms');end=x.get('gpu_end_ms')
  if start is None:continue
  lane=0 if name in ['video_total','action_total','guidance_setup'] else 1
  if name.startswith('guidance_') and name!='guidance_setup' and mode in ['reference_joint_late_main','split_early_main','split_late_main']:lane=0
  ax.broken_barh([(start,end-start)],(lane-.27,.54),facecolors=colors[name],edgecolors='white',linewidth=.4,zorder=3 if name.startswith('guidance_') else 2)
  text={'video_total':'Video','action_total':'Action' if mode=='no_guidance' else 'Action + guidance','current_jepa_encoder':'Current','prefix_decode':'Decode','future_jepa_encoder':'Future','joint_jepa_encoder':'JEPA'}.get(name)
  if text and end-start>27:ax.text((start+end)/2,lane,text,ha='center',va='center',fontsize=8)
 ax.set_yticks([0,1],['Main','Aux']);ax.set_ylim(1.55,-.55);ax.grid(axis='x',color='.9',zorder=0)
 ax.axvline(r['wall_ms'],color='.35',lw=.8,ls='--')
 first_action=one(r,'wam_input_vae_encode_action')['gpu_start_ms']
 ax.plot([first_action],[.4],marker='^',color='#444444',ms=5)
 ax.spines[['top','right','left']].set_visible(False)
axes[-1].set_xlabel('Elapsed time (ms); representative iteration for each mode')
axes[0].legend(handles=[Patch(color=colors[k],label=t) for k,t in [('current_jepa_encoder','Current JEPA'),('prefix_decode','Decode'),('future_jepa_encoder','Future JEPA'),('guidance_setup','Guidance setup'),('guidance_forward_total','Gradient forward'),('guidance_backward_total','Backward'),('guidance_verification_total','Loss check')]],loc='lower left',ncol=4,frameon=False,fontsize=8,title='Legend',title_fontsize=8)
fig.savefig(OUT/'measured_pipeline.png',dpi=180);fig.savefig(OUT/'measured_pipeline.pdf')
print(json.dumps(audit,indent=2))

names={'no_guidance':'无 guidance','reference_joint_late_main':'排查对照：晚准备、合并编码',
 'split_early_aux_prepared':'只前移 guidance 准备','split_early_aux_ready':'最终方案：输入准备后并发 Decode / Action'}
table='\n'.join(f"| {names[m]} | {v['n']} | {v['median_ms']:.2f} | {v['p95_ms']:.2f} |" for m,v in modes.items())
requested=summary['requested_mode'];v=modes[requested];a=audit[requested]
stage_names=[('guidance_setup','guidance 准备','host'),('current_cpu_preprocess','当前图 CPU 处理','host'),
 ('current_jepa_encoder','当前图 JEPA 编码','gpu'),('video_total','Video Gen 总跨度','gpu'),
 ('prefix_decode','前缀 VAE Decode','gpu'),('future_d2h','未来图 .cpu() 调用，含等待','host'),
 ('future_cpu_preprocess','未来图 PIL 处理','host'),('future_jepa_encoder','未来图 JEPA 编码','gpu'),
 ('wam_input_vae_encode_action','动作侧输入 VAE 编码','gpu'),('video_cache_prefill','视频 cache prefill','gpu'),
 ('action_denoise_step','10 次动作去噪合计','gpu'),('guidance_forward_total','3 次 guidance 梯度前向合计','gpu'),
 ('guidance_backward_total','3 次 guidance 反向合计','gpu'),('guidance_verification_total','3 次下降核对前向合计','gpu'),
 ('action_total','Action Gen 总跨度（含等待）','gpu')]
stages='\n'.join(f"| {label} | {v['gpu_span_median_ms' if kind=='gpu' else 'host_call_median_ms'][key]:.2f} | {'CUDA 事件跨度' if kind=='gpu' else 'host 调用跨度'} |" for key,label,kind in stage_names)
g=v['gpu_span_median_ms'];base=modes['no_guidance']['gpu_span_median_ms']
guidance_sum=sum(g[k] for k in ['guidance_forward_total','guidance_backward_total','guidance_verification_total'])
foreign=audit['_checks']['foreign_gpu_processes']
report=f'''# 新编排逐组件延迟与阻塞排查（2026-10-27）

最终测试已完成：{read('launcher_completed.json')['completed']}；GPU {read('launcher_completed.json')['gpu']}（NVIDIA GeForce RTX 4090 D）；4 组各 20 次，共 80 次稳态采样。另有 2 次 profiler 调用，不计入稳态结果。

## 同轮结果

| 方案 | 次数 | 中位数 ms | P95 ms |
|---|---:|---:|---:|
{table}

“排查对照”保留了本轮发现的晚准备/拷贝阻塞，用于定位问题。它不是此前 1027ms 测试的一比一复现，不能据此宣称相对原系统的总体提速。此前 837ms 的完整配置未恢复，本轮不能直接与其比较。

## 实际编排

1. 在完整计时范围内先构造 guidance，避免其同步 H2D 拷贝挡住后续并发。
2. 主流生成视频；辅助线程处理并编码当前图像。
3. 主流沿用原生随机动作生成和输入拷贝；在动作侧输入 VAE 编码入口提交辅助 Decode，随后两条流执行实际计算。
4. 辅助流继续未来图转换、CPU/PIL 处理、未来 JEPA 编码。
5. 去噪第 7/8/9 节点上，辅助流完成 Predictor 前向/反向；主流等待梯度并修正，之后继续去噪。下降核对仍开启。

图里的 Predictor 若画成“未来 JEPA 后紧接一整块”，实际还有动作依赖：必须等相应去噪节点的动作就绪，所以时间线上是后段三组前向/反向/核对，不能在未来编码完成后提前一次做完。

最终方案 Decode 与 Action 阶段重叠中位数 {a['decode_action_overlap_ms']:.2f}ms；动作侧输入 VAE 起点相对 Decode 起点为 {a['action_vae_start_minus_decode_start_ms']:.2f}ms。后者用于辨别“进入 Action 函数”与“实际动作计算开始”。

![逐次测量时间线](measured_pipeline.png)

图用各模式中总耗时最接近中位数的一次真实迭代绘制，不能把不同组件的中位数拼成一条时间线。三角标记表示动作侧输入 VAE 开始。Action 色带包含初始化和等待，不表示全程 GPU 忙碌。

## 最终方案的组件时间

| 组件 | 中位数 ms | 口径 |
|---|---:|---|
{stages}

父子区间与两条流存在重叠，表中数字不能相加。CUDA 事件跨度包含排队、同步和 host 发射空档，并非纯内核算时；host 调用跨度也包含等待。`.cpu()` 的 host 时间包含等待先前 Decode 完成，不能把它全部解释成 PCIe 传输。Predictor 前向包括 AR2 的 P1、P2 和加权 loss，未进一步分拆两次预测。

## 瓶颈判断

1. 三个节点的梯度前向、反向、下降核对共约 {guidance_sum:.2f}ms；主流必须等待其结果，放到另一条流也不能越过这条依赖。当前显式梯度等待跨度为 {g['guidance_aux_dependency_wait']:.2f}ms，和上述合计接近，属于主路径开销。
2. 真正并发后，视频 cache prefill 从无 guidance 的 {base['video_cache_prefill']:.2f}ms 增到 {g['video_cache_prefill']:.2f}ms。它与 Decode、未来 JEPA 编码重叠，是剩余最明确的并发慢化位置。其 host 调用也约 {v['host_call_median_ms']['video_cache_prefill']:.2f}ms；现有数据不能把全部慢化归结为 GPU 算力或显存带宽，host 发射、同步也在其中。
3. 提前当前图编码时，Video Gen 从 {base['video_total']:.2f}ms 增到 {g['video_total']:.2f}ms。它有重叠收益，也有约 {g['video_total']-base['video_total']:.2f}ms 的视频阶段慢化。第一个 guidance 的编码就绪等待只有 {g['encoder_dependency_wait']:.4f}ms，编码未就绪不是这里的主要阻塞。
4. 最终 Decode 与动作侧 VAE 已实际重叠，但不能把整段 Decode 耗时直接从总延迟扣掉：被重叠的其他阶段也变慢了。后续若优化，应优先针对 prefill 与解码/图像拷贝的交互，以及 guidance 的串行前后向开销，而非只继续增加 stream。

## 已核实与限制

- 固定 task559、seed7、同一观测、同一份 T5、同一白杯任务权重。WAM BF16，JEPA/Predictor FP32；guidance 节点 7/8/9、scale4、RMS 上限0.02、下降核对开启。
- 原生首轮执行命令与历史记录完全一致；不同编排的热身动作最大差为 0。
- 辅助流梯度最大差 {audit['_checks']['aux_gradient_max_abs']:.12g}，夹爪梯度为 0；校验阈值沿用 1e-5。
- 正式测量前后全部编译计数一致，{read('compiler_counters_after_measure.json')['stats']['unique_graphs']} 个 unique graphs 无新增。热身补编译不计入稳态。
- 全部源码快照 SHA256 核对通过；逐次阶段数量核对通过。
- 活跃张量峰值 {a['peak_allocated_mib']:.2f}MiB，缓存保留峰值 {a['peak_reserved_mib']:.2f}MiB。
- 正式计时区间资源采样中的同卡其他进程：{json.dumps(foreign,ensure_ascii=False)}。采样间隔5秒，不能证明每个时刻独占，也不能单凭常驻显存判断对方计算强度。测试结束后该进程增至23346MiB，不属于本轮测量区间。
- 测量覆盖模型流水线及 JEPA 图像处理，包含 guidance 准备；不覆盖传感器、最外层 WAM 观测处理、环境执行或录像。这不是任务成功率实验；末尾原生评估器的 `0/1` 是 benchmark 提前返回的占位输出。

原始数据：`latency_samples.jsonl`；汇总：`latency_summary.json`；审计：`timing_audit.json`；编排/梯度校验：`scheduling_validation.json` 与 `aux_gradient_validation.json`；GPU/进程记录：`gpu_resource_samples.jsonl`；完整 trace 位于同名服务器目录的 `requested_async_trace.json`。
'''
(OUT/'结果报告_20261027.md').write_text(report)
