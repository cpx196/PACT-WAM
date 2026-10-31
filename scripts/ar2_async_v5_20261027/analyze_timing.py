"""Audit ablations and concurrency slowdowns without equating spans to busy time."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/'outputs/ar2_async_timing_v5_20261027'
def read(n):return json.loads((OUT/n).read_text())
def med(xs):return float(np.median(xs))
summary=read('latency_summary.json');variants=summary['variants'];rows=[json.loads(s) for s in (OUT/'latency_samples.jsonl').read_text().splitlines()]
assert read('progress.json')['state']=='COMPLETED'
assert read('compiler_counters_before_measure.json')==read('compiler_counters_after_measure.json')
for name,h in read('manifest.json')['source_sha256'].items():assert hashlib.sha256((OUT/'source_snapshot'/name).read_bytes()).hexdigest()==h,name
assert read('gripper_boundary_validation.json')['max_abs']==0
assert read('serial_async_action_validation.json')['max_abs']<1e-5
assert read('verify_flag_action_validation.json')['max_abs']<1e-5
for v in read('full_path_validation.json').values():
 assert v['gripper_gradient_max']==0
 assert all(v[k]['max_abs']<1e-5 for k in ['native_loss','native_gradient','eager_loss','eager_gradient'])
for r in rows:
 assert np.isfinite(r['wall_ms'])
 for name,count in [('video_denoise_step',10),('action_denoise_step',10)]:assert sum(x['name']==name for x in r['intervals'])==count
 if r['mode']!='no_guidance':
  for name,count in [('guidance_forward_total',3),('guidance_backward_total',3),('guidance_verification_total',3 if r['verify_descent'] else 0)]:assert sum(x['name']==name for x in r['intervals'])==count
 for name in ['future_d2h','future_cpu_preprocess','future_h2d']:
  if r['gpu_images']:assert not any(x['name']==name for x in r['intervals']),name
for mode,v in variants.items():assert sum(r['mode']==mode for r in rows)==20
serial=variants['full_gpu_verify0_serial'];parallel=variants['full_gpu_verify0_async']
names=sorted(set(serial['gpu_span_median_ms'])|set(parallel['gpu_span_median_ms']))
component={n:dict(serial_ms=serial['gpu_span_median_ms'].get(n,0),parallel_ms=parallel['gpu_span_median_ms'].get(n,0)) for n in names}
for n,v in component.items():v['ratio']=v['parallel_ms']/v['serial_ms'] if v['serial_ms'] else None
effects={
 'full_compile_and_remove_cpu_action_copy_ms':variants['partial_cpu_verify1']['median_ms']-variants['full_cpu_verify1']['median_ms'],
 'gpu_future_preprocess_ms':variants['full_cpu_verify1']['median_ms']-variants['full_gpu_verify1']['median_ms'],
 'disable_verification_ms':variants['full_gpu_verify1']['median_ms']-parallel['median_ms'],
 'serial_to_parallel_ms':serial['median_ms']-parallel['median_ms']}
resource=[json.loads(x) for x in (OUT/'competition_samples.jsonl').read_text().splitlines()];resource_by_mode={}
for mode in variants:
 rs=[r for r in rows if r['mode']==mode];samples=[x for x in resource if any(r['started_unix_s']<=x['unix_s']<=r['finished_unix_s'] for r in rs)];vals=[];foreign={};errors=[]
 for x in samples:
  if 'error' in x:errors.append(x['error']);continue
  try:
   fields=[v.strip() for v in x['gpu_state'].split(',')]
   values=dict(gpu_util_pct=float(fields[4]),memory_controller_util_pct=float(fields[5]),power_w=float(fields[6]),sm_clock_mhz=float(fields[8]))
   values['worker_cpu_pct']=x.get('worker_cpu',{}).get('cpu_pct')
   for role,t in x.get('thread_cpu',{}).items():values[role+'_cpu_pct']=t['cpu_pct']
   vals.append(values)
   for p in x.get('compute_processes','').splitlines():
    f=[v.strip() for v in p.split(',')]
    if len(f)==3 and f[0]==fields[0] and int(f[1])!=x['worker_pid']:foreign.setdefault(f[1],set()).add(f[2])
  except (ValueError,IndexError) as e:errors.append(str(e))
 keys=set().union(*(v.keys() for v in vals)) if vals else set()
 resource_by_mode[mode]=dict(samples=len(samples),mean={k:float(np.mean([v[k] for v in vals if v.get(k) is not None])) for k in keys if any(v.get(k) is not None for v in vals)},foreign_processes={k:sorted(v) for k,v in foreign.items()},errors=errors)
profile_stream=read('profile_stream_contention.json') if (OUT/'profile_stream_contention.json').exists() else []
audit=dict(passed=True,steady_samples=len(rows),no_recompilation=True,action_comparison=read('serial_async_action_validation.json'),verify_action_comparison=read('verify_flag_action_validation.json'),effects_ms=effects,component_serial_parallel=component,resources=resource_by_mode,profile_stream_contention=profile_stream)
(OUT/'timing_competition_audit.json').write_text(json.dumps(audit,indent=2))

chosen=['partial_cpu_verify1','full_gpu_verify0_serial','full_gpu_verify0_async'];labels=['Previous path / CPU images / verify on','Full compiled path / GPU images / verify off / serial','Full compiled path / GPU images / verify off / parallel']
fig,axes=plt.subplots(3,1,figsize=(13,7),sharex=True,layout='constrained')
colors={'video_total':'#a9bfd9','action_total':'#efe0a7','current_jepa_encoder':'#91c4b0','prefix_decode':'#83bbce','future_jepa_encoder':'#c8aa8c','guidance_forward_total':'#b4a0c6','guidance_backward_total':'#85709d','guidance_verification_total':'#cfc3da'}
for ax,mode,label in zip(axes,chosen,labels):
 rs=[r for r in rows if r['mode']==mode];r=min(rs,key=lambda r:abs(r['wall_ms']-variants[mode]['median_ms']))
 ax.set_title(f'{label} | median {variants[mode]["median_ms"]:.1f} ms',loc='left',fontsize=10)
 for x in r['intervals']:
  if x['name'] not in colors or 'gpu_start_ms' not in x:continue
  n=x['name'];start=x['gpu_start_ms'];end=x['gpu_end_ms'];lane=0 if n in ['video_total','action_total'] else 1
  ax.broken_barh([(start,end-start)],(lane-.26,.52),facecolors=colors[n],edgecolors='white',linewidth=.4)
  text={'video_total':'Video','action_total':'Action + guidance','prefix_decode':'Decode','current_jepa_encoder':'Current','future_jepa_encoder':'Future'}.get(n)
  if text and end-start>25:ax.text((start+end)/2,lane,text,ha='center',va='center',fontsize=8)
 ax.set_yticks([0,1],['Main','Aux']);ax.set_ylim(1.5,-.5);ax.grid(axis='x',alpha=.2);ax.axvline(r['wall_ms'],color='.4',ls='--',lw=.8);ax.spines[['top','right','left']].set_visible(False)
axes[-1].set_xlabel('Elapsed ms; actual iteration closest to each mode median. Spans include waits.')
fig.savefig(OUT/'pipeline_comparison.png',dpi=180);fig.savefig(OUT/'pipeline_comparison.pdf')
print(json.dumps(audit,indent=2))

labels_zh={'no_guidance':'无 guidance','partial_cpu_verify1':'原路径：CPU 图像、核对开启',
 'full_cpu_verify1':'整段编译：CPU 图像、核对开启','full_gpu_verify1':'整段编译：GPU 图像、核对开启',
 'full_gpu_verify0_serial':'新配置：核对关闭、串行','full_gpu_verify0_async':'新配置：核对关闭、并行'}
table='\n'.join(f"| {labels_zh[m]} | {v['n']} | {v['median_ms']:.2f} | {v['p95_ms']:.2f} |" for m,v in variants.items())
comp_keys=[('video_total','Video Gen'),('current_jepa_encoder','当前 JEPA'),('prefix_decode','前缀 Decode'),('future_jepa_encoder','未来 JEPA'),('video_cache_prefill','视频 cache prefill'),('action_denoise_step','10步动作去噪合计'),('guidance_forward_total','3次完整 guidance 前向合计'),('guidance_backward_total','3次反向合计')]
ctable='\n'.join(f"| {label} | {component[k]['serial_ms']:.2f} | {component[k]['parallel_ms']:.2f} | {component[k]['ratio']:.3f} |" for k,label in comp_keys)
resources='\n'.join(f"| {labels_zh[m]} | {v['samples']} | {v['mean'].get('gpu_util_pct',float('nan')):.1f} | {v['mean'].get('memory_controller_util_pct',float('nan')):.1f} | {v['mean'].get('worker_cpu_pct',float('nan')):.1f} | {json.dumps(v['foreign_processes'],ensure_ascii=False)} |" for m,v in resource_by_mode.items())
names=['video_total','prefix_decode','current_jepa_encoder','future_jepa_encoder','video_cache_prefill','action_denoise_step','guidance_forward_total','guidance_backward_total']
fig,ax=plt.subplots(figsize=(12,4),layout='constrained');x=np.arange(len(names));ax.bar(x-.19,[component[k]['serial_ms'] for k in names],.38,label='Serial');ax.bar(x+.19,[component[k]['parallel_ms'] for k in names],.38,label='Parallel');ax.set_xticks(x,['Video','Decode','Current\nJEPA','Future\nJEPA','Cache\nprefill','Action\ndenoise','Guidance\nforward','Guidance\nbackward']);ax.set_ylabel('CUDA-event span (ms)');ax.legend();ax.spines[['top','right']].set_visible(False);fig.savefig(OUT/'component_contention.png',dpi=180)
new=parallel;prep=read('gpu_preprocess_validation.json');boundary=read('gripper_boundary_validation.json')
profile_note=''
if profile_stream:
 profile_note=f"独立的两次profiler中，prefill kernel执行时间合计分别为 {profile_stream[0]['prefill_kernel_sum_ms']:.2f}/{profile_stream[1]['prefill_kernel_sum_ms']:.2f}ms，但从首kernel到末kernel跨度为 {profile_stream[0]['prefill_kernel_span_ms']:.2f}/{profile_stream[1]['prefill_kernel_span_ms']:.2f}ms。约 {profile_stream[0]['queued_gaps_covered_by_other_stream_ms']:.2f}/{profile_stream[1]['queued_gaps_covered_by_other_stream_ms']:.2f}ms 的间隔中，下一个prefill kernel已经由CPU提交，同时辅助流kernel在运行。这表明仅靠把CPU发射提前并不足以消除GPU流间等待。它支持内部调度/资源竞争的判断，仍不能单独区分SM、带宽或其他具体硬件限制。两次profiler不计入120次稳态样本；详细关联见 `profile_stream_contention.json`，不能拿profile的kernel耗时直接替换稳态计时。"
report=f'''# 整段编译修正与竞争记录（2026-10-27）

状态：完成，{read('launcher_completed.json')['completed']}，GPU {read('launcher_completed.json')['gpu']}。六组各20次，共120次稳态采样。模型仍为当前JEPA2.1 ViT-L384与AR2 step19690，未回退到旧vitg。

## 同轮结果

| 方案 | 次数 | 中位数 ms | P95 ms |
|---|---:|---:|---:|
{table}

## 修正范围

- 整段编译 clean action反归一化、GPU执行夹爪、RawAdapter状态接入、AR2 P1/P2及加权loss；其反向由AOTAutograd编译。不再在每次guidance时把动作拷回CPU获取夹爪。
- 全部核心模型计算路径继续编译：视频去噪、动作去噪、prefill、VAE输入编码和前缀Decode、当前/未来JEPA。
- 未来图像转uint8、bicubic resize到384、归一化与pair组合留在GPU上，并编译该处理函数；保留CPU/PIL处理的单独对照。
- 新配置关闭下降核对，以对齐找回的837ms实验。其余对照保留核对，用于量化开销。下降核对在原实现中仅记录诊断，不改变修正是否执行；本轮核对开关动作差为 {read('verify_flag_action_validation.json')['max_abs']:.12g}。
- 使用修复后的并发提交点：原生动作输入准备完成后，同时开始动作侧VAE与辅助Decode；当前JEPA与Video Gen重叠；辅助流Predictor根据7/8/9节点动作执行，主流保留梯度依赖。
- CUDA Graph仍关闭。旧日志明确显示其energy CUDA Graph因CPU tensor被跳过；不能把旧版更快归因于必然使用了CUDA Graph。外层Python调度、显式流依赖及安全检查仍有图外逻辑，不宣称整个控制流程成为一个完整CUDA Graph。

## 分步收益

以下为同轮中位数之差，正数表示后者更快。各处理差异已明确标注，不能机械相加，也不能全部解释为单一机制。

| 对照 | 减少 ms |
|---|---:|
| 原路径→整段编译并取消CPU动作往返，均CPU图像/核对开启 | {effects['full_compile_and_remove_cpu_action_copy_ms']:.2f} |
| CPU未来图→GPU未来图，均整段编译/核对开启 | {effects['gpu_future_preprocess_ms']:.2f} |
| 开启核对→关闭核对，均GPU未来图/整段编译/并行 | {effects['disable_verification_ms']:.2f} |
| 相同新配置串行→并行 | {effects['serial_to_parallel_ms']:.2f} |

GPU图像路径相对PIL参考的输入tensor最大差为 {prep['tensors']['max_abs']:.12g}，是否逐元素等价：{prep['exact_equivalence']}。其最终归一化动作相对PIL参考最大差为 {read('scheduling_validation.json')['full_gpu_verify0_async']['max_abs']:.12g}，RMS为 {read('scheduling_validation.json')['full_gpu_verify0_async']['rms']:.12g}。整段编译但仍用PIL时，动作相对原路径最大差为 {read('scheduling_validation.json')['full_cpu_verify1']['max_abs']:.12g}。GPU图像是显式处理变更，不把它混同为纯编译提速，也不据此证明原30/30/34成功率保持不变。串行/并行最终对照使用相同GPU处理，动作最大差 {read('serial_async_action_validation.json')['max_abs']:.12g}。

## 并发竞争：相同新配置的串行与并行

| 组件 | 串行 ms | 并行 ms | 并行/串行 |
|---|---:|---:|---:|
{ctable}

![组件慢化对照](component_contention.png)

这些是CUDA事件跨度，包含GPU排队、同步及主机发射空档，不等于纯kernel忙碌时间。prefill与Decode/编码重叠时的跨度变化能定位并发慢化位置，不能单凭它认定SM或DRAM带宽饱和。父子阶段与两条流重叠，表内数字不能相加得到总延迟。

{profile_note}

![实际时间线](pipeline_comparison.png)

每行采用最接近该模式中位数的一次真实迭代，不用不同组件的中位数拼图。Action色带含等待，紫色依次为guidance前向、反向及（如启用）核对。

## 外部负载与CPU活动

| 方案 | 资源采样数 | GPU利用率均值 % | 显存控制器利用率均值 % | 进程CPU均值 % | 同卡其他计算进程及显存 |
|---|---:|---:|---:|---:|---|
{resources}

资源采样按每次迭代的Unix起止时间归组，排除热身及进程退出后的其他任务。目标采样间隔1秒，实际间隔和采集成本逐条保留。设备利用率有自身统计窗口，短迭代间可能混合相邻模式，表内均值只作为背景负载参考。原始记录还含功耗、温度、SM/显存频率、主/辅助线程CPU、主线程上下文切换和主机负载。pmon若返回`-`则表示进程级利用率不可用，不当作0。设备利用率不等于SM占用率；显存控制器利用率不等于已经测得带宽吞吐量。CPU百分比按进程/线程采样计算，100%约为一个逻辑核。

## 校验与边界

- 同一task559、seed7、观测、T5及任务权重；原生历史首执行命令通过精确核对。
- {boundary['cases']}个GPU夹爪边界案例最大差{boundary['max_abs']}，实际guidance夹爪命令核对通过，夹爪梯度为0。
- 整段编译loss/梯度与相同特征下原生路径、eager完整路径比较，误差均小于1e-5，详见`full_path_validation.json`。
- 正式测量编译计数前后完全相同，无稳态重编译。源码快照SHA、逐组件次数均已检查。
- 本轮是固定输入latency测试，包含模型流水线和当前/未来JEPA处理；不含传感器、最外层WAM观测处理、仿真执行和录像。末尾评估器`0/1`是benchmark提前返回的占位，不是成功率结果。
- 旧837ms为task365连续rollout下的13轮、旧vitg模型、无核对、不同预处理和夹爪实现。本轮复用其优化方式，不宣称同模型同输入复现旧数值。

文件入口：`latency_samples.jsonl`逐次完整时间线；`competition_samples.jsonl`竞争原始记录；`timing_competition_audit.json`审计；`latency_summary.json`汇总；`source_snapshot/`运行源码。完整profiler另保存在服务器同名目录`requested_async_trace.json`，不计入120次采样。
'''
(OUT/'结果与竞争记录_20261027.md').write_text(report)
