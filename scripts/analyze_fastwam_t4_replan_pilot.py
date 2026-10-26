"""Descriptive paired results and cost-aware plots, no threshold fitting."""
import json
from pathlib import Path

def analyze(out,partial=False):
    rows=json.loads((out/'partial_results.json').read_text())
    windows=[json.loads(line) for line in (out/'windows.jsonl').read_text().splitlines()]
    groups={arm:[r for r in rows if r['arm']==arm] for arm in ('fixed8','adaptive48','random48','fixed4')}
    summary=dict(partial=partial,rollouts=rows,groups={})
    for arm,rs in groups.items():
        if not rs:continue
        steps=sum(r['control_steps'] for r in rs);valid=sum(r['valid_decisions'] for r in rs)
        summary['groups'][arm]=dict(n=len(rs),successes=sum(r['success'] for r in rs),wam_calls_per100=100*sum(r['wam_calls'] for r in rs)/steps,wall_s=sum(r['wall_s'] for r in rs),wam_s=sum(r['wam_s'] for r in rs),monitor_s=sum(r['monitor_s'] for r in rs),early_rate=sum(r['early'] for r in rs)/valid if valid else None)
    a=summary['groups'].get('adaptive48');r=summary['groups'].get('random48')
    matched=None
    if a and r and a['n']==r['n']==4:
        call_gap=abs(a['wam_calls_per100']/r['wam_calls_per100']-1)
        # Sequential GPU work durations include monitored encoder and policy forwards.
        time_gap=abs((a['wam_s']+a['monitor_s'])/(r['wam_s']+r['monitor_s'])-1)
        matched=max(call_gap,time_gap)<=.15
        summary.update(budget_matched=matched,call_rate_gap=call_gap,gpu_stage_time_gap=time_gap)
    verdict='当前小样本不确定'
    if matched is False:verdict='预算未匹配，无法隔离时机收益；当前小样本不确定'
    summary['verdict']=verdict
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    text=['# t+4 自适应 replan 试验',f"\n状态：{'进行中，以下为部分结果' if partial else '核心试验完成'}。",'\n|Task|B8|A48|R48|B4|','|---|---|---|---|---|']
    for tid in (151,242,559,590):
        cells=[]
        for arm in groups:
            x=next((x for x in groups[arm] if x['task_id']==tid),None)
            cells.append(('成功' if x['success'] else '未成功')+f"；{x['control_steps']}步 / {x['wam_calls']}次WAM" if x else '待完成/未追加')
        text.append('|'+str(tid)+'|'+'|'.join(cells)+'|')
    text+=['\n|组|成功|WAM/100步|监控秒|策略秒|墙钟秒|提前率|','|---|---|---|---|---|---|---|']
    for arm,g in summary['groups'].items():text.append(f"|{arm}|{g['successes']}/{g['n']}|{g['wam_calls_per100']:.2f}|{g['monitor_s']:.1f}|{g['wam_s']:.1f}|{g['wall_s']:.1f}|{g['early_rate']:.1%}|")
    text+=['\n判读：'+verdict+'。','\n固定失败筛选任务、seed47/init0；不代表跨任务泛化。GPU阶段时间由同步CUDA段墙钟统计，含拷贝/调度；不是CUDA kernel计时。墙钟不代表真实机器人deadline。单帧复制latent仅为描述性基线，与联合pair预测具有不同表示语义。','\n当前报告自动汇总数值；局部救回/损害的因果片段仍需人工核验，不能凭成功数宣称机制成立。']
    (out/'report_zh.md').write_text('\n'.join(text))
    if partial:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(4,1,figsize=(12,12))
    for ax,tid in zip(axs,(151,242,559,590)):
        for arm in ('fixed8','adaptive48','random48'):
            ws=[w for w in windows if w['task_id']==tid and w['arm']==arm and not w['censored']]
            ax.plot([w['control_end'] for w in ws],[w['error'] for w in ws],label=arm,alpha=.8)
            early=[w for w in ws if w['decision']=='replan4']
            ax.scatter([w['control_end'] for w in early],[w['error'] for w in early],s=12)
        tau=json.loads((out/'threshold.json').read_text())['tau'];ax.axhline(tau,color='black',ls='--');ax.set_title(f'Task {tid}');ax.set_xlabel('Control step');ax.legend()
    fig.tight_layout();fig.savefig(out/'error_and_replan_timeline.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for arm,g in summary['groups'].items():ax.scatter(g['wam_calls_per100'],g['successes']/g['n'],label=arm,s=70)
    ax.set_xlabel('WAM calls / 100 control steps');ax.set_ylabel('Success fraction');ax.legend();fig.tight_layout();fig.savefig(out/'quality_cost_comparison.png');plt.close(fig)
if __name__=='__main__':
    import sys
    analyze(Path(sys.argv[1]))
