# PACT-WAM 实验与诊断｜2026-11-23

**先修 AR2 Predictor 的训练时序，再复测 guidance。** 真实示教目标下，小步修正改善了 8 步内的局部跟随；用 WAM 生成目标做的小规模 OOD 闭环没有提高成功率。日志里已经确认训练动作和观测错一拍，但还没有修复后训练结果，不能把所有问题归到这个错位。

English summary: A final-action 0.25× update improved short-horizon tracking under real demonstration targets, but produced no success-rate change on 14 matched OOD episodes. AR2 underestimates executed changes in the future latent and was trained with a confirmed one-step action/observation offset. Fix the alignment first, then test its impact on guidance and the WAM target separately.

**归档范围：**本次从服务器读取时核对了完成状态和结果，但批量传输中断。此版本保存已完整传回的原始日志及本地备份；尚未传完的 600/100 组和闭环原始日志，用[已核对的汇总及闭环逐对结果](server_verified_summary.json)标明，未冒充原始日志。

## 闭环与局部实验

| 实验 | 比较与结果 | 数据 |
|---|---|---|
| 干净指令 OOD，seed 42/43，3 个任务 | 每组 244 条。无 guidance **212/244**；旧 7/8/9、原始 MSE 梯度 1× **207/244**，救回 6、破坏 11。 | [逐任务、逐 seed 汇总](logs/evaluate_results/three_tasks_cleanprompt_42_43_20261111/final_results_20261113.json) · [动作、梯度和配置](logs/evaluate_results/three_tasks_cleanprompt_42_43_20261111/) |
| 输入梯度 OOD，seed 42/43，2 个任务 | **133/162**，与匹配的无 guidance **133/162** 持平，救回和破坏各 4；同一子集的旧 MSE 方案为 **128/162**。162 条含先导实验的 6 条复用记录。 | [配对报告](logs/evaluate_results/input_gradient_two_tasks_20261116/paired_report.json) |
| 真实示教，600 组 | 10 个任务、100 条示教、每条 3 个时间窗口、两个 seed。只执行 8 步。最终动作 0.25× 的第 8 步末端误差 **3.86 → 3.75 mm**；1× 和旧 7/8/9 分别为 **5.27、4.99 mm**。 | [汇总](server_verified_summary.json) |
| 修复回放后的真实示教，100 组 | 覆盖全部 10 个任务、50 个示教时刻 × seed 42/43。正向 0.25× 的实际画面误差在 **81/100** 组优于同幅反向；只看前期和中期 68 组，第 4、8 步末端误差分别下降 **7.8%、7.0%**，实际画面 JEPA MSE 下降 **4.7%**。放大到 1× 后平均变差。 | [清单](logs/evaluate_results/demo_sixarm_diag_20261118/manifest.json) · [汇总](server_verified_summary.json) |
| 最终动作 0.25× 的 OOD 闭环，seed 42/43，3 个任务 | **14 对**完整 rollout；无 guidance 和 0.25× 均为 **8/14**，救回和破坏均为 0。**808/809** 次更新降低模型目标 loss。成功组平均步数 **243.5 → 278.25**，主要由一个案例（256 → 564 步）拉高；配对变化中位数 **+1.5 步**。 | [已核对的逐对结果](server_verified_summary.json) |

最终动作 0.25×：Action Expert 完成 10 次去噪后，只对前 8 步的 6 维位姿指令做一次原始 MSE 梯度更新，夹爪不变。梯度通过 AR2 Predictor 回到动作，不穿过 Action Expert；没有后续去噪。另存了 [去噪节点诊断](logs/evaluate_results/guidance_branch_diag_20261117/)、[最终 1×](logs/evaluate_results/real_ab_final_eta1_20261117/)和[旧 7/8/9](logs/evaluate_results/real_ab_old789_20261117/)的记录。

真实示教组给 Action Expert 输入真实示教视频条件，guidance 也用相应的真实视觉目标；与 OOD 闭环不是同一批状态、也不是只换目标的消融。在相同 100 组里，示教回放修复使第 8 步的平均误差 **1.24 → 0.30 mm**，超过 1 mm 的从 **22 → 4** 组。修复包括读数时序对齐和恢复夹爪指令历史。剩余回放误差不能视为零；排除这 4 组不改变主要结论。[回放检查](local/demo_replay_check_20261118/results.json)

## WAM 和 Predictor 分别与真实执行对比

11 月 17 日，4 个场景/seed 组合 × 3 个重规划时刻，共 **12 个 OOD 状态**。从同一起点分别执行两个动作分支，各自取第 4、8 步实际画面并编码为 JEPA latent。WAM 目标在每个配对内固定。

| JEPA MSE | 无 guidance | 最终动作 0.25× | 含义 |
|---|---:|---:|---|
| 固定 WAM 未来 vs. 各分支真实执行 | 0.72976 | 0.73288 | 实际跟随没有改善；只有 4/12 组更靠近 WAM 目标 |
| AR2 的预测 vs. 对应动作的真实执行 | 0.37197 | 0.37222 | 两组各自的整体预测误差基本持平 |
| AR2 的预测 vs. 固定 WAM 未来 | 0.76275 | 0.75764 | 模型自己的目标 loss 在 12/12 组都下降 |

[聚合值及逐状态结果](logs/evaluate_results/real_ab_v2_20261117/aggregate.json)。第一行不能独自区分 WAM 画错、实际动作没跟上，还是两条可行动作路径不同。第二行比较的是绝对预测误差，不能验证将一个动作换成另一个时，AR2 有没有正确预测其后果。

## Predictor：已确认的问题与下一步

1. **动作响应偏弱。** 在同一状态分别试执行原动作和修正动作，不用 WAM 或示教目标打分，直接比较两种动作造成的 latent 差。第 8 步，正向 0.25× 的预测变化幅度 / 实际变化幅度的中位数为 **28%（示教前中期 68 组）**、**37%（OOD 12 组）**，方向余弦中位数约 **0.11、0.07**。这说明对扰动的预测不够准，不等于把倍率放大就会恢复正确方向。[逐对统计](logs/evaluate_results/latent_action_response_audit_20261118/paired_responses.json)
2. **已确认训练错一拍。** HDF5 的 `obs[t]` 是执行完 `actions[t]` 之后保存的观测。若目标是 `obs[t+4]`，需要 `actions[t+1:t+5]`，旧缓存训练实际使用 `actions[t:t+4]`。20 条原始示教的关节记录和 45 个训练缓存窗口支持这个结论。部署端 100 组的动作核对通过，未看到归一化或控制器裁剪抹掉小步差异。图像预处理和精度在训练和推理之间有差异，影响尚未隔离。[输入审计](logs/evaluate_results/latent_action_response_audit_20261118/action_audit.json) · [时序证据](logs/evaluate_results/latent_action_response_audit_20261118/training_frame_state_timing.json) · [缓存、限幅和图像审计](logs/evaluate_results/latent_action_response_audit_20261118/ar2_consistency_audit.json)
3. **先单独修时序。** 原始 10 epoch checkpoint 保留；34,995 个缓存窗口仍可复用。短训 A/B 从同一最终权重起跑，一组旧索引、一组修正索引；目前停在预检，**没有新训练权重或改善结论**。验收用未参与训练的实执行动作对，检查是否更准确地预测第 4、8 步 latent 的变化；整体验证 L1 只作辅助。通过后固定现有 0.25× 复测，再单独隔离 WAM 目标。

与 [Feedback World Model](https://arxiv.org/html/2605.15705#A10) 的共同点是利用动作条件下的未来预测指导动作。注入和反馈方式不同，不能排除本项目的更新幅度或注入位置也有问题。论文 Fast-WAM-Joint 的 **18.3%** 是固定执行轨迹上对预测作反馈校正后，Wan VAE latent 的预测 MSE 降幅；这里报告的 **7.0%** 是改变执行动作后，8 步末端跟随误差的降幅。二者不直接比较。

## 日志索引与重算

- [实验文件清单](log_manifest.json)列出归档路径、原始位置、大小和 SHA-256。优先打开各目录的 `manifest.json`、`aggregate.json`、`completed.json` 或 `summary.json`；`metrics.json` 为逐状态指标，`plans.jsonl`、`commands.jsonl`、`updates.jsonl` 等为逐次重规划记录。
- 大于 8 MiB 的 JSONL 保留完整内容并压缩为 `*.jsonl.gz`，可用 `gzip -dc 文件.jsonl.gz` 阅读。公开归档替换了机器上的用户路径和 GPU UUID；原运行配置仍可查看，路径需要在本机重新配置。模型权重、视频、原始图像和 latent 张量不在 Git 中。失败尝试和重跑的记录可用于排查，不能作为额外独立样本计入结果。
- `python3 reports/research_20261123/verify.py` 从已归档的日志和汇总重算主要指标并检查清单。加 `--deep` 可逐行解析所有 JSONL。另有 [11 月 13 日动作与梯度诊断](local/guidance_debug_20261113/) 和 [WAM 画面人工检查记录](local/horizon_quality_cases_20261111/)。更早的结果见 [2026-11-02 报告](../research_20261102/README.md)。
