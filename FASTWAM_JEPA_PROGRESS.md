# FastWAM–JEPA 混合闭环改造进度

> 更新时间：2026-09-08  
> 当前结论：闭环原型已跑通并完成第一轮 64 对 64 配对评测；总体提升较小（`54.7% -> 57.8%`），但不同物体上的效果方向明显不同。Salad Dressing 显著受益，Tomato Sauce 明显退化，下一阶段应优先补齐候选级日志并做严格消融。

## 1. 当前状态

| 模块 | 状态 | 说明 |
|---|---|---|
| FastWAM 一次生成 8 组动作候选 | 已完成 | 共享视觉、文本和 proprio 条件，在动作分支从 8 组独立高斯噪声开始采样 |
| V-JEPA2-AC 动作适配器 | 已完成 | 将 LIBERO 的 7D delta EEF 动作转换为 predictor 所需的 7D 状态/动作表示 |
| 两帧时序输入 | 已完成 | 当前 clip 使用 `[V-1, V0]`，目标 clips 使用 `[V0, V1]`、`[V1, V2]` |
| JEPA Best-of-8 排序闭环 | 已完成 | 每 8 个低层动作重新规划一次，执行 energy 最低候选的前 8 步 |
| 单 GPU worker 设备映射 | 已修复 | JEPA 不再硬编码 `cuda:1`，与 FastWAM 共用 worker 可见的本地 `cuda:0` |
| T5 提前编码并卸载 | 已完成 | 先缓存任务文本 embedding，再释放 T5，随后加载 FastWAM 和 JEPA |
| 主视角预处理对齐 | 已完成 | 当前帧与 FastWAM 预测帧都按 agent-view 区域处理；JEPA 暂不使用腕部视角 |
| 动作适配器与双帧单测 | 已通过 | `6/6`，使用 `unittest` 运行；当前 HMoE 环境未安装 `pytest` |
| 候选动作、energy、选择 ID 持久化 | 未完成 | 现有完整实验无法事后还原 JEPA 每次具体选择了哪个候选 |
| 严格 matched-control 消融 | 未完成 | 原始基线 replan=10，JEPA replan=8，当前总成绩还不是纯粹的 selector 因果比较 |

## 2. 当前闭环到底怎么运行

每次重规划时的帧和动作关系如下：

```text
真实观测历史             FastWAM 共同参考未来
V-1 ---- V0             V0 ---- V1 ---- V2 ---- ... ---- V8
  \______/​                   \___________/
  JEPA 当前双帧             当前只用前两个未来转移做排序

8 组候选动作：A0 ... A7，每组 32 个 delta-EEF 动作
                  |
                  | 每 4 个低层动作聚合为 1 个 AC predictor step
                  v
每个候选取前 8 步 -> 2 个 AC steps -> JEPA 预测两个未来 latent
                  |
                  v
与 [V0,V1]、[V1,V2] 的目标 latent 计算 energy
                  |
                  v
选择平均 energy 最低的候选，执行其前 8 步，然后重新规划
```

这里需要注意：

- FastWAM 一次输出完整的 9 帧序列，其中第一帧是当前画面 `V0`，后面是 8 个预测画面；相邻预测画面对应约 4 个低层动作。
- 当前 JEPA 排序只使用 `V0、V1、V2`，因为一次只执行 8 个低层动作，即两个 predictor steps；其余预测帧目前不进入排序目标。
- 8 个动作候选来自同一条件下的独立初始高斯噪声，经 10 步 flow-matching 推理得到。它不是手工八方向采样，也不是 CEM，而是 **random shooting / Best-of-N sample-and-rank receding horizon**。
- 8 个候选目前都与同一条、独立采样出的 FastWAM 未来参考比较，并没有为每个动作候选生成与其因果对应的未来视频。
- 排序指标不是 cosine similarity，而是对 layer-normalized latent 计算逐元素平均 L1 distance，再对两个预测时刻取平均；数值越低越好。
- FastWAM 与 JEPA predictor 的动作语义已经对齐为 7D EEF 表示：位置和旋转为 delta，夹爪按适配后的绝对/开合状态表达。它并非仅做一个统一的线性倍数缩放，还包含裁剪、尺度转换、旋转组合、低层动作聚合和状态积分。

主要代码位置：

- 闭环评测：[experiments/libero/eval_libero_single.py](experiments/libero/eval_libero_single.py)
- JEPA ranker 与动作适配：[experiments/libero/vjepa2_ac_ranker.py](experiments/libero/vjepa2_ac_ranker.py)
- 多 GPU worker：[experiments/libero/run_libero_manager.py](experiments/libero/run_libero_manager.py)
- FastWAM 多候选采样：[src/fastwam/models/wan22/fastwam.py](src/fastwam/models/wan22/fastwam.py)
- 配置：[configs/sim_libero.yaml](configs/sim_libero.yaml)
- 单测：[tests/test_vjepa2_ac_adapter.py](tests/test_vjepa2_ac_adapter.py)

## 3. 资源占用与部署方式

已经验证 T5 可以先完成文本编码后卸载，再加载 FastWAM–JEPA：

| 阶段 | 实测显存 |
|---|---:|
| T5 单独编码峰值 | 约 10.60 GiB |
| T5 卸载后，FastWAM + JEPA 已加载 | 约 17.45 GiB |
| 第一次 Best-of-8 推理峰值 | 约 18.56 GiB |

因此单卡可以运行，只是吞吐更低。双卡评测采用 GPU 1 和 GPU 3，每个 worker 只看见一张卡，并在其局部 `cuda:0` 上同时加载 FastWAM 和 JEPA；完整 64 次运行未出现 worker 崩溃或 OOM。

## 4. 第一轮配对实验

### 4.1 实验设置

- 数据：LIBERO-Plus 的 16 个 pose task ID。
- 每个 task 使用 seeds `7、17、27、37`，共 64 次 rollout。
- 原始 FastWAM：官方 `libero_uncond_2cam224.pt`，单候选，无 JEPA，replan=10。
- FastWAM–JEPA：Best-of-8，V-JEPA2-AC FP32，replan=8，保存 rollout 和 future-prediction 视频。
- 两组使用相同 task IDs 和 seeds，但 replan 间隔不同，因此结果是当前系统级对比，不应当表述为严格隔离 JEPA 排序贡献的消融。

### 4.2 总体结果

| 方法 | 成功数 | 成功率 |
|---|---:|---:|
| 原始 FastWAM | 35/64 | 54.7% |
| FastWAM–JEPA Best-of-8 | 37/64 | 57.8% |
| 差值 | +2/64 | +3.1 个百分点 |

配对变化：JEPA 挽救了 7 个原本失败的 rollout，同时使 5 个原本成功的 rollout 退化；其余 52 个结果不变。现阶段只能说 selector 改变了失败分布，不能宣称已经获得稳定的全局提升。

### 4.3 按难度汇总

| 难度 | 原始 | JEPA | 差值 |
|---|---:|---:|---:|
| add_10 control/reference | 16/16 | 16/16 | 0 |
| pose level 1 | 9/16 | 11/16 | +2 |
| pose level 3 | 9/16 | 7/16 | -2 |
| pose level 5 | 1/16 | 3/16 | +2 |

`Level 1/3/5` 是相对 reference pose 的目标物体位姿扰动等级，数字越大通常表示偏移范围/难度越高；这里的 control 是实验中配套的 reference/control task。

### 4.4 按目标物体汇总

| 目标物体 | 原始 | JEPA | 差值 |
|---|---:|---:|---:|
| Alphabet Soup | 9/16 | 10/16 | +1 |
| Milk | 4/16 | 4/16 | 0 |
| Salad Dressing | 9/16 | 14/16 | +5 |
| Tomato Sauce | 13/16 | 9/16 | -4 |

### 4.5 每个任务

| 目标 | 版本 | Task ID | 原始 | JEPA | 差值 |
|---|---|---:|---:|---:|---:|
| Alphabet Soup | control | 1818 | 4/4 | 4/4 | 0 |
| Alphabet Soup | level 1 | 1839 | 4/4 | 4/4 | 0 |
| Alphabet Soup | level 3 | 1847 | 1/4 | 2/4 | +1 |
| Alphabet Soup | level 5 | 1855 | 0/4 | 0/4 | 0 |
| Milk | control | 2037 | 4/4 | 4/4 | 0 |
| Milk | level 1 | 2062 | 0/4 | 0/4 | 0 |
| Milk | level 3 | 2070 | 0/4 | 0/4 | 0 |
| Milk | level 5 | 2078 | 0/4 | 0/4 | 0 |
| Salad Dressing | control | 2131 | 4/4 | 4/4 | 0 |
| Salad Dressing | level 1 | 2156 | 1/4 | 3/4 | +2 |
| Salad Dressing | level 3 | 2163 | 4/4 | 4/4 | 0 |
| Salad Dressing | level 5 | 2169 | 0/4 | 3/4 | +3 |
| Tomato Sauce | control | 2174 | 4/4 | 4/4 | 0 |
| Tomato Sauce | level 1 | 2203 | 4/4 | 4/4 | 0 |
| Tomato Sauce | level 3 | 2211 | 4/4 | 1/4 | -3 |
| Tomato Sauce | level 5 | 2217 | 1/4 | 0/4 | -1 |

## 5. 视频观察与当前解释

### Salad Dressing：主要正向案例

- 共出现 5 个 rescue、0 个 regression。
- Level 1 从 `1/4` 提高到 `3/4`，Level 5 从 `0/4` 提高到 `3/4`。
- 视频中白色瓶身和绿色瓶盖较显著，FastWAM 的短期预测通常能较稳定地保持物体身份以及抓取后朝篮筐运动的趋势，JEPA 因而能提供有用的相对排序信号。
- Level 5 唯一失败的 seed 7 更像是在篮筐附近悬停/停滞；短期视觉目标对于“放入、释放、是否已经进入容器”仍不够敏感。

### Tomato Sauce：主要负向案例

- Level 3 从 `4/4` 降到 `1/4`，Level 5 从 `1/4` 降到 `0/4`。
- 失败视频中可见 FastWAM 在接触阶段出现参考未来不一致：真实画面里夹爪未抓住或物体已倒下，预测中却继续把物体画成被抓住和移动，个别序列还会在腕部附近重新“生成”物体。
- JEPA 只负责寻找与该参考未来更接近的候选，所以错误参考可能被 Best-of-8 放大。这可称为 **reference-model bias amplification（参考模型偏差放大）**。
- 目前没有候选级 trace，所以上述是基于帧的强烈迹象，而不是已完成的逐候选因果证明。

### 关于“不同采样噪声影响很大”

现有结果确实表明 FastWAM 的动作生成对随机初始噪声较敏感，否则 8 个候选不会产生足够不同的轨迹供 JEPA 筛选。但 Salad 的提升和 Tomato 的退化不能只归因于噪声：最终结果同时由候选覆盖范围、FastWAM 参考未来质量、JEPA energy 是否对任务关键事件敏感，以及闭环中早期选择造成的轨迹分叉共同决定。

## 6. 当前已知问题

1. **缺少候选级原始数据。** 完成的实验保存了结果和视频，但没有持久化 8 组动作、每步/平均 energy、被选 candidate ID、第一二名 margin，因此无法精确回答某一步 JEPA 为什么选了该动作。
2. **参考未来与动作候选不成对。** 8 组动作都匹配同一条独立生成的未来，而不是“候选动作 -> 对应未来”的 action-conditioned rollout。
3. **参考模型会产生接触幻觉。** Tomato 的失败说明错误的抓取/物体运动参考可能引导 selector 选错。
4. **L1 latent energy 未必关注任务关键区域。** 大量静态背景可能掩盖抓取接触、物体是否在夹爪中、是否进入篮筐等小区域事件。
5. **基线协议未完全匹配。** 原始 FastWAM 使用 replan=10，JEPA 使用 replan=8；需要 matched control 排除重规划频率影响。
6. **跨重规划复用噪声模板。** 当前固定 cfg seed 会使每轮从同一 RNG 模板开始，虽然条件观测会变化，但不等同于每轮重新抽取独立噪声集合。
7. **样本量仍小。** 每个具体 task 只有 4 个 seed，物体级结论值得追踪，但尚不足以支持强统计结论。
8. **JEPA 仅使用主视角。** 腕部视角暂未加入，符合当前阶段决定；FastWAM 本身仍处理双相机画面。
9. **PSNR 不是可靠的语义指标。** Tomato 成功与失败之间的 PSNR 没有清晰分界，不能用它替代接触一致性或任务成功判断。

## 7. 下一步建议（按优先级）

### P0：先让选择过程可审计

每次 replan 写一条结构化 JSONL/NPZ，至少包含：task、seed、环境步、8 组动作、mean/per-step energy、selected ID、top-2 margin、当前双帧和参考未来帧路径。这样才能直接分析 Tomato 被带偏和 Salad 被挽救的具体选择过程。

### P1：完成 matched-control 消融

对重点任务 `2211`（Tomato Level 3）和 `2169`（Salad Level 5）先运行：

| 组别 | 用途 |
|---|---|
| K=1，replan=8 | 匹配重规划周期的单采样基线 |
| K=8，固定选 candidate 0 | 检查批采样实现是否改变单样本轨迹 |
| K=8，随机选 | 测量仅增加随机候选但不排序的效果 |
| K=8，JEPA-L1 | 当前方法 |
| K=8，JEPA-cosine（可选） | 比较距离度量，不预设 cosine 一定更好 |

### P2：改进排序可靠性

- 使用多条 FastWAM 参考未来做 ensemble，降低单条预测幻觉的影响。
- 当第一、第二候选 energy 差距过小时回退 candidate 0，避免无把握的改写。
- 尝试物体/夹爪 ROI、接触一致性或对象中心特征，减少背景对 L1 的支配。
- 在日志完整后，再评估是否需要扩大 K 或使用 CEM；当前 8 个候选已经足以暴露明显的选择差异，不必立刻增加计算量。

## 8. 论文层面的谨慎表述

当前结果更适合表述为：

> JEPA-guided stochastic action selection can improve closed-loop control when the imagined future preserves object identity and contact-consistent motion, but it may amplify world-model bias when the reference future hallucinates object contact.

中文：**当想象未来能够保持物体身份和接触运动的一致性时，JEPA 引导的随机动作选择可以改善闭环控制；当参考未来产生物体接触幻觉时，排序器也可能放大世界模型偏差。**

不建议现阶段使用“解决单次采样失效”或“整体显著提升”作为主要结论。

## 9. 数据、日志和视频位置

### 原始 FastWAM

```text
evaluate_results/fastwam_original_libero_plus_pose16_20260907_150235/
├── background.log
├── summary.md
├── summary.csv
└── seed_{7,17,27,37}/
```

### FastWAM–JEPA

```text
evaluate_results/fastwam_jepa_libero_plus_pose16_20260907_153555/
├── background.log
├── summary.md
├── summary.csv
├── comparison_with_original.md
├── comparison_with_original.csv
└── seed_{7,17,27,37}/
```

JEPA 目录中共保存 64 个 rollout 视频和 2,183 个 future-prediction 视频；完整目录约 121 MiB。原始基线目录约 26 MiB。

常用查看命令：

```bash
cd /data/chenpengxu/FastWAM

# 总体对比
less evaluate_results/fastwam_jepa_libero_plus_pose16_20260907_153555/comparison_with_original.md

# 找 Salad Dressing 的 rollout 视频
find evaluate_results/fastwam_jepa_libero_plus_pose16_20260907_153555 \
  -type f \( -iname '*2156*.mp4' -o -iname '*2163*.mp4' -o -iname '*2169*.mp4' \) | sort

# 找 Tomato Sauce Level 3 的 rollout / future 视频
find evaluate_results/fastwam_jepa_libero_plus_pose16_20260907_153555 \
  -type f -iname '*2211*.mp4' | sort

# 运行已有单测（HMoE 环境没有 pytest，直接用 unittest 入口）
PYTHONPATH=. /data/chenpengxu/conda_envs/HMoE/bin/python tests/test_vjepa2_ac_adapter.py
```

## 10. 工作区说明

上述 JEPA 改造目前仍是本地未提交工作，涉及配置、评测入口、manager、FastWAM 多候选采样、JEPA ranker 和测试文件。后续提交前应先补候选级 trace，再将“功能改造”和“实验/部署脚本”分开整理，便于复现与审查。
