# PACT-WAM

## Latest results (2026-11-02)

The seven-task, 271-version robot-initial-pose study completed all 813 episodes. Results: **224/271 no guidance**, **221/271 unweighted guidance**, **220/271 weighted guidance**. Weighted guidance rescued 11 failures and introduced 15 failures. The earlier two positive tasks do not establish broad improvement. [Complete results and scope](reports/research_20261102/README.md).

All ten base-task calibration tables are available. The remaining four missing evaluation groups were queued (158 episodes); their results are pending. [Predictor hosting on Hugging Face](docs/PREDICTOR_HOSTING.md) includes an exact-checkpoint upload helper with private visibility by default.

## 最新实验：42个机器人初始状态扰动

任务是 LIBERO-Plus `libero_10` 中的 **“把白杯放到盘子上，把巧克力布丁放到盘子右边”**。测试覆盖该任务官方 **Robot Initial States 类别的全部42个版本**，不是42个不同基础任务，也不是只挑失败案例。

| 方案 | 成功 / 总数 | 成功率 | 相对无 guidance |
|---|---:|---:|---:|
| 无 guidance，固定执行8步 | 30 / 42 | 71.43% | — |
| 无加权 guidance，所有特征权重为1 | 30 / 42 | 71.43% | 0个百分点 |
| **离线加权 guidance** | **34 / 42** | **80.95%** | **+9.52个百分点** |

三组共126次正式 episode。固定 `seed=7`、每版本1次、`trial=0`、初始状态索引0；运行时 task ID 为525–566，对应官方分类ID526–567。每个场景在各组复用相同的 T5 tensor、初始状态和观测，检查未加 guidance 的首32步计划一致，再运行各自方法。

加权相对无 guidance 救回7个场景，同时损失3个，净增4个；相对无加权 guidance 救回6个、损失2个，也是净增4个。**这支持“离线加权在当前任务和扰动集合中优于统一加权”**，并不表示每个场景都改善，也不证明对其他任务、seed或扰动类型普遍有效。无加权与无 guidance 总成功数相同，但成功场景不完全相同。

- [逐场景三组结果](reports/research_20261026/robot42/unweighted_20261026/three_arm_summary.json)
- [原始正式结果与配置](reports/research_20261026/robot42/)
- [可重算结果的核对脚本](scripts/verify_robot42_report.py)
- [本次实验说明](docs/LATEST_ROBOT42_20261026.md)

## 当前 guidance 如何工作

```text
当前观测 + 固定T5上下文
          │
          ▼
       FastWAM ─────────────► 原始32步动作计划
          │                          │
          ▼                          ▼
   想象未来图像                  取前8步动作
          │                          │
          ▼                          ▼
  冻结JEPA编码 → 想象目标 W    AR2 Predictor → 预测未来 P
                   │                 │
                   └──── 加权 |P−W| ─┘
                             │
                        对位姿动作求梯度
                             │
                        有界修正并执行8步
```

AR2 分两段预测：前4步动作得到 `P1`，接着用 `P1` 和后4步动作得到 `P2`。当前保留跨段反向传播，没有对 `P1` 做 detach。JEPA 特征为 `B×576×1024`，分别对应576个空间 token、每个1024维。

### 离线权重

参考 FeedbackWM 的离线动作响应加权思路，保留本项目原有的 L1 loss 和 WAM 想象目标；**没有实现它的完整在线反馈或目标替换机制**。

1. 只从该基础任务45条训练示范中，选200个当前状态；不使用5条验证示范或测试 rollout 校准。
2. 固定每个状态，分别输入32组示范8步动作，计算 `P1` 和 `P2`。
3. 对每个特征元素计算“更换动作后预测变化多大”，再对200个状态求平均。
4. 每个预测时刻分别除以全部元素的均值，得到均值约1的固定权重。

```text
v[h,n,d] = mean_200_states(variance_32_actions(P[h,n,d]))
w[h,n,d] = v[h,n,d] / (mean_n,d(v[h,n,d]) + 1e-8)
L = mean_h,n,d(w[h,n,d] * abs(P[h,n,d] - W[h,n,d]))
```

权重形状为 **`[2,576,1024]`**：两个未来预测时刻，各 token、各通道分别加权；不是二值机械臂掩码，也不保证动作敏感的特征一定代表正确动作。无加权对照使用全1权重，其他 guidance 参数相同。

最新42场景复用白杯基础任务的**同一张离线权重表**，不对每个扰动重新调权。此前固定六案例集合 `6task_seed7_sameT5` 对应三个基础任务，因此保存了三张表；不能把“六案例”理解为六个不同基础任务。[六案例定义](docs/6task_seed7_sameT5.md)

### 固定参数与夹爪修复

| 项目 | 最新成功率实验的设置 |
|---|---|
| WAM | `libero_optional_idm_2cam224.pt`，冻结 |
| JEPA encoder | V-JEPA2.1 ViT-L，384px，FP32，冻结 |
| Predictor | AR2，epoch10 / step19690，FP32，冻结 |
| 每次计划 / 实际执行 | 32步 / 8步 |
| 去噪 / guidance节点 | 10步 / 7、8、9 |
| 梯度倍率 / 单次RMS上限 | 4 / 0.02，不归一化梯度 |
| loss核对 / 自适应触发 | 开启 / 关闭，每次replan启用guidance |
| video/action sigma shift | 1 / 1 |
| 最多控制步 / 开始等待步 | 700 / 30 |
| 离线权重 | beta=1，epsilon=1e-8；无加权组为全1 |
| 编译 | 本组126次成功率实验关闭编译；单独进行性能编译测试 |

LIBERO 路径中，Predictor 接收与执行端一致的二值夹爪命令；直接 guidance 对夹爪的梯度为0，只修正连续位姿动作。后续策略去噪或新的观测仍可能改变夹爪决定。运行日志保留夹爪输入与梯度核对记录。

## 代码与实验导航

| 入口 | 用途 |
|---|---|
| [`experiments/libero/eval_libero_single.py`](experiments/libero/eval_libero_single.py) | 原生闭环评测、实际动作后处理和夹爪修复 |
| [`experiments/libero/text_context_cache.py`](experiments/libero/text_context_cache.py) | 固定T5缓存及指纹核对 |
| [`scripts/fastwam_ar2_eval.py`](scripts/fastwam_ar2_eval.py) | 当前FP32 JEPA / AR2接口 |
| [`scripts/weighted_ar2_ranker.py`](scripts/weighted_ar2_ranker.py) | 逐元素加权L1 |
| [`scripts/calibrate_weighted_six.py`](scripts/calibrate_weighted_six.py) | 200状态×32动作离线校准 |
| [`scripts/launch_robot42.py`](scripts/launch_robot42.py)、[`scripts/eval_robot42.py`](scripts/eval_robot42.py) | 加权、无guidance两组正式评测与配对检查 |
| [`scripts/launch_unweighted42.py`](scripts/launch_unweighted42.py) | 补充全1权重guidance组；实际生成的运行源码保存在结果快照中 |
| [`research/runtime/`](research/runtime/) | AR2依赖的JointPredictor和向量化attention源码 |
| [`reports/research_20261026/robot42/`](reports/research_20261026/robot42/) | 126次正式结果、首计划核对、T5指纹、配置、loss/梯度结构化日志与源码快照 |
| [`reproducibility/robot42_20261026/`](reproducibility/robot42_20261026/) | 原样T5缓存、三张离线权重表及SHA256索引 |
| [`reports/research_20261026/archives/`](reports/research_20261026/archives/) | 历史实验配置、summary、训练/评测日志及运行源码的完整文本快照压缩包 |

运行源码快照是“当时实际执行了什么”的依据，`scripts/` 是当前整理后的入口。历史脚本继续保留，不能仅凭其文件名中的“latest/default”判断当前方案。

## 核对结果和准备复现

无需 GPU 即可从126份正式结果重算成功率，并检查预检记录及已发布T5/权重的SHA256：

```bash
python scripts/verify_robot42_report.py
```

归档索引位于 [`archive_index.json`](reports/research_20261026/archive_index.json)，每个包附逐文件SHA256和大小；源码、失败尝试及历史结果保留用于追溯。查看某个包：

```bash
tar -tzf reports/research_20261026/archives/evaluate_results__white_mug_robot42_seed7_sameT5_20261025.tar.gz
```

这是研究快照，**不是一键安装包**。原脚本含服务器路径 `/data/pxchen/PACT-WAM` 以及外部 LIBERO-Plus、JEPA_WAM / V-JEPA2 checkout。正式复现还需要匹配环境、WAM与Predictor checkpoint、LIBERO示范/状态资产，并按保存的 `command.json`、`runtime_config.json` 和源码快照设置路径。原启动器拒绝覆盖已有输出目录，不能在旧结果目录直接重跑。

- 42场景共享T5缓存SHA256：`a308f7ad80a0b385ee3f5460cbe296a0a4dba75418ad806f54d7ea0067e228a3`。
- 白杯权重表SHA256：`a539121cd6c4a5356523fd2b38330901ff177ae041d2beedbf601695ce1f0c6b`。
- Predictor checkpoint SHA256：`dec6eef937335a04c9fd8e23d7065163d2348919041e548749d8fe132313289e`。
- [发布资产相对路径与哈希](reproducibility/robot42_20261026/artifact_manifest.json)。固定T5必须原样复用，不能重新编码后假定完全相同。

上传包含代码、实验文本记录、T5和离线权重表；不包含大模型checkpoint、示范HDF5、token缓存、原始视频或编译二进制。其来源、配置与服务器位置保留在清单中。

## 独立性能测试与尚未完成的编排

2026-10-26在RTX4090D上，对task559/seed7固定输入完成当前模型编译，每方案预热后20次采样。它与上面的闭环成功率实验是**两套独立测量**。

| 已编译方案 | 单次计划中位延迟 |
|---|---:|
| 无guidance | 571.7ms |
| 完整解码、编码串行放到动作之前的消融基准 | 1152.6ms |
| 前缀解码＋串行 | 1084.2ms |
| 前缀解码＋异步 | 1027.0ms |

本次串行/并行动作完全一致，编译/未编译动作存在小幅差异（最大0.005859375）。没有用编译版重跑126个episode，因此不能声称编译后成功率不变。

**本次异步实现尚未完整复现旧837ms流程图：** 当前图像编码仍在视频生成、前缀解码之后，与未来图像一起编码；Predictor前向/反向在主流的guidance节点执行。当前图像编码与视频生成提前重叠仍待实现和验证。图中Predictor若迁到异步流，也必须保留对当前动作及梯度结果的同步依赖。

旧837ms的完整配置尚未恢复，不能把这次1027ms当作相同配置的直接回归；旧680ms是40%触发率的混合估算，并非本次实测。当前消融中，前缀解码省68.4ms，再并行省57.1ms；guidance前后向约340ms、视频生成约320ms，是明显开销。并行使JEPA编码约42.7→47.3ms，存在组件变慢，尚不能据此认定显存带宽饱和。性能测试时另有378MiB常驻GPU进程，非严格独占。

[性能原始结果、追踪索引与报告](reports/research_20261026/latency/)。完整163MB原始trace随对应文本日志压缩包保存；编译缓存留在服务器，不把二进制缓存当作可移植模型发布。

## 历史路线与结论边界

此前的latent-action IDM、residual Transformer、144场景guidance、六案例多seed、adaptive replan等实验均作为历史保留。当前README的主结论只使用上述42版本三组闭环结果，不将不同模型、旧坐标适配、旧T5或筛选子集的数字混在一起。

- [2026-10-06研究状态](docs/RESEARCH_STATE_20261006.md)
- [更早guidance历史说明](docs/HISTORICAL_GUIDANCE_README.md)
- [本次完整文本归档清单](reports/research_20261026/publication_manifest.json)

代码与第三方依赖遵循各自许可证；见 [`LICENSE`](LICENSE)。
