# FastWAM × V-JEPA2-AC：按时间顺序的实验记录与结果

**整理日期：2026-09-17。** 本文汇总当前工作区内已完成的 LIBERO-Plus 闭环评测，以及影响方法选择的推理耗时和动作诊断实验。所有成功率都按“成功 episode 数 / episode 总数”计算；同一 task ID 的四个 seed 为 `7、17、27、37`。不同日期的任务集合、视频生成、重规划间隔或候选生成路径不一定相同，不能把各行成功率当作同一榜单直接排序。

## 1. 先看目前最有参考价值的结果

最新、规模最大的 Long Horizon 比较使用 **36 个固定 Plus ID × 4 个 seed = 每组 144 轮**：10 步 FastWAM 无 guidance 成功 **80/144（55.56%）**，10 步 FastWAM 加末三步 JEPA guidance 成功 **88/144（61.11%）**。逐 ID、逐 seed 配对后，guidance 有 **12 次 rescue、4 次 regression**，净增 8 次成功，即 **+5.56 个百分点**。两组都生成 10 步未来视频，动作也各做 10 步 flow denoise，`replan=8`。结果与日志见[本轮目录](../../evaluate_results/long36_ten_vs_789_20260916_231154_2gpu/)；状态为 `complete`。

这个增益并非均匀分布。“两个杯子分放左右盘”原始目标跨三个扰动类别从 **0/24 到 6/24**，贡献了净增 8 次中的 6 次。布局/干扰物类别为 **39/48→44/48**，机器人初始姿态为 **19/48→22/48**，背景纹理为 **22/48→22/48**。去掉纹理后，两组分别为 **58/96（60.42%）**和 **66/96（68.75%）**，差 **+8.33 个百分点**；这是看完结果后的子集分析，不宜当作预先定义的主要指标。

代价同样明显：本轮按全部 replan 加权，10 步对照为 **0.559 秒/次**，789 guidance 为 **2.390 秒/次**，约 **4.28 倍**。这一差额包含实验组额外的 `infer_action` 调用、JEPA encoder/predictor、动作适配与其他处理，不能全部解释为 JEPA 模型本身的耗时。完整运行从 **2026-09-16 23:12:08 到 2026-09-17 04:37:45**，GPU 0、1 双卡总墙钟约 **5 小时 26 分钟**。

## 2. 模型架构与两条方法路线

### 2.1 FastWAM 主干

本轮主要使用发布版 `libero_uncond_2cam224.pt` 和对应的 dataset statistics。输入包括两路 **224×224 相机图像**、T5 编码的任务文本、**8 维 proprioception**。FastWAM 基于 Wan2.2 TI2V-5B 组件；视频 DiT 为 **30 层、hidden size 3072**，动作 DiT 为 **30 层、hidden size 1024**。两路 token 在 joint core 内交互，视频和动作使用各自的 flow scheduler。一次输出 **32×7** 动作 chunk：前六维运动，末维夹爪。Long Horizon 评测通常只执行前 **8 个低层动作**后重新观测和规划；环境最多允许 700 个策略动作步。模型配置见[fastwam.yaml](../../configs/model/fastwam.yaml)，实际推理见[fastwam.py](../../src/fastwam/models/wan22/fastwam.py)。

`infer_joint` 生成未来视频，同时生成一条内部动作。无 JEPA 的未来视频对照直接执行这条 joint 动作。有 JEPA 且 `candidate_generation_mode=separate` 时，程序先调用 `infer_joint` 得到参考未来，再调用 `infer_action` 生成最终动作；前一次 joint 调用内产生的动作被丢弃。文本由单独 T5 预编码并缓存，释放 T5 后才加载 FastWAM/JEPA，多个 GPU worker 的 T5 加载依次错峰。

### 2.2 V-JEPA2-AC 与时间对齐

V-JEPA2-AC 使用冻结的 **ViT-G 视觉 encoder** 和 **action-conditioned predictor**，本地 checkpoint 为 `vjepa2-ac-vitg.pt`。模型权重不参与测试时更新。每 4 个 FastWAM 低层动作聚合成 1 个 JEPA AC step，因此执行前 8 步时可对齐两个视觉转移：

```text
当前视觉上下文：Encoder([上一真实观测, 当前真实观测])
未来目标 1：   Encoder([f0, f1])  ← 低层动作 0–3
未来目标 2：   Encoder([f1, f2])  ← 低层动作 4–7
E(a) = 0.5 × L1(JEPA_pred1(a[0:4]), target1)
     + 0.5 × L1(JEPA_pred2(a[0:8]), target2)
```

这里的 `f0、f1、f2` 来自 **FastWAM 自己预测的未来视频**，并非环境真实未来。LIBERO 的归一化 7 维动作先反归一化，再经[动作适配器](vjepa2_ac_ranker.py)转为 JEPA 使用的位姿/夹爪状态与动作；适配过程中有运动裁剪、旋转组合、四步聚合和状态积分。[视觉能量实现](vjepa2_ac_ranker.py)对两个预测 latent 与目标 latent 分别取平均 L1，再对时间步取平均。

### 2.3 两条 JEPA 使用路线

| 路线 | FastWAM 动作来源 | JEPA 如何影响输出 | 代表实验 |
|---|---|---|---|
| **Best-of-8 排序** | 从 8 份独立动作噪声批量生成 8 条动作，视频参考仍只有 1 条 | 给每条候选计算能量，选最低的一条；不修改候选内部动作 | 2026-09-07、09-09、09-10 |
| **动作 flow 梯度 guidance** | 单条动作轨迹；先单独生成参考视频，再生成执行动作 | 在指定 flow step 对 JEPA loss 反传并修正正在采样的动作 | 2026-09-15 至 09-17 |

目前的 789 guidance 在动作 flow 的索引 **7、8、9** 使用梯度。对当前带噪动作 `aσ` 与 FastWAM 速度预测 `v`，先形成干净动作估计 `â0 = aσ − σv`；对它计算 `g = ∂E/∂â0`。只保留前 8 步动作的梯度，按整体 RMS 归一化，然后执行 `a_next = aσ + Δσ·v − 0.02·g_normalized`。step 7、8 的修正还会影响后续 denoise；step 9 的修正直接影响最终动作。这里的 `0.02` 是归一化动作空间的一步修正幅度，**不是动作的 2%**。[guidance 构造](eval_libero_single.py)与[更新代码](../../src/fastwam/models/wan22/fastwam.py)记录了实际实现。FastWAM 速度预测在构造 `â0` 时被截断梯度，故这是针对当前干净动作估计的近似引导，不是对完整 FastWAM 采样链求精确梯度。

## 3. 实验时间线

下表按主要运行或结果目录的时间排序。`K=1` 指单动作候选；`K=8` 指八动作候选。表中“有效范围”与局限在后文逐轮说明。

| 时间 | 实验与任务 | 方法及关键区别 | 成功数 | 结果地位 |
|---|---|---|---:|---|
| 09-07 | LIBERO-Plus Object pose16，16 ID×4 seed | 原始 FastWAM 对比两调用 Best-of-8 JEPA | 35/64→37/64 | 早期系统对比；`replan=10` 对 `8` |
| 09-09 | Long core3：410、662、457 | 原始 action-only FastWAM | 1/12 | 非视频对齐参考 |
| 09-09 | 同一 core3 | 历史两调用 Best-of-8 | 4/12 | selector 方法初筛 |
| 09-09 | 同一 core3 | 未来视频对齐的 K=1、`replan=8` | 2/12 | 历史 Best-of-8 的匹配对照 |
| 09-09 | LIBERO Spatial task 0 延迟探针 | K=1、两调用 Best-of-8、joint K=8 的 warm replan | 498.0、965.3、860.2 ms | 仅耗时，不计任务成功率 |
| 09-10 | 同一 core3 | 一次 `infer_joint(K=8)` 替代两调用 | 1/12 | 加速但未保持成功率 |
| 09-15，早期 | `libero_object` 的 2169、2211 | `last_flow_steps` 探索扫描 | 完整 K=1 5/8、last2 5/8；last4 仅 3/4 | 扫描未完成，不外推 |
| 09-15，18 时 | Long core3 | 10 步；末 2 步 guidance；只对齐 1 个 AC step | 3/12 | guidance 初版 |
| 09-15，19:25 | Long core3 | 10 步；末 3 步 guidance；对齐 2 个 AC step | 5/12 | 正确 789 方案的小样本先导 |
| 09-16，12 时 | Long core3，seed 7 | 动作 denoise trace 与初始噪声检查 | 诊断运行 | 不计主要成功率 |
| 09-16，14:20 | Long core3 | `after_flow_steps=[6,7,8]` 与 1 步＋JEPA 并列运行 | 3/12 与 5/12 | 前者插入位置错误，**不作 789 结果** |
| 09-16，15 时 | task 410，seed 7 | 同一初始观测的首次动作 chunk 对比 | cosine/L2 诊断 | 不计主要成功率 |
| 09-16，16 时 | Long core3 | 视频、动作均 1 步；无 JEPA | 3/12 | 1 步小样本对照 |
| 09-16，18:12 | Long8，8 ID×4 seed | 10 步无 JEPA vs 视频/动作均 1 步＋JEPA | 14/32 vs 14/32 | 配对系统比较；不同步数 |
| 09-16，23:12 至 09-17，04:37 | Long36，36 ID×4 seed | 10 步无 JEPA vs 10 步＋789 JEPA | **80/144 vs 88/144** | 目前主实验 |

### 3.1 2026-09-07：pose16 的 Best-of-8 早期验证

16 个 `libero_object` 位姿/参考变体，每个 4 个 seed。原始 FastWAM 单候选成功 **35/64（54.69%）**，Best-of-8 JEPA 成功 **37/64（57.81%）**；7 次 rescue、5 次 regression。按目标物体看，Salad Dressing **9/16→14/16**，Tomato Sauce **13/16→9/16**，效果方向相反。原始 FastWAM 为 `replan=10`、无未来视频；JEPA 为 `replan=8`、未来视频开启，所以这是两套完整系统的比较，不是排序器单因素收益。逐 ID 的 16 行表、视频观察和文件索引见[历史实验日志](FASTWAM_JEPA_EXPERIMENT_LOG.md)与[原始比较表](../../evaluate_results/fastwam_jepa_libero_plus_pose16_20260907_153555/comparison_with_original.md)。

### 3.2 2026-09-09 至 09-10：Long core3、对齐对照与 joint K=8

core3 固定 `libero_10` 的 Plus **410**（黑碗放进底层抽屉并关闭）、**662**（黄白杯放入微波炉并关门）、**457**（两杯分放左右盘），各 4 seed，共每方法 12 轮。三者均为机器人初始姿态变体。结果按任务展开：

| 方法 | 410 | 662 | 457 | 合计 |
|---|---:|---:|---:|---:|
| 原始 action-only FastWAM：无未来视频、`replan=10` | 1/4 | 0/4 | 0/4 | **1/12** |
| 对齐 K=1：未来视频开启、`replan=8` | 2/4 | 0/4 | 0/4 | **2/12** |
| 历史两调用 Best-of-8 JEPA：`infer_joint(K=1)`＋`infer_action(K=8)` | 3/4 | 0/4 | 1/4 | **4/12** |
| 新 joint Best-of-8：`infer_joint(K=8)` | 1/4 | 0/4 | 0/4 | **1/12** |

历史 Best-of-8 对**对齐 K=1**净增 2 次成功，不是对原始 action-only 的净增 3 次。新 joint K=8 在这个切片上只成功 1 次；它虽将一条视频与八条动作放入同一 joint flow，却改变了动作生成路径。候选排名接近时，小数值差异可能翻转选择并令后续闭环分叉。662 在这些方法下持续 0/4，后来未纳入主要 Long36。详见[固定协议](LIBERO_PLUS_LONG_CORE3_JEPA_PROTOCOL.md)、[历史实验日志](FASTWAM_JEPA_EXPERIMENT_LOG.md)、[joint K=8 原始结果](../../evaluate_results/fastwam_jepa_joint_k8_libero_plus_long_core3_20260910/)。

同日的单卡延迟探针测得：未来视频对齐 K=1 **498.0 ms/replan**，历史两调用 Best-of-8 **965.3 ms/replan**，joint K=8 **860.2 ms/replan**，均为排除首次编译后的 warm 值。历史两调用中 JEPA encoder＋predictor/ranking 约 **278.0 ms/replan**；其余增量主要来自单独的 K=8 动作生成。joint K=8 虽快约 105.1 ms/replan，但上表显示其成功率不可直接继承历史两调用结果。三次 replan 的延迟探针样本很小，不能与后续全量 rollout 平均耗时混为一谈。细节见[延迟实验日志](FASTWAM_JEPA_EXPERIMENT_LOG.md)。

### 3.3 2026-09-15：从单段 guidance 到正确的 789 guidance

在 core3 上，先试 **10 步 action flow 的末两步** `last_flow_steps=2`，`ac_steps=1`、`step_size=0.02`，只使用第一段视觉目标和动作 0–3，得到 **3/12**（410 为 1/4、662 为 0/4、457 为 2/4）。随后将设置改为 **`last_flow_steps=3`，`ac_steps=2`**：索引 7、8、9 的每个 flow step 都用两段视觉目标和动作 0–7 求梯度，得到 **5/12**（410 为 3/4、662 为 0/4、457 为 2/4）。与上一版配对，有 3 次 rescue、1 次 regression。正确脚本为[steps789_ac2](../../scripts/run_jepa_guidance_steps789_ac2_long_core3.sh)，报告见[core3 guidance 简报](JEPA_GUIDANCE_STEPS789_AC2_LONG_CORE3_REPORT.md)。

更早的 `last_flow_steps` 扫描只在 `libero_object` 两个目标 ID **2169、2211** 上完整跑了 K=1 和 `last_2`（均 **5/8**）；`last_4` 只留下 4 个 episode（**3/4**），其余计划档位没有完整结果。它是探索记录，不能作为 Long core3 或 Long36 的正式对照。数据见[扫描目录](../../evaluate_results/jepa_guidance_last_steps_sweep_20260915_step_interval_2_4_6_8_10/)。

### 3.4 2026-09-16：插入位置问题、1 步消融和动作诊断

14:20 的 boundary 对比运行了两个设置。`between_67_78_89` 配置文件中实际是 **`after_flow_steps=[6,7,8]`**，即在索引 6、7、8 后修正，所得 **3/12**（410 2/4、662 0/4、457 1/4）。它**不是**“7→8、8→9、9 之后”对应的 `[7,8,9]`，本报告从正确 789 结果中剔除这条旧错误案例。另一个设置是视频和动作都 1 步、`after_flow_steps=[0]` 加 JEPA，得到 **5/12**（410 3/4、662 0/4、457 2/4）。结果见[boundary 目录](../../evaluate_results/jepa_guidance_boundary_compare_long_core3_20260916_1420/)；错误配置见[对应脚本](../../scripts/run_jepa_guidance_boundary_compare_long_core3.sh)。

之后运行视频和动作都 **1 步、无 JEPA**的 core3 对照，得 **3/12**（410 2/4、662 0/4、457 1/4）；[脚本](../../scripts/run_one_denoise_no_jepa_long_core3.sh)、[结果](../../evaluate_results/one_denoise_no_jepa_long_core3_20260916_one_denoise_no_jepa/)。与 core3 的 1 步＋JEPA 5/12 相比可看到净增 2 次，但 JEPA 组还通过单独 `infer_action` 产生实际动作，而无 JEPA 组执行 `infer_joint` 的动作，因此不能把这 2 次全部归因于梯度本身。

另有 task 410、seed 7、同一初始观测下的**首次动作 chunk 诊断**：完整 32 步动作 cosine **0.9987**、L2 **0.325**；实际执行的前 8 步 cosine **0.9983**、L2 **0.192**，以 10 步动作为基准的相对 L2 **5.78%**；前 8 步只看 6 维运动，cosine **0.9940**、相对 L2 **10.98%**。前 8 步夹爪命令相同，因而会抬高整体 cosine。这个诊断只说明一例中方向接近、幅值并不相同；视频与动作的步数都发生变化，不应据此给 action denoise 单独归因。对应[原始动作和视频](../../evaluate_results/action_compare_task410_seed7_20260916/)；相邻的[action denoise trace](../../evaluate_results/action_denoise_trace_long_core3_seed7_20260916/)用于检查推理轨迹，均不计入成功率汇总。

### 3.5 2026-09-16，18:12：Long8，10 步与 1 步＋JEPA

四类扰动各选两个固定 ID：机器人初始姿态 410/457，物体布局/干扰物 1945/2046，相机 1027/931，背景纹理 145/158。四个 seed，每组 32 个 episode，总共 64。对照组视频和动作都是 **10 步、无 guidance**；实验组视频和动作都是 **1 步**，随后用 JEPA `after_flow_steps=[0]`、`ac_steps=2`、`step_size=0.02` 引导单独生成的动作。共同使用 `sigma_shift=5.0`、未来视频开启、`replan=8`。

| 扰动类别 | 对照成功 | 1 步＋JEPA 成功 |
|---|---:|---:|
| 机器人初始姿态 | 1/8 | 4/8 |
| 布局/干扰物 | 8/8 | 8/8 |
| 相机视角 | 0/8 | 0/8 |
| 背景纹理 | 5/8 | 2/8 |
| **总计** | **14/32（43.75%）** | **14/32（43.75%）** |

逐 task–seed 有 4 次 rescue、4 次 regression；两组都成功 10 对，都失败 14 对。对照每次 replan 平均 **0.625 s**，实验组 **1.211 s**。因此 1 步方案没有在该实现中获得端到端推理加速。两组同时改变视频步数、动作步数、guidance 和动作生成路径，成功率相等不代表动作一致，也不能单独测出 JEPA 贡献。相机两 ID 都为 0/4，后来从主要任务集合去掉。详细的架构、逐 ID、动作步数和耗时见[Long8 完整报告](LONG8_ONE_JEPA_VS_TEN_CONTROL_REPORT_20260916.md)。

### 3.6 2026-09-16 23:12 至 09-17 04:37：Long36 正式扩展

去掉相机扰动，保留 **6 个原始目标**，每个目标在**机器人初始姿态、物体布局/干扰物、背景纹理**三类下各选 **2 个 Plus 变体**，形成 `6 × 3 × 2 = 36` 个不同 task ID。纹理统一使用 `table_1` 与 `table_12`；选定 ID 的 BDDL 与初始状态文件均已核对。任务清单见[Long36 列表](task_lists/libero_plus_long_36_three_perturbations_two_variants.txt)。

| 项目 | 10 步对照 | 10 步＋789 guidance |
|---|---|---|
| 视频 flow denoise | 10 步 | 10 步 |
| 实际执行动作 flow denoise | 10 步 | 10 步，索引 7、8、9 施加 guidance |
| JEPA / 候选 | 关闭；K=1 | 冻结 V-JEPA2-AC；单条动作，不做 Best-of-8 |
| 其他共同设置 | `sigma_shift=5.0`、未来视频开启、`replan=8`、四个 seed、每组合 1 trial、同一 checkpoint 与统计文件 | 同左；`ac_steps=2`、`step_size=0.02`、`verify_descent=false` |

对照组复用 Long8 已有的 **6 ID×4 seed=24 轮**（410、457、1945、2046、145、158）；补跑其余 **30 ID×4 seed=120 轮**。789 组对全部 **36 ID×4 seed=144 轮**重新运行。两张 RTX 4090（物理 GPU 0、1）先跑补充对照，再跑完整实验组；每个 seed 的 T5 加载在前一张卡释放权重后开始。启动器验证了每个对照 seed 有 30 个结果、每个实验 seed 有 36 个结果，最终 `status.txt=complete`。实验组 worker 的 guidance 诊断实际记录了 `flow_step=7,8,9`，确认使用了正确插入位置。因此**本次新运行 264 个 episode，最终比较 144 对、共 288 个有效 episode**。[运行脚本](../../scripts/run_long36_ten_vs_789_2gpu.sh)、[结果目录](../../evaluate_results/long36_ten_vs_789_20260916_231154_2gpu/)、[step 诊断日志](../../evaluate_results/long36_ten_vs_789_20260916_231154_2gpu/ten_guidance_789/seed_7/.worker_pool/logs/worker_0.log)。

#### 总体、按类别和按原始目标

| 范围 | 对照 | 789 guidance | 净变化 |
|---|---:|---:|---:|
| **全部 36 ID** | **80/144（55.56%）** | **88/144（61.11%）** | **+8/144；+5.56 pp** |
| 机器人初始姿态，12 ID | 19/48 | 22/48 | +3 |
| 布局/干扰物，12 ID | 39/48 | 44/48 | +5 |
| 背景纹理，12 ID | 22/48 | 22/48 | 0 |
| 去掉背景纹理，24 ID | 58/96（60.42%） | 66/96（68.75%） | +8/96；+8.33 pp |

| 原始目标 | 包含的三类扰动、两变体；每组 24 轮 | 对照 | 789 guidance |
|---:|---|---:|---:|
| 0 | 开炉灶、放 moka pot | 12/24 | 13/24 |
| 1 | 黑碗入底层抽屉并关上 | 8/24 | 11/24 |
| 4 | 汤罐和奶酪盒入篮 | 24/24 | 24/24 |
| 5 | 汤罐和番茄酱入篮 | 16/24 | 16/24 |
| 6 | 奶酪盒和黄油入篮 | 20/24 | 18/24 |
| 7 | 两个杯子分放左右盘 | 0/24 | 6/24 |

完整配对四格：两组都成功 **76 对**；仅 789 组成功 **12 对**；仅对照成功 **4 对**；两组都失败 **52 对**。四个 seed 各含 36 ID：seed 7 为 **19→19**、seed 17 为 **22→26**、seed 27 为 **19→21**、seed 37 为 **20→22**。按类别，机器人姿态有 5 次 rescue、2 次 regression；布局有 5 次 rescue、0 次 regression；纹理有 2 次 rescue、2 次 regression。

#### 逐 Plus ID 成功数

每行分母都是 4 个 seed；两行属于同一个原始目标、同一扰动类别下的两个不同变体。

| 类别 | 原始目标 | Plus ID | 对照 | 789 | 净变化 |
|---|---:|---:|---:|---:|---:|
| 初始姿态 | 0 | 365 | 1/4 | 2/4 | +1 |
| 初始姿态 | 0 | 366 | 1/4 | 1/4 | 0 |
| 初始姿态 | 1 | 410 | 1/4 | 1/4 | 0：1 rescue、1 regression |
| 初始姿态 | 1 | 411 | 0/4 | 1/4 | +1 |
| 初始姿态 | 4 | 567 | 4/4 | 4/4 | 0 |
| 初始姿态 | 4 | 568 | 4/4 | 4/4 | 0 |
| 初始姿态 | 5 | 289 | 0/4 | 0/4 | 0 |
| 初始姿态 | 5 | 290 | 0/4 | 0/4 | 0 |
| 初始姿态 | 6 | 330 | 4/4 | 4/4 | 0 |
| 初始姿态 | 6 | 331 | 4/4 | 3/4 | −1 |
| 初始姿态 | 7 | 457 | 0/4 | 2/4 | +2 |
| 初始姿态 | 7 | 458 | 0/4 | 0/4 | 0 |
| 布局/干扰物 | 0 | 1945 | 4/4 | 4/4 | 0 |
| 布局/干扰物 | 0 | 1933 | 4/4 | 4/4 | 0 |
| 布局/干扰物 | 1 | 1959 | 4/4 | 4/4 | 0 |
| 布局/干扰物 | 1 | 1960 | 3/4 | 4/4 | +1 |
| 布局/干扰物 | 4 | 2046 | 4/4 | 4/4 | 0 |
| 布局/干扰物 | 4 | 2047 | 4/4 | 4/4 | 0 |
| 布局/干扰物 | 5 | 2059 | 4/4 | 4/4 | 0 |
| 布局/干扰物 | 5 | 2060 | 4/4 | 4/4 | 0 |
| 布局/干扰物 | 6 | 2093 | 4/4 | 4/4 | 0 |
| 布局/干扰物 | 6 | 2094 | 4/4 | 4/4 | 0 |
| 布局/干扰物 | 7 | 2127 | 0/4 | 4/4 | +4 |
| 布局/干扰物 | 7 | 2128 | 0/4 | 0/4 | 0 |
| 背景纹理 | 0 | 0 | 2/4 | 2/4 | 0 |
| 背景纹理 | 0 | 2 | 0/4 | 0/4 | 0 |
| 背景纹理 | 1 | 33 | 0/4 | 0/4 | 0 |
| 背景纹理 | 1 | 35 | 0/4 | 1/4 | +1 |
| 背景纹理 | 4 | 137 | 4/4 | 4/4 | 0 |
| 背景纹理 | 4 | 138 | 4/4 | 4/4 | 0 |
| 背景纹理 | 5 | 145 | 4/4 | 4/4 | 0 |
| 背景纹理 | 5 | 147 | 4/4 | 4/4 | 0 |
| 背景纹理 | 6 | 158 | 1/4 | 0/4 | −1 |
| 背景纹理 | 6 | 160 | 3/4 | 3/4 | 0：1 rescue、1 regression |
| 背景纹理 | 7 | 193 | 0/4 | 0/4 | 0 |
| 背景纹理 | 7 | 195 | 0/4 | 0/4 | 0 |

其中 **2127 的 0/4→4/4** 是单 ID 最大正向变化；它的同一原始目标、同类扰动第二变体 **2128 仍 0/4→0/4**。这说明“对某个具体 Plus 场景有用”与“对该扰动类别普遍有用”是不同结论。goal 4 的 24 对在两组均成功，提供的是天花板信息；若只看差异发现能力，这部分对区分方法贡献很少。goal 7 的 6 次净增占总体 8 次净增的 **75%**；若事后去掉该目标，剩余为 **80/120→82/120**，差距仅 **+2/120**。这些均应作为异质性描述，而不是事后筛选总体结论。

#### 动作步数和时间

| 指标 | 10 步对照 | 10 步＋789 | 口径 |
|---|---:|---:|---|
| 全部 episode 平均低层策略动作步 | 464.7 | 450.9 | 失败常到 700；成功集合不同 |
| 全部 episode 的 replan 总数 | 8,436 | 8,186 | 各 144 轮累计 |
| 按 replan 加权的平均耗时 | **0.559 s** | **2.390 s** | 包含首次 replan；按每次记录加权 |
| 其中 `infer_joint` 平均耗时 | 0.552 s | 0.543 s | 两组都做 10 步视频/内部动作 flow |
| 实验组 `infer_action` 平均耗时 | — | **1.688 s** | 含 10 步动作 flow 与 789 JEPA guidance |
| 每个 episode 平均 rollout wall time | 47.3 s | 153.3 s | 不含 worker 初始化和 T5 加载 |
| 每个 task 结果平均 `duration` | 79.6 s | 185.9 s | 含评测与视频/结果写盘 |

运行层面的总墙钟：补充对照 **23:12:08→00:40:56，约 1 小时 29 分**；789 组 **00:40:56→04:37:45，约 3 小时 57 分**。以上 per-replan 均值是由各 task 结果中的 `episode_timings[0].per_replan` 重新汇总；它们不是仅排除首次编译后的 warm microbenchmark。24 轮复用对照来自同日的 Long8 运行，其 GPU 并发环境与本轮不完全相同，因此时间数字适合工程参考，不应视为严格隔离硬件负载的微基准。

## 4. 现阶段可以支持的结论与边界

1. **最新固定集合上存在描述性的净提升。** 36 ID 的配对差为 +8/144，12 次 rescue、4 次 regression；不意味着 JEPA 对所有长任务、所有扰动都稳定提高成功率。每个具体 Plus ID 只有四个推理 seed，36 ID 又嵌套在六个原始目标之下。
2. **效应高度依赖任务与变体。** goal 7 和 ID 2127 是主要贡献者；同一目标的 2128 没有变化。布局类别有大量 4/4 天花板；初始姿态和纹理也有多项 0/4 地板。
3. **现有对照比较的是完整推理路径。** 无 JEPA 组执行 `infer_joint` 的动作；789 组执行单独 `infer_action` 经 JEPA 修正的动作。因此即使 flow 步数和未来视频设置一致，增益也不能完全分解成“JEPA 梯度的纯因果作用”。若要单独归因，还需同一 `infer_joint`＋`infer_action` 路径、只关闭 guidance 的控制组。
4. **速度代价在最新实现中较大。** 789 组每次 replan 约 2.390 s，对照约 0.559 s；`infer_action` 的 1.688 s 包括动作 flow、JEPA 反传及其他处理。数字不能称为“JEPA 本体单独增加 1.688 s”。
5. **视觉目标是模型预测而非环境真值。** JEPA 能量低不保证抓取/接触真实成功。早期 Tomato Sauce 退化视频提示预测未来可能放大错误接触假设；需要结合具体 rescue/regression 视频与 JEPA loss 轨迹分析。
6. **背景纹理子集和任意事后删目标均为探索性分析。** Long36 的主分母是全部 144 对；去纹理的 96 对、去 goal 7 的 120 对应作为敏感性展示。

## 5. 复现和后续核查入口

| 内容 | 文件或目录 |
|---|---|
| 主要时间顺序日志 | [FASTWAM_JEPA_EXPERIMENT_LOG.md](FASTWAM_JEPA_EXPERIMENT_LOG.md) |
| Long core3 固定协议 | [LIBERO_PLUS_LONG_CORE3_JEPA_PROTOCOL.md](LIBERO_PLUS_LONG_CORE3_JEPA_PROTOCOL.md) |
| Long8 完整报告 | [LONG8_ONE_JEPA_VS_TEN_CONTROL_REPORT_20260916.md](LONG8_ONE_JEPA_VS_TEN_CONTROL_REPORT_20260916.md) |
| 最新 Long36 全部 task ID | [libero_plus_long_36_three_perturbations_two_variants.txt](task_lists/libero_plus_long_36_three_perturbations_two_variants.txt) |
| 最新 Long36 补充对照 ID | [libero_plus_long_30_ten_step_control_supplement.txt](task_lists/libero_plus_long_30_ten_step_control_supplement.txt) |
| 最新 Long36 启动脚本与配置 | [run_long36_ten_vs_789_2gpu.sh](../../scripts/run_long36_ten_vs_789_2gpu.sh)、[实验组 seed 7 的解析配置](../../evaluate_results/long36_ten_vs_789_20260916_231154_2gpu/ten_guidance_789/seed_7/manager_config.yaml) |
| 最新 Long36 原始结果、rollout 视频、worker 日志 | [long36_ten_vs_789_20260916_231154_2gpu](../../evaluate_results/long36_ten_vs_789_20260916_231154_2gpu/) |
| 被复用的 24 轮对照 | [Long8 的 ten_control](../../evaluate_results/long8_one_jepa_vs_ten_control_20260916_long8_4gpu/ten_control/) |
| 评测与 JEPA 适配代码 | [eval_libero_single.py](eval_libero_single.py)、[vjepa2_ac_ranker.py](vjepa2_ac_ranker.py) |

后续若要检验“JEPA 梯度本身”的增益，优先在 Long36 的相同 seed 与 ID 上加一组 **10 步 `infer_joint`＋10 步 `infer_action`，但关闭 JEPA guidance**；这样动作生成路径、视频参考和计算流程才更接近。然后重点复核 ID 2127 的四次 rescue、457 的两次 rescue，以及 331、158 的 regression，并记录每次 guidance 的 loss、梯度 RMS、修正前后动作和实际子目标进度。
