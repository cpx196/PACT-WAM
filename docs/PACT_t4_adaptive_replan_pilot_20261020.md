# PACT：仅依赖 t+4 Predictor 的自适应 replan 早期验证

日期：2026-10-20，Asia/Shanghai。现场核对时间：15:47–15:53。

状态：**方案已核对并落盘；本文不是实验已启动或结果已成立的声明。**

目标：今晚以小规模实验回答一个问题——现有 Predictor 对实际执行的四步预测误差，能否帮助选择提前重新规划的时机，而不只是增加 WAM 调用次数？

## 1. 决策摘要

- 保留当前 FastWAM、冻结 JEPA Encoder、最终 epoch10 Predictor。
- 只调用 Predictor 第一段 `phase=0`：输入当前观测、当前状态、接下来四条动作；输出 `[I_t, I_{t+4}]` 的联合 latent。
- 不训练、不 guidance、不改动作、不调用第二段 AR，不扩大到 16/32 步预测。
- 动态方案只在 **replan=4 与 replan=8** 之间选择，最大执行仍为 8 步。
- 第 4 步得到真实反馈后，误差高则取消本计划第 5–8 步并重新规划；否则执行完第 8 步后正常重规划。
- 第一版使用同步模拟器：做清晰的机制验证，不混入异步队列、动作拼接、真实时间截止控制。
- 初版主信号只用 `Predictor ↔ 实际执行`。不使用 P–W loss，不解码 WAM 视频作为触发输入。
- 单卡、单进程常驻加载，计划约 2–3 小时实现核验 + 1.5–2.5 小时实验与整理。由正式实施开始计，预留 **4–6 小时**，排队另计。若约 16:00 开始，目标约 20:00–22:00 给出结论。
- 今晚得到的是“是否值得继续”的结论，不做 144/800 条评测。

这版只验证“偏离时提前 replan”的价值，**不验证延长到 12/16 步的计算节省**。动态方案的 WAM 调用次数可能高于固定 8；必须用调用预算对照判断增加计算是否用对了地方。

## 2. 已核实的现有资产与证据

### 2.1 权重及接口

服务器仓库：

`/data/pxchen/PACT-WAM`

| 模块 | 路径 / 约束 |
| --- | --- |
| FastWAM | `checkpoints/fastwam/libero_optional_idm_2cam224.pt` |
| Encoder | `checkpoints/jepa_wam_assets_20261009/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt` |
| Predictor | `artifacts/fastwam_predictor21_ar2_full10epoch_bs16_20261014/step19690.pt` |
| Predictor 归一化 | `artifacts/fastwam_predictor21_ar2_stride20_20261014/train_only_stats.json` |
| 当前模型封装 | `scripts/fastwam_ar2_eval.py` |
| 原生评测 | `experiments/libero/eval_libero_single.py` |
| Backbone 定义 | `artifacts/predictor2.1ac1009/smoke_joint_predictor.py` |
| Encoder 封装 | `/data/pxchen/JEPA_WAM/prismatic/models/backbones/vision/vjepa_vit.py` |

核实情况：checkpoint 是两段各 4 步的 AR 训练版本；首段 `phase=0` 已独立受监督。本方案只用它，等价于使用一个 t+4 接口，不声称现有模型可直接处理 16/32 步。

源码指纹（2026-10-20）：

```text
scripts/fastwam_ar2_eval.py
6b1f9cc7ef47c8de0e188b12db7c9add34595d262be4743aa77843c647767552

experiments/libero/eval_libero_single.py
8a24a14bdfdc484a925ca8ad35276a87754a6cf8232c88b0c6f86ea40d2c29bc

artifacts/fastwam_predictor21_ar2_stride20_20261014/train_only_stats.json
df9127e098763ca587212d0fbb5f882417c822ce2b4034512b3816d4c8c3ce2a
```

启动时记录权重路径、文件元信息和实际加载 step；不为生成方案重复扫描大权重或复制权重。

### 2.2 必须吸收的历史证据

1. `artifacts/ar2_long642_alignment_probe_20261015`：
   642 个验证演示窗口的 Predictor–真实未来匹配优于错时间/错任务配对。说明表示对应有基础；不等于能检测执行失败，也不证明重规划可修复。
2. `artifacts/ar2_pw_real30_alignment_20261015`：
   30 窗口中，靠近专家动作时 P–真实演示 future L1 下降，但 P–W loss 反而略升。不能继续把 P–W loss 当作动作质量标签。原生动作没有真的在演示分支上执行，因此其与专家 future 的距离也不是物理预测误差。
3. 六条人工筛选案例：
   task151 的关键掉落发生在同一 8 步前缀内；其他若干案例的想象进展主要位于 +20/+32。t+4 系统不应被宣称能提前检测那些远期阶段停滞。
4. 最新多 seed 测试已完成：
   `evaluate_results/fastwam_selected6_paired_seeds17_27_37_20261020/summary.json`
   36 rollouts、18 对；无 guidance 成功 6/18，guidance 5/18，rescue 2、harm 3；完成时间约 15:34。
   这支持先验证新的 replan 用法，不继续堆叠 guidance 强度。

## 3. 精确输入输出与因果顺序

令 t 表示扣除 30 步 settling 后的有效控制步。

在本轮 WAM 计划开始、尚未执行四步时：

```text
I_t：实时主视角图像
s_t：实时状态 [eef_xyz(3), eef_rotvec(3), gripper_qpos(2)]
a[t:t+4]：随后真正交给 env.step 的四条命令

z_t = E(I_t)
p_t4 = Predictor(z_t, normalize(a[t:t+4]), normalize(s_t), phase=0)
```

实际执行四次 `env.step` 后：

```text
y_t4 = E_pair([I_t, I_{t+4}])
e_t4 = mean(abs(p_t4 - y_t4))   # 全 576×1024 个元素
```

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| 当前图像送 Encoder | `[1,3,384,384]` | 使用官方 wrapper 的当前图像接口 |
| 当前 latent | `[1,576,1024]` | 不池化，不加额外 LayerNorm |
| 原始动作 | `[1,4,7]` | 四条有顺序的命令，保留 gripper |
| 状态 | `[1,8]` | 当前状态，不是未来状态序列 |
| Predictor 输入序列 | `[1,581,1024]` | 4 action + 1 state + 576 visual |
| 首段输出 | `[1,576,1024]` | 联合图像对表示，不是单独 t+4 的图像 latent |
| 真实图像对 | `[1,2,3,384,384]` | 在执行结束后才构造 |
| 真实联合 latent | `[1,576,1024]` | 必须调用 `encode_pair` |

重要约束：

- 不把 `E(I_{t+4})` 单帧 latent 当作真实监督；不平均两个单帧 latent 冒充联合编码。
- 不把四条动作求和或平均，不做 OSC 位移积分，不把旋转指令当绝对姿态。
- 状态轴角与动作相对旋转是不同量，分别走原投影层。使用既有 `quat2axisangle` 和训练统计。
- 从 `_predict_action_chunk` 返回后的可执行动作取样；它已经完成 native postprocess/gripper 处理。不能再反转或二值化第二次。
- 核对四条 Predictor 输入与四次实际 `env.step` 参数逐元素相等；若 env 内部有裁剪，记录 API 命令及裁剪口径，不把内部物理位移混称输入动作。
- 保存 `I_t` 的副本；不能让引用被后续观测覆盖。
- 当前 P 在执行前生成；y 只在执行后生成。未来图像仅作已经发生的反馈，不是在线前视输入。
- WAM 两视角保持原样；JEPA 只读主视角。
- native LIBERO 取图执行一次既有 180° 校正，JEPA 额外旋转为 0；不靠“正/倒”文字猜测，smoke 保存同一帧的输入路径对照。

## 4. 初版策略：仅提前，不延长

在每个 WAM 计划开始时保存完整动作，但最多提交前 8 步。四步检查后再决定是否提交第 5–8 步。

```text
生成动作 plan_id=k，记录起点 t
保存 I_t / s_t / 最终四条命令，计算 p_t4
执行四步（若提前 done，立即结束，窗口作 censored，不补帧）
编码实际图像对，得到 e_t4

若 e_t4 > 固定阈值 tau：
    丢弃尚未执行的第 5–8 步，从当前真实观测重新调用 WAM
否则：
    执行本计划第 5–8 步，再重新调用 WAM
```

- 第 8 步无条件刷新，所以不用第二次 Predictor，也不用 phase=1。
- 新计划的第一段始终从新的真实图像与真实状态重新开始。
- 不修改 gripper，不做动作 blending/ensembling，不额外加人工暂停。
- NaN/Inf、维度错误、帧索引错误直接记录失败并停止当前实验；不能悄悄退化成“全部触发”后继续出结果。
- e 高表示模型与执行不一致，可能是 OOD/模型误差，不宣称识别出掉落或碰撞原因。
- e 低也不能证明下一段一定正常。首版检验的是这个决策是否有经验价值。

### 4.1 阈值校准

先跑独立的两条校准轨迹（见第 6 节），固定 replan=8，监控模块只记录、不改变动作。

取两条轨迹四步误差的**轨迹等权 75% 分位数**作为唯一主阈值 tau：

- 每条轨迹的窗口权重总和均为 1/2，避免长轨迹支配阈值。
- 它是初始化调用频率的无标签校准，不是“正常误差分布”或 25% 故障概率。
- 不按任务成败、不按 held-out 成功率调阈值。
- 不按 replan 位置单独设阈值，不强制前四个 replan 触发；保留每个位置的数据看是否出现历史位置偏置。
- 固定后写 `threshold.json`，记录校准 manifest、分位数、样本数、文件 hash。正式评测期间不更新。
- 如果有效窗口总数 <64 或任一轨迹 <16，或严重数值异常，则停止并报告校准不足，不使用测试轨迹补调。

## 5. 必须先排除的实现错误

### 5.1 计数与录像解耦

原生 `run_single_episode` 的 `current_replan_step` 只在 `visualize_future_video` 分支中增加；`current_replan_idx` 也依赖有无 predicted frames。

本方案关闭预测视频解码，因此必须新增独立的：

`control_step / plan_id / plan_step / monitor_window_id`

四次真实 `env.step` 后才核验，不能依赖录像变量。只在新观测已经更新后计算 y。最终不足四步窗口标记 censored。

### 5.2 保持主策略等价

- 初版独立入口，不改历史评测脚本的默认行为，不覆盖历史目录。
- 不能直接套用 `fastwam_ar2_fullvideo_eval.py`：它硬编码 guidance 开启、replan=8 和 9 帧视频。
- 不直接调用 `ARRanker.differentiable_energy`：它要求 8 步、会算两段 AR。
- 可以复用其 Encoder、Predictor、统计量加载与像素处理；新增单独 first-step monitor 接口。
- monitor 的实例化会消耗 RNG：加载顺序或保存/恢复 RNG 必须保证 shadow 监控不改变 WAM 采样。
- 模型 `.eval()`、冻结、`inference_mode()`；WAM BF16，JEPA FP32，不编译。
- 单进程顺序使用 GPU，暂不加 stream overlap，便于测清真实开销。

### 5.3 关闭不必要解码

- `guidance=false`、候选 ranking=false、原 trigger/probe=false。
- 不打开 `visualize_future_video`；这会改变调用路径且产生 VAE decode 成本。
- WAM 仍生成其 action 所需的视频 latent；不能把“不解码”误称“不生成视频”。
- smoke 对同一观测、同一 RNG，比较既有 full-video 和关闭 decode 的前 8 步动作。若差异超出事先规定容差，先定位，不能把更换推理路径造成的差异当调度收益。
- 相同路径 shadow on/off 的动作及 RNG 应一致；若底层非确定性影响复现，用重复 no-monitor 运行建立数值噪声底，报告误差，不随意放宽。

### 5.4 历史视频不能代替精确监控缓存

原视频是 192 视角压缩、实际视频每两步取一帧，还带展示文字；原日志通常仅可选保存第一条 action chunk，没有每个窗口的全动作/状态记录。它们适合定位事件，不适合直接重建精确 Predictor 输入。

今晚重新采集少量在线原始窗口；不重新处理 LIBERO90/10 大缓存，不下载数据，不训练新模型。

## 6. 小规模矩阵与公平对照

### 6.1 固定案例

| 用途 | Task IDs | Seed | 说明 |
| --- | --- | --- | --- |
| 无标签阈值校准 | 260、321 | 47 | 固定8、shadow监控；不参与主效果汇总 |
| 主要评测 | 151、242、559、590 | 47 | 四条完全配对，新增seed；trial0/init0 |

选择包括历史短时掉落、杯口停滞、提罐失败及已有控制成功可能的案例。不要只保留救回案例。四条本身来自历史失败筛选，存在选择偏差。

校准与测试没有重复 task ID，但部分共享 base task；seed47只改变随机性，不代表新的初始布局。本次是机制试验，不声称跨任务泛化或总体 OOD 提升。环境成功标志只在事后汇总，不作为在线 trigger。

### 6.2 必做与条件对照

| 组 | 策略 | 轨迹数 | 用途 |
| --- | --- | ---: | --- |
| C | 固定8 + shadow监控 | 2 | 校准tau |
| B8 | 固定8 + shadow监控 | 4 | 原策略效果、同监控成本对照 |
| A48 | tau驱动4/8 | 4 | 核心机制 |
| R48 | 随机4/8 + shadow监控 | 4 | 同目标调用频率，不依赖误差信号 |
| B4（条件追加） | 固定4 + shadow监控 | 4 | 更密集重规划的效果/代价参照 |

核心共 **14 条**；仅在进度允许且 A48 非退化时追加 B4，最多 18 条正式轨迹。smoke 不计效果。

R48 在每次第4步用独立 RNG，以校准触发率 q=0.25 选择提前 replan；实际 q 也记录。独立 RNG 不得消费策略 RNG。期望每轮执行 `4q+8(1-q)` 步。

注意：动态闭环改变数据分布，所以 A48 与 R48 的实际调用数未必一致。**q 相同不是严格等算力**，必须报告每100控制步 WAM调用数、总 GPU 时间。若预算相差 >15%，今晚只报告“预算未匹配，无法隔离时机收益”，不强行宣称 trigger 比随机好。后续再做预注册预算匹配，不能测试后选择有利阈值。

所有组 shadow 监控都计算相同的四步误差，但只有 A48 使用它改动作调度。它保证每次 replan 的监控路径相同；由于 replan 次数不同，总监控成本仍要分别统计。原始无监控 B8 的额外开销用 smoke 同路径计时与最近历史控制作参考，不把历史视频录制开销直接相减当实测加速。

所有正式组固定：

- 模型权重、环境版本、task description、IDM、两视角 WAM。
- denoise=10；video/action shift=1/1；WAM BF16 / JEPA FP32。
- 无guidance、无ranker、无原trigger、无compile、无action ensemble。
- max有效控制步700，settling30，trial0、initial-state index0。
- 主视角JEPA，原生相机无extrinsic/FOV override，native图像校正一次。
- 原动作反归一化、gripper符号及二值化路径保持原样。
- 每条轨迹独立重置所有随机数与环境；不得用上一条轨迹的监控状态。

## 7. 实施顺序与硬停止条件

### 阶段 A：最小实现和 smoke（预算 90–150 分钟）

拟新增独立脚本，名称为规划、当前不声称已存在：

```text
scripts/eval_fastwam_t4_replan_pilot.py
scripts/run_fastwam_t4_replan_pilot.py
scripts/analyze_fastwam_t4_replan_pilot.py
```

smoke 必须覆盖：

1. 权重/统计加载，输入输出形状、phase=0、有限数值。
2. 一张图的 native→JEPA 处理路径、主视角、旋转次数、当前接口与 encode_pair 区别。
3. Predictor 的四条命令与 env.step 一致，恰好四步后取真实帧。
4. monitor shadow on/off 不改变主策略动作和 RNG。
5. 固定8执行步索引正确；可在 CPU 控制器测试中用合成高/低误差覆盖4/8两条分支，真实评价不得伪造误差。
6. 强制done不足四步、最大步数结尾不足四步、不解码录像关闭三种边界。
7. 时延、显存峰值；检查与 WAM 不重叠时能否单卡常驻。

实现超时或等价性未通过：停止在接口报告，不连夜扩大工程范围。

### 阶段 B：校准与固定8 shadow（约 25–45 分钟）

跑2条校准并冻结 tau，再跑4条B8。

输出 e 随控制步变化、各 replan 位置分布、与简单图像变化/复制当前latent基线的对比。

可从存储的标量判断 trigger 是否近乎恒开/恒关。不能因为终局成功/失败不理想而改tau。

### 阶段 C：A48 + R48（约 40–80 分钟）

各4条完整轨迹，700步上限不变。GPU正常时全部完成；采用先A48后R48的固定顺序，记录每组实际负载，不能将共享GPU干扰导致的时间差当算法收益。

若 A48 提前触发率 >90% 或 <5% 且至少64个有效决策，先报告策略退化为近似固定4或8；可结束本轮而不追加B4。已经完成结果照常保留。

### 阶段 D：可选B4与报告（约 30–65 分钟）

到20:30尚未开始B4则跳过，不牺牲报告时间。报告最晚目标22:00；若GPU排队或实现核验导致延迟，应报告实际完成阶段，而不是将截断轨迹记为失败。

18条不是必须全部跑；核心14条已包含“错误信号是否比随机时机好”的必要对照。

## 8. 如何判读，避免再次做无效大实验

本轮不设 p<0.05 等小样本无法支撑的门槛；按轨迹列出原始结果，允许结论为不确定或无收益。

### 继续做的条件

- 接口smoke全部通过，数值可复现。
- 信号不只是 replan位置/任务身份的固定差异，不是恒触发。
- 至少看到可复核的局部现象：某次高误差后提前replan实际改变后续执行，且同任务对照更差；不只展示L1下降。
- A48 相对于B8的结果需要同时结合R48、实际调用预算。若A48与随机一样，不能把“更频繁replan”包装成JEPA贡献。
- 总成本要算当前图像编码、Predictor、真实图像对编码、同步等待与更多WAM调用；不能只算Predictor forward。

### 停止/转向的条件

- P–Z误差主要反映外观变化或模型偏差，与提前replan的收益无可见关联。
- A48与R48/B4相当或更差，且没有计算优势。
- 监控成本压过其执行收益；此时先判断是否还有做轻量核验的必要，不自动启动蒸馏/重训。
- 远期停滞没有触发而短期预测准确：这是t+4视野的边界，不立即归罪于实现，也不在本轮增加新loss。

“高误差是否代表坏动作”和“重新规划是否能解决”是两个不同问题。主效果由闭环对照回答，不能仅凭误差和终局失败的相关性代替。

## 9. 时延、显存、磁盘与运行约束

### 9.1 时间估计依据

最新36条完整评测中，控制失败轨迹约208–228秒/条，guidance失败约380–419秒/条，包含此前全视频保存等开销。

本轮不反传、只一段预测且不做WAM视频解码，但新增真实反馈编码；动态replan可最多翻倍WAM调用。**尚未实测本轮吞吐**，保守按每条约3–7分钟、14条约45–100分钟计算，再加加载、smoke、作图及实现。

smoke采样至少20个完整监控周期后更新ETA，分别记录：

```text
WAM生成
当前帧Encoder
四步Predictor
实际图像对Encoder
同步/数据拷贝
env.step/render
视频写盘
```

仿真在GPU计算时暂停，不存在真实机器人等待期间的物理演化。报告“推理/完成墙钟时间”和“有效控制步数”，不把仿真步差换算成真实控制毫秒，不宣称真实deadline已满足。

### 9.2 资源

- 优先单GPU，Batch=1，无梯度；预计峰值比此前同权重guidance低，但以smoke测量为准，计划预留约16–18GiB可分配空间。
- 15:47 GPU6总占用约4568MiB、空余19544MiB，没有活跃训练进程记录，只有历史 `[Not Found]` 4544MiB驱动残留。
- 残留不能证明属于本项目，不kill、不reset。GPU1/2也显示约4GiB占用、低利用率，不能只凭利用率就认定完全空闲。
- 启动前复查GPU进程和可用显存，单卡使用，不抢占其他用户。若OOM，停止报告，不擅自改JEPA精度或换权重。
- 磁盘`/data`现场空余约5.2TiB。本次小缓存配额2GiB，超出则停止额外影像保存；不得删除历史实验或权重。
- 模型和已有数据均从服务器本地读取；关闭自动下载、沿用已验证公共权重路径，避免意外T5下载。

### 9.3 保存最小充分数据

- 每个窗口保存起止控制步、plan_id、状态、四条实际命令、e、阈值、决策及分阶段耗时。
- 原始主视角图像只保存少量诊断窗口：校准均匀抽样、最高误差、首次提前replan及失败前片段。图像使用无损PNG，无文字覆盖；不要保存全量latent。
- 展示视频可低清，但不可回灌给Encoder。视频IO单独计时。
- 每条结束即写结果，JSONL实时flush，模型常驻，避免每个task重复加载。
- nohup + 唯一lock + launcher.pid + active_process.json + progress.json。
- 稳定条件：smoke通过、首条正式轨迹至少3个真实监控周期、数值有限、计数正确、无OOM。达到后即可退出交互，不需要盯完整夜。
- 本文不创建等待器或自动监控；实现时避免重复launcher，失败信息必须保留。

## 10. 交付结构

拟用新目录（运行前创建，禁止复用已有输出）：

`/data/pxchen/PACT-WAM/evaluate_results/fastwam_t4_replan_pilot_20261020`

```text
manifest.json                # task/seed/init、配置、代码和资产版本
smoke_summary.json           # 数值、动作等价性、方向、计数、峰值
threshold.json               # 两条独立校准轨迹、固定阈值
calibration/
fixed8/
adaptive48/
random48/
fixed4/                      # 可选
windows.jsonl                # 精确四步记录
progress.json
summary.json                 # 各任务成败/调用次数/成本/触发比例
report_zh.md
error_and_replan_timeline.png
quality_cost_comparison.png
```

报告必须包含：

1. 四个测试task的完整逐组对照表，不能只列rescue。
2. 误差与提前replan时间线，标清控制步/墙钟时刻。
3. 每100控制步调用次数、监控开销、总时间；成功提前结束会影响总调用数，不能直接横比总数。
4. A48相对B8和R48的收益/损害案例；若预算不匹配明确标记。
5. 一句决定：值得做更大验证 / 信号无明显价值 / 接口有误需修 / 当前小样本不确定。

## 11. 后续边界

今晚不承诺完整系统落地。只有本轮证明核验信号有用途，才考虑延长执行到12/16步、检查频率优化、异步推理与动作前缀衔接。

如果需要更长执行，每次仍可以从最新真实观测重新做4步预测；这与从最初状态自回归预测16步不同。但可重锚不等于可靠，应另做实验。

真正系统目标仍然是：**同样计算预算下，把重新规划放在更有价值的时刻。** 初版通过严格限定t+4、4/8决策和独立对照，先验证这个核心是否成立。
