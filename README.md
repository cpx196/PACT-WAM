# PACT-WAM

PACT-WAM 是一个面向机器人动作生成与闭环控制研究的实验代码包。项目以
[Fast-WAM](https://github.com/yuantianyuan01/FastWAM) 为基础，加入了
V-JEPA2-AC 动作排序、action-flow guidance、LIBERO-Plus 扰动评测，以及
denoising/flow-matching 动作收敛分析。

> **当前主实验配置（2026-09）**：使用 Optional-IDM 的严格级联路径
> `infer_video() → infer_action_from_video()`。video 阶段只生成 imagined future，
> 不生成无用 action；action 阶段读取冻结的 video latents 后单独生成 action chunk。
> LIBERO evaluation 显式设置 `video_sigma_shift=1.0`、
> `action_sigma_shift=1.0`，并保持旧的联合覆盖项 `sigma_shift=null`。

这个仓库保留了当前研究阶段的完整快照：源码、配置、实验脚本、历史实验报告和
轻量级评测 JSON/CSV/YAML 都保存在仓库中。模型权重、rollout 视频和本地缓存不上传 GitHub。

[![默认语言：中文](https://img.shields.io/badge/README-%E9%BB%98%E8%AE%A4%E4%B8%AD%E6%96%87-d14836.svg)](./README.md)

## 1. 项目目标

PACT-WAM 研究以下问题：

- FastWAM 的随机 action flow 在不同 denoising step 中如何收敛到可执行动作；
- 冻结的 V-JEPA2-AC 是否可以在闭环 rollout 中筛选或修正 FastWAM 动作；
- imagined future 质量、动作候选覆盖范围和 JEPA energy 如何共同影响控制成功率；
- 当 imagined future 出现错误接触或物体幻觉时，JEPA 排序是否会放大这种偏差。

项目目前只在推理阶段组合 FastWAM 和 V-JEPA2-AC，没有对两者进行联合再训练。

## 2. 系统流程

### 2.1 当前 Optional-IDM 两阶段推理

```text
LIBERO 双相机观测 + 任务文本 + proprioception
                      ↓
        infer_video（video shift = 1）
                      ↓
     imagined future / frozen video latents
                      ↓
        MoT video K/V cache（video 不读取 action）
                      ↓
 infer_action_from_video（action shift = 1）
                      ↓
               32×7 action chunk
                      ↓
              执行前 N 个动作
                      ↓
              重新观测并规划
```

当前主路径使用发布版 `libero_optional_idm_2cam224.pt`，并固定
`action_infer_mode=idm`。视频阶段不会创建或去噪 action latent；动作阶段才通过
MoT attention 读取冻结的完整 future-video tokens。不要在 PACT 的 `separate` 路径
中调用 `infer_joint()` 后再调用一次 `infer_action()`，否则会白算并丢弃第一条 action。

为保持原始 IDM 的加载与显存时序，非 guidance 路径的执行顺序严格为：视频 latent
去噪 → action 去噪 → VAE decode future video。启用 action-flow guidance 时，需要先把
future video decode 成图像；decode 完成后，ActionDiT 去噪与冻结的 JEPA encoder 在
两个 CUDA stream 上并发执行。到指定 guidance boundary 时只等待 encoder event 并运行
JEPA predictor，不会在 guidance 点才开始编码，也不会额外生成 action chunk。

### 2.2 Best-of-K JEPA ranking

当前推荐的 `separate` 路径为：

```text
infer_video()             → 只生成一条共同的 imagined future
infer_action_from_video() → 基于该 future 生成动作候选或 guided action
V-JEPA2-AC                → ranking 或 action-flow guidance
执行动作                  → 执行选中/修正后 action chunk 的前 N 步
```

每 4 个 LIBERO 低层动作对应 1 个 JEPA action-conditioning step。当前 JEPA
实验执行 8 个低层动作，即使用两个未来转移进行排序。

### 2.3 Action-flow guidance

guidance 在动作 flow 的指定边界上计算 JEPA loss 对 clean-action estimate 的梯度，
再对 FastWAM 的普通 scheduler 更新结果施加 additive correction。代码支持两种尺度：

- `normalize_gradient=true`：历史固定-RMS方案；先归一化梯度，再由 `step_size`
  指定每次 correction RMS；
- `normalize_gradient=false`：保留原始梯度相对大小；`step_size` 是全局倍率，
  `max_delta_rms` 只作为最大信任区域，不会把小梯度强制放大到上限。

当前原始梯度验证协议使用：

```yaml
guidance.enabled: true
guidance.overlap_encoder_with_action: true
guidance.after_flow_steps: [7, 8, 9]
guidance.normalize_gradient: false
guidance.step_size: 1.0
guidance.max_delta_rms: 0.02
guidance.ac_steps: 2
guidance.verify_descent: true
```

`configs/sim_libero.yaml` 为兼容历史实验仍默认 `normalize_gradient=true`；复现实验时必须
显式写出上述三个尺度参数。`verify_descent=true` 只增加一次诊断 predictor forward，
记录 clean-action candidate 的 loss，不决定是否接受 correction。guidance 关闭后，
原有 FastWAM 和 Best-of-K ranking 路径保持不变。

## 3. 当前核心配置

### 3.1 FastWAM

```yaml
model: FastWAMOptionalIDM
checkpoint: libero_optional_idm_2cam224.pt
dataset_stats: libero_optional_idm_2cam224_dataset_stats.json
action_infer_mode: idm
video_sigma_shift: 1.0
action_sigma_shift: 1.0
input_resolution: 224×224
action_shape: 32×7
proprioception: 8D
action_video_freq_ratio: 4
eval_num_inference_steps: 10
```

`sigma_shift` 是旧的联合 override；当前两阶段路径将其保持为 `null`，分别使用
`video_sigma_shift` 和 `action_sigma_shift`，避免一个参数同时覆盖两个 scheduler。
当前 LIBERO evaluation 显式使用 `1/1`，与原始 FastWAM README 的评测命令一致。
部分训练/模型 YAML 中仍可见 video scheduler 为 5、action scheduler 为 1；那是
底层 scheduler/训练配置，不是本仓库当前评测协议。

当前 action 的 7 个维度为 6D 末端执行器运动量和 1D 夹爪动作。

### 3.2 当前 LIBERO 默认配置

配置文件：[`configs/sim_libero.yaml`](./configs/sim_libero.yaml)

```yaml
EVALUATION.num_inference_steps: 10
EVALUATION.replan_steps: 10
EVALUATION.visualize_future_video: false
EVALUATION.action_denoise_trace: false
EVALUATION.offload_text_encoder: false
EVALUATION.compile_action_infer: true
EVALUATION.action_infer_mode: idm
EVALUATION.video_sigma_shift: 1.0
EVALUATION.action_sigma_shift: 1.0
EVALUATION.sigma_shift: null
```

启用 V-JEPA2-AC 时，当前实验协议使用：

```yaml
EVALUATION.vjepa2_ac.enabled: true
EVALUATION.vjepa2_ac.replan_steps: 8
EVALUATION.vjepa2_ac.candidate_generation_mode: separate
EVALUATION.vjepa2_ac.low_level_steps_per_ac_step: 4
EVALUATION.vjepa2_ac.dtype: float32
```

严格 shift-1 原始梯度 guidance 还需要：

```yaml
EVALUATION.replan_steps: 8
EVALUATION.num_inference_steps: 10
EVALUATION.video_sigma_shift: 1.0
EVALUATION.action_sigma_shift: 1.0
EVALUATION.sigma_shift: null
EVALUATION.vjepa2_ac.guidance.overlap_encoder_with_action: true
EVALUATION.vjepa2_ac.guidance.after_flow_steps: [7, 8, 9]
EVALUATION.vjepa2_ac.guidance.normalize_gradient: false
EVALUATION.vjepa2_ac.guidance.step_size: 1.0
EVALUATION.vjepa2_ac.guidance.max_delta_rms: 0.02
EVALUATION.vjepa2_ac.guidance.ac_steps: 2
EVALUATION.vjepa2_ac.guidance.verify_descent: true
```

### 3.3 Task-365 相机配置

当前 task-365 smoke 使用固定的高位、近距离 robot-left 视角：

```yaml
preset: robot_left_60_high
name: agentview
position: [0.3293, -0.5703643309324312, 1.6104]
quaternion: [0.8715703672034573, 0.4163865954915577,
             0.1115704520011075, 0.23353657603906355]
fovy: 45.0
image_rotation_degrees: 180
```

四元数使用 MuJoCo 的 `[w, x, y, z]` 顺序。

### 3.4 T5 / FastWAM / JEPA 显存时序

部署时建议先对所有 task prompt 做 T5 编码，将 embedding 缓存在 CPU，释放 T5，
再加载 FastWAM 和 V-JEPA2-AC。不要在 T5 仍驻留 GPU 时初始化 JEPA predictor，否则
24 GiB 卡可能在 predictor 初始化阶段 OOM。

```yaml
EVALUATION.offload_text_encoder: true
EVALUATION.text_encoder_batch_size: 8
```

## 4. Action denoising 收敛分析

打开配置：

```yaml
EVALUATION.action_denoise_trace: true
```

当前 scheduler 使用线性 flow-matching 插值：

```text
x_sigma = (1 - sigma) * a0 + sigma * epsilon
v       = epsilon - a0
sigma   = timestep / 1000
```

因此每个 flow step 的 clean-action estimate 为：

```text
a0_hat = x_sigma - sigma * v
```

记录内容包括：

- 每个 denoise step 的 `a0_hat`、timestep 和 tensor shape；
- 相邻 clean estimate 的 action change：`r_full`、`r_exec`；
- 当前 clean estimate 到最后一步 `a0_hat^(N)` 的距离：`d_full`、`d_exec`；
- 最终 flow action 和跨多个 closed-loop generation 的统计。

输出目录：

```text
<EVALUATION.output_dir>/<suite>/task<task_id>_gpu<gpu_id>/action_denoise_convergence/
```

输出文件：

```text
action_denoise_convergence.csv
action_denoise_convergence_summary.csv
action_denoise_convergence_estimates.npz
action_change_curve.png
distance_to_final_action.png
```

executed chunk 始终读取实际配置的 `replan_steps`，不会写死为 8 或 10。

## 5. 历史实验

历史评测元数据位于 [`artifacts/evaluate_results/`](./artifacts/evaluate_results/)。
其中保存 task-level JSON、summary CSV、manager YAML、日志和部分实验说明；原始
rollout MP4 已排除，以避免仓库膨胀。

所有主要实验的 policy seed 为 `7、17、27、37`。

### 5.1 Pose16 Best-of-8

LIBERO-Plus object pose task：

```text
1818, 1839, 1847, 1855,
2037, 2062, 2070, 2078,
2131, 2156, 2163, 2169,
2174, 2203, 2211, 2217
```

每个 task 使用 4 个 seed，共 64 个 rollout：

| 方法 | 成功数 | 成功率 |
|---|---:|---:|
| 原始 FastWAM | 35/64 | 54.7% |
| Best-of-8 JEPA | 37/64 | 57.8% |

主要正向案例是 Salad Dressing：`9/16 → 14/16`；主要负向案例是 Tomato Sauce：
`13/16 → 9/16`。

### 5.2 Long core3

固定 task：

| Task ID | 任务 |
|---:|---|
| 410 | 黑碗放入底层抽屉并关闭 |
| 662 | 黄白杯放入微波炉并关闭 |
| 457 | 白杯放左盘、黄白杯放右盘 |

每个 task 使用 4 个 seed，共 12 个 rollout。step `7/8/9` 的两段未来 guidance
取得 `5/12`，优于 action-only control 的 `1/12`。

### 5.3 Long8

```text
410, 457, 1945, 2046, 1027, 931, 145, 158
```

10-step control 与 1-step + JEPA guidance 均为 `14/32`。这轮同时改变了 denoise、
replan 和 action-generation protocol，不能视为严格单因素消融。

### 5.4 Long36

Long36 使用以下 36 个 LIBERO-Plus task：

```text
# 机器人初始姿态
365, 366, 410, 411, 567, 568, 289, 290, 330, 331, 457, 458

# 物体布局 / 额外干扰物
1945, 1933, 1959, 1960, 2046, 2047, 2059, 2060,
2093, 2094, 2127, 2128

# 背景纹理
0, 2, 33, 35, 137, 138, 145, 147, 158, 160, 193, 195
```

这是早期 unconditioned/联合 scheduler 协议，公共设置为
`num_inference_steps=10`、`sigma_shift=5.0`、`replan_steps=8`、
`visualize_future_video=true`、`offload_text_encoder=true`。它不是当前 Optional-IDM
严格级联的 `video_shift=1/action_shift=1` 协议，二者结果不能直接混合。

| 范围 | FastWAM control | step 7/8/9 guidance |
|---|---:|---:|
| 全部 36 tasks | 80/144（55.56%） | 88/144（61.11%） |
| 机器人初始姿态 | 19/48 | 22/48 |
| 物体布局 / 干扰物 | 39/48 | 44/48 |
| 背景纹理 | 22/48 | 22/48 |

task-365 在本轮 Long36 中为：control `1/4`，guidance `2/4`。

### 5.5 严格 shift-1 配对诊断

2026-09-22 的诊断集合从 Long36 的 144 个 task-seed pair 中选出64个历史失败案例，
使用 Optional-IDM 严格级联、video/action shift 均为1、10步denoise和`replan=8`。
固定pair清单保存在
[`strict_shift1_matched64_20260922.yaml`](./experiments/libero/task_lists/strict_shift1_matched64_20260922.yaml)。
同一64个pair上的结果为：

| 方法 | 成功数 | Rescue | Harm |
|---|---:|---:|---:|
| No guidance | 40/64 | — | — |
| 固定RMS 0.02 | 41/64 | 5 | 4 |

固定RMS候选的JEPA loss在 `9654/9939` 次 correction 中下降（97.13%），但净成功
只增加1例，说明 correction尺度必须单独校准，不能仅用candidate loss下降替代闭环结果。

随后仅在24个 no-guidance失败pair上测试原始梯度倍率1、最大RMS 0.02：得到
`3/24` rescue，分别为 `seed17/task195`、`seed27/task1960`、
`seed37/task411`。同一24个pair的固定RMS版本为`5/24`；原始梯度没有新增独有
rescue，但修正RMS中位数从固定的0.02降为0.00137。完整统计和边界条件见
[`RAW_GRADIENT_GUIDANCE_REPORT_20260922.md`](./experiments/libero/RAW_GRADIENT_GUIDANCE_REPORT_20260922.md)。

剩余40个 no-guidance成功pair用于衡量原始梯度的success retention/harm；在该评测
完成前，不应把`3/24`与`40/64`直接相加并报告为完整方法成功率。

## 6. 当前结论与限制

当前结果支持较谨慎的结论：

> 当 imagined future 能够保持物体身份和接触运动一致性时，JEPA-guided
> stochastic action selection 可能改善闭环控制；当参考未来出现接触幻觉时，
> 排序器也可能放大 FastWAM 的 world-model bias。

目前仍存在以下限制：

- 历史 Best-of-8 实验没有完整保存每次候选动作、energy、selected ID 和 top-2 margin；
- 8 条候选共享同一条 FastWAM imagined future，不是候选动作各自生成的因果 future；
- Pose16 的 FastWAM 与 JEPA 对照使用了不同的 replan 周期；
- 每个具体 task 通常只有 4 个 seed，样本量不足以支持强统计结论；
- JEPA 当前主要使用 agent-view，未把腕部视角加入 predictor 输入；
- PSNR 不能替代接触一致性或任务成功率。
- `verify_descent`验证的是clean-action candidate，不是写回普通scheduler后的完整
  flow trajectory；它是诊断量，不是任务成功保证。

## 7. 目录结构

```text
PACT-WAM/
├── checkpoints/                     # 本地统一权重入口；被 Git 忽略
│   ├── wan/                         # Wan2.2、T5、tokenizer
│   ├── fastwam/                     # Optional-IDM checkpoint 与 stats
│   └── vjepa2/                      # V-JEPA2-AC checkpoint
├── configs/                         # 训练、模型、LIBERO/RoboTwin 配置
├── src/fastwam/                     # FastWAM 核心实现
├── experiments/libero/              # LIBERO rollout、JEPA、worker 和分析代码
├── experiments/robotwin/            # RoboTwin rollout 代码
├── scripts/                         # 训练、预处理和部署脚本
├── tests/                           # 单元测试
├── third_party/RoboTwin/            # 随项目保留的 RoboTwin 代码
├── artifacts/evaluate_results/      # 历史 JSON/CSV/YAML/日志元数据
├── FASTWAM_JEPA_PROJECT_SUMMARY.md  # 完整项目与实验总结
└── PACT_WAM_PACKAGE_MANIFEST.md     # 打包范围说明
```

## 8. 安装

建议使用 Python 3.10 及 CUDA 12.8：

```bash
conda create -n pact-wam python=3.10 -y
conda activate pact-wam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

模型权重和 LIBERO/RoboTwin 数据集需要根据机器环境单独下载，不包含在本仓库中。

### 8.1 统一权重目录

所有运行配置只引用仓库内的 `checkpoints/` 入口。大文件可以实际存放在共享盘，
这里使用软链接，避免复制几十 GB 或破坏 FastWAM、JEPA_WAM 原项目：

```bash
mkdir -p checkpoints
ln -s /path/to/FastWAM/checkpoints checkpoints/wan
ln -s /path/to/FastWAM/checkpoints/fastwam_release checkpoints/fastwam
ln -s /path/to/JEPA_WAM/checkpoints/vjepa2 checkpoints/vjepa2
```

期望布局：

```text
checkpoints/
├── wan/Wan-AI/Wan2.2-TI2V-5B/
│   ├── Wan2.2_VAE.pth
│   └── models_t5_umt5-xxl-enc-bf16.pth
├── wan/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/
├── fastwam/
│   ├── libero_optional_idm_2cam224.pt
│   └── libero_optional_idm_2cam224_dataset_stats.json
└── vjepa2/vjepa2-ac-vitg.pt
```

加载前执行：

```bash
source scripts/local_libero_env.sh
```

该脚本会统一导出 `FASTWAM_CHECKPOINT`、`FASTWAM_DATASET_STATS`、
`VJEPA2_CHECKPOINT` 和 `DIFFSYNTH_MODEL_BASE_PATH`，并默认设置
`DIFFSYNTH_SKIP_DOWNLOAD=true`。当前配置使用本地原始 `Wan2.2_VAE.pth`，即
`model.redirect_common_files=false`，不会自动转向 DiffSynth converted checkpoint。

## 9. LIBERO 运行示例

Optional-IDM 两阶段评测：

```bash
source scripts/local_libero_env.sh
python experiments/libero/run_libero_manager.py \
  task=libero_optional_idm_2cam224_1e-4 \
  ckpt="$FASTWAM_CHECKPOINT" \
  EVALUATION.dataset_stats_path="$FASTWAM_DATASET_STATS" \
  EVALUATION.action_infer_mode=idm \
  EVALUATION.video_sigma_shift=1.0 \
  EVALUATION.action_sigma_shift=1.0 \
  EVALUATION.sigma_shift=null \
  EVALUATION.visualize_future_video=true \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=50
```

启用 action denoise 收敛记录：

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_optional_idm_2cam224_1e-4 \
  ckpt="$FASTWAM_CHECKPOINT" \
  EVALUATION.action_denoise_trace=true \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0
```

启用 V-JEPA2-AC guidance 时，还需要准备 predictor checkpoint，并设置：

```yaml
EVALUATION.visualize_future_video: true
EVALUATION.vjepa2_ac.enabled: true
EVALUATION.vjepa2_ac.checkpoint_path: /path/to/vjepa2-ac-vitg.pt
EVALUATION.vjepa2_ac.repo_path: /path/to/vjepa2
EVALUATION.vjepa2_ac.guidance.enabled: true
EVALUATION.vjepa2_ac.guidance.after_flow_steps: [7, 8, 9]
```

原始梯度、最大RMS 0.02 的单seed可复现入口：

```bash
nohup env \
  GPU_ID=6 \
  SEED=7 \
  TASK_FILE=/absolute/path/to/tasks.txt \
  OUTPUT_ROOT=/absolute/path/to/output \
  bash scripts/run_jepa_guidance_raw_gradient.sh \
  > /absolute/path/to/output.nohup.log 2>&1 < /dev/null &
```

脚本固定使用 Optional-IDM checkpoint、严格 `infer_video() → infer_action_from_video()`、
video/action shift `1/1`、denoise 10步、`replan=8`和guidance step `7/8/9`。

### 9.1 Smoke test 基线

提交前至少验证：模型类型为 `FastWAMOptionalIDM`，`infer_video()` 不产生 action，
`infer_action_from_video()` 复用完全相同的 video latents，并输出 `32×7` action。
当前服务器在 RTX 4090D 上用 1 个 video step 和 1 个 action step 的结果为：

```text
model: FastWAMOptionalIDM
video_sigma_shift: 1.0
action_sigma_shift: 1.0
video_latents: [1, 48, 3, 14, 28]
action: [32, 7]
peak_allocated: 12.698 GiB
status: PASS
```

完整 LIBERO smoke 建议使用单 task、单 trial、`max_steps_override=1`，同时保持
`EVALUATION.visualize_future_video=true`，以确保评测器实际进入拆分后的两阶段路径。

### 9.2 Offline trigger probe

第一阶段 trigger 实验只做测量，不改变执行策略：Optional-IDM 先生成完整 WAM
future，再生成一个 action chunk；ActionDiT 每个 denoise step、每层额外记录
observation / imagined-future / action-token 三部分 attention response，并保存
`R_future`。同一 planning point 的最终 action 再经过冻结的 V-JEPA2-AC predictor，
得到一个 JEPA consistency loss。此模式不启用 Best-of-K 或 guidance。

```yaml
EVALUATION.visualize_future_video: true
EVALUATION.compile_action_infer: false
EVALUATION.video_sigma_shift: 1.0
EVALUATION.action_sigma_shift: 1.0
EVALUATION.sigma_shift: null
EVALUATION.vjepa2_ac.enabled: false
EVALUATION.vjepa2_ac.guidance.enabled: false
EVALUATION.trigger_probe.enabled: true
EVALUATION.trigger_probe.ac_steps: 8
```

这里的 `1/1` 对应原始 FastWAM README 的 LIBERO evaluation 设置；模型训练配置中
出现的 `video=5/action=1` 不应与该评测设置混淆。完整 9 帧 WAM clip 用于 8-step
JEPA verifier，而 `replan_steps=8` 仍只决定闭环实际执行多少个低层动作，两者彼此独立。

任务完成后执行离线统计：

```bash
python experiments/libero/analyze_trigger_probe.py /path/to/results
```

脚本输出每个 denoise step 的 Pearson / Spearman 相关性、低中高 `R_future` 分组，
以及“最低 20% response 召回最高 20% JEPA loss”的 recall 和 JEPA call rate。

## 10. 相关文档与引用

- [完整项目与实验总结](./FASTWAM_JEPA_PROJECT_SUMMARY.md)
- [实验进度记录](./FASTWAM_JEPA_PROGRESS.md)
- [原始梯度 guidance 报告](./experiments/libero/RAW_GRADIENT_GUIDANCE_REPORT_20260922.md)
- [LIBERO 本地部署说明](./LOCAL_LIBERO_DEPLOYMENT.md)
- [打包范围说明](./PACT_WAM_PACKAGE_MANIFEST.md)
- [Fast-WAM 原始项目页](https://yuantianyuan01.github.io/FastWAM/)

本项目继承 FastWAM 的代码和实验基础。使用 FastWAM 原始方法、模型或数据时，
请同时遵循上游仓库的许可证和引用要求。
