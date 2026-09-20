# PACT-WAM

PACT-WAM 是一个面向机器人动作生成与闭环控制研究的实验代码包。项目以
[Fast-WAM](https://github.com/yuantianyuan01/FastWAM) 为基础，加入了
V-JEPA2-AC 动作排序、action-flow guidance、LIBERO-Plus 扰动评测，以及
denoising/flow-matching 动作收敛分析。

这个仓库保留了当前研究阶段的完整快照：源码、配置、实验脚本、历史实验报告和
轻量级评测 JSON/CSV/YAML 都保存在仓库中。模型权重、rollout 视频和本地缓存不上传 GitHub。

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d14836.svg)](./README_zh.md)

## 1. 项目目标

PACT-WAM 研究以下问题：

- FastWAM 的随机 action flow 在不同 denoising step 中如何收敛到可执行动作；
- 冻结的 V-JEPA2-AC 是否可以在闭环 rollout 中筛选或修正 FastWAM 动作；
- imagined future 质量、动作候选覆盖范围和 JEPA energy 如何共同影响控制成功率；
- 当 imagined future 出现错误接触或物体幻觉时，JEPA 排序是否会放大这种偏差。

项目目前只在推理阶段组合 FastWAM 和 V-JEPA2-AC，没有对两者进行联合再训练。

## 2. 系统流程

### 2.1 普通 FastWAM

```text
LIBERO 双相机观测 + 任务文本 + proprioception
                      ↓
              FastWAM flow-matching
                      ↓
        未来视频 + 32×7 action chunk
                      ↓
              执行前 N 个动作
                      ↓
              重新观测并规划
```

### 2.2 Best-of-K JEPA ranking

当前推荐的 `separate` 路径为：

```text
infer_joint(K=1)  → 生成共同的 FastWAM imagined future
infer_action(K=8) → 生成 8 条动作候选
V-JEPA2-AC        → 计算每条候选的未来 latent energy
选择最低 energy   → 执行选中候选的前 N 个动作
```

每 4 个 LIBERO 低层动作对应 1 个 JEPA action-conditioning step。当前 JEPA
实验执行 8 个低层动作，即使用两个未来转移进行排序。

### 2.3 Action-flow guidance

guidance 在动作 flow 的指定边界上计算 JEPA loss 对动作的梯度，并在归一化后
对 FastWAM 的 action flow state 做 additive correction。当前主实验使用：

```yaml
guidance.enabled: true
guidance.after_flow_steps: [7, 8, 9]
guidance.step_size: 0.02
guidance.ac_steps: 2
```

guidance 是推理时的附加模块；关闭后，原有 FastWAM 和 Best-of-K ranking 路径保持不变。

## 3. 当前核心配置

### 3.1 FastWAM

```yaml
checkpoint: libero_uncond_2cam224.pt
dataset_stats: libero_uncond_2cam224_dataset_stats.json
input_resolution: 224×224
action_shape: 32×7
proprioception: 8D
action_video_freq_ratio: 4
eval_num_inference_steps: 10
```

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
```

启用 V-JEPA2-AC 时，当前实验协议使用：

```yaml
EVALUATION.vjepa2_ac.enabled: true
EVALUATION.vjepa2_ac.replan_steps: 8
EVALUATION.vjepa2_ac.candidate_generation_mode: separate
EVALUATION.vjepa2_ac.low_level_steps_per_ac_step: 4
EVALUATION.vjepa2_ac.dtype: float32
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

共同设置：`num_inference_steps=10`、`sigma_shift=5.0`、`replan_steps=8`、
`visualize_future_video=true`、`offload_text_encoder=true`。

| 范围 | FastWAM control | step 7/8/9 guidance |
|---|---:|---:|
| 全部 36 tasks | 80/144（55.56%） | 88/144（61.11%） |
| 机器人初始姿态 | 19/48 | 22/48 |
| 物体布局 / 干扰物 | 39/48 | 44/48 |
| 背景纹理 | 22/48 | 22/48 |

task-365 在本轮 Long36 中为：control `1/4`，guidance `2/4`。

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

## 7. 目录结构

```text
PACT-WAM/
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

## 9. LIBERO 运行示例

普通 FastWAM 评测：

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=50
```

启用 action denoise 收敛记录：

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
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

## 10. 相关文档与引用

- [完整项目与实验总结](./FASTWAM_JEPA_PROJECT_SUMMARY.md)
- [实验进度记录](./FASTWAM_JEPA_PROGRESS.md)
- [LIBERO 本地部署说明](./LOCAL_LIBERO_DEPLOYMENT.md)
- [打包范围说明](./PACT_WAM_PACKAGE_MANIFEST.md)
- [Fast-WAM 原始项目页](https://yuantianyuan01.github.io/FastWAM/)

本项目继承 FastWAM 的代码和实验基础。使用 FastWAM 原始方法、模型或数据时，
请同时遵循上游仓库的许可证和引用要求。
