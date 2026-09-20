# FastWAM–V-JEPA2-AC 项目与实验总览

> 服务器迁移与实验设置核对版
>
> 更新时间：2026-09-20

本文总结当前工作区中的 FastWAM–V-JEPA2-AC 项目、已经完成的实验、具体 task ID、主要结果和迁移到另一台服务器时需要保持的配置。

## 1. 项目目标

项目在发布版 FastWAM 的 LIBERO 闭环控制中加入冻结的 V-JEPA2-AC，用于判断或修正 FastWAM 生成的动作是否会产生合理的短期视觉未来。

基本数据流：

```text
LIBERO 双相机观测 + 任务文本 + proprioception
        ↓
FastWAM 生成未来视频和 32×7 动作 chunk
        ↓
V-JEPA2-AC 对动作对应的未来进行评价或提供梯度
        ↓
选择/修正动作
        ↓
执行前 8 个低层动作
        ↓
重新观测并重新规划
```

FastWAM 和 V-JEPA2-AC 都是推理阶段使用，当前项目没有对两者进行再训练。

## 2. FastWAM 主干

### 2.1 模型和输入

- FastWAM checkpoint：`libero_uncond_2cam224.pt`
- 数据统计：`libero_uncond_2cam224_dataset_stats.json`
- 两路 LIBERO 相机输入，训练/评估分辨率为 `224×224`
- T5 任务文本编码
- 8 维 proprioception
- 动作输出为 `32×7`：6 维末端执行器运动 + 1 维夹爪
- 视频和动作时间比例为 `4:1`
- FastWAM 视频 DiT：30 层，hidden size 3072
- FastWAM 动作 DiT：30 层，hidden size 1024

一次推理会产生一段未来视频和一段动作。Long Horizon 实验通常执行前 8 个低层动作，然后重新规划；`libero_10` 环境最多允许 700 个策略动作步。

### 2.2 FastWAM 推理路径

没有 JEPA 时：

```text
infer_joint → 直接执行生成的动作
```

Best-of-8 JEPA 时，当前推荐的 `separate` 路径为：

```text
infer_joint(K=1)  → 生成 FastWAM 未来视频参考
infer_action(K=8) → 生成 8 条动作候选
JEPA 排序         → 选择 energy 最低的候选
```

`joint` 路径把视频和 8 条动作放在一次 joint flow 中生成，延迟较低，但在 Long core3 上没有保持旧 two-call 路径的行为，因此目前只作为实验选项。

## 3. V-JEPA2-AC

### 3.1 模型

- 视觉 encoder：冻结的 ViT-G
- action-conditioned predictor：冻结
- checkpoint：`vjepa2-ac-vitg.pt`
- 推理 dtype：`float32`
- 每 4 个 LIBERO 低层动作对应 1 个 JEPA action-conditioning step

执行 8 个低层动作时，JEPA 对齐两段未来：

```text
动作 0–3 → 目标未来 f0→f1
动作 4–7 → 目标未来 f1→f2
```

能量函数为两个未来转移的归一化 latent L1 平均：

```text
E(a) = 0.5 × L1(pred1, target1)
     + 0.5 × L1(pred2, target2)
```

其中目标未来通常来自 FastWAM 自己预测的视频，不是环境真实未来。因此，如果 FastWAM 未来产生错误的抓取或物体运动，JEPA 可能会放大这个错误。

### 3.2 两种使用方式

| 路线 | 机制 | 当前定位 |
|---|---|---|
| Best-of-8 ranking | 8 条独立噪声动作，经 JEPA energy 排序 | 已完成历史实验；`separate` 保留为兼容路径 |
| Action-flow guidance | 对单条动作 flow 在指定步骤施加 JEPA 梯度 | 当前主线，最佳结果为 flow step 7/8/9 |

当前主线 guidance 的动作更新是近似梯度引导：先由当前 flow 状态和 FastWAM 速度估计 clean action，再计算 JEPA loss 对动作的梯度，截取执行范围内的动作并做 RMS 归一化，最后以 `step_size=0.02` 修正。

## 4. 已完成实验总览

所有实验中的 policy/inference seed 主要使用：`7、17、27、37`。

| 实验 | 具体 task ID | 主要设置 | 结果 |
|---|---|---|---|
| Pose16 | 1818, 1839, 1847, 1855, 2037, 2062, 2070, 2078, 2131, 2156, 2163, 2169, 2174, 2203, 2211, 2217 | 原始 FastWAM 对比 Best-of-8 JEPA | `35/64 → 37/64` |
| Long core3 | 410, 662, 457 | 多组 K=1、Best-of-8、guidance 对照 | 789 guidance 最好为 `5/12` |
| Long8 | 410, 457, 1945, 2046, 1027, 931, 145, 158 | 10-step control 对比 1-step + JEPA | `14/32 → 14/32` |
| Long36 | 36 个 LIBERO-Plus task ID | 10-step control 对比 789 guidance | `80/144 → 88/144` |

### 4.1 Pose16：Best-of-8 早期验证

任务清单：

```text
libero_object:
1818, 1839, 1847, 1855,
2037, 2062, 2070, 2078,
2131, 2156, 2163, 2169,
2174, 2203, 2211, 2217
```

每个 task 使用 4 个 seed，共 64 个 rollout。

| 方法 | 成功率 |
|---|---:|
| 原始 FastWAM | 35/64（54.7%） |
| Best-of-8 JEPA | 37/64（57.8%） |

配对结果为 7 次 rescue、5 次 regression。主要正向案例是 Salad Dressing：`9/16 → 14/16`；主要负向案例是 Tomato Sauce：`13/16 → 9/16`。

这轮实验不是严格的 selector 消融，因为原始 FastWAM 使用 `replan=10`，JEPA 使用 `replan=8`，且未来视频开关不同。

重点 task：

- Salad Dressing：`2131, 2156, 2163, 2169`
- Tomato Sauce：`2174, 2203, 2211, 2217`
- 后续 denoise/guidance 探索重点：`2169, 2211`

### 4.2 Long core3

固定 task：

| Plus task ID | 任务 |
|---:|---|
| 410 | 黑碗放入底层抽屉并关闭 |
| 662 | 黄白杯放入微波炉并关闭 |
| 457 | 白杯放左盘、黄白杯放右盘 |

每个 task 使用 4 个 seed，共 12 个 rollout。

| 方法 | 410 | 662 | 457 | 总计 |
|---|---:|---:|---:|---:|
| 原始 action-only，future video 关闭，replan=10 | 1/4 | 0/4 | 0/4 | 1/12 |
| 对齐 K=1，future video 开启，replan=8 | 2/4 | 0/4 | 0/4 | 2/12 |
| 历史 separate Best-of-8 | 3/4 | 0/4 | 1/4 | 4/12 |
| joint K=8 | 1/4 | 0/4 | 0/4 | 1/12 |
| step 7/8/9、两段未来 guidance | 3/4 | 0/4 | 2/4 | 5/12 |

这个切片中 task662 始终是 `0/4`，因此只能作为 stress test，不能单独代表总体性能。

### 4.3 Long8

固定 task：

| Plus task ID | 扰动类别 |
|---:|---|
| 410, 457 | 机器人初始姿态 |
| 1945, 2046 | 物体布局/额外干扰物 |
| 1027, 931 | 相机视角 |
| 145, 158 | 背景纹理 |

共 8 tasks × 4 seeds × 2 方法 = 64 episodes。

| 方法 | 成功率 |
|---|---:|
| 10-step FastWAM control | 14/32（43.75%） |
| 1-step + JEPA guidance | 14/32（43.75%） |

这轮同时改变了 denoise 步数、动作生成路径和 guidance，属于系统级比较，不是严格的 JEPA 单因素消融。两个相机扰动 task `1027、931` 都是 `0/4`，因此没有纳入后续 Long36 主集合。

### 4.4 Long36：当前主结果

Long36 使用 6 个原始 LIBERO-10 目标，每个目标选择三类扰动、每类两个 Plus 变体：

#### 机器人初始姿态

```text
原始目标 0: 365, 366
原始目标 1: 410, 411
原始目标 4: 567, 568
原始目标 5: 289, 290
原始目标 6: 330, 331
原始目标 7: 457, 458
```

#### 物体布局/额外干扰物

```text
原始目标 0: 1945, 1933
原始目标 1: 1959, 1960
原始目标 4: 2046, 2047
原始目标 5: 2059, 2060
原始目标 6: 2093, 2094
原始目标 7: 2127, 2128
```

#### 背景纹理

```text
原始目标 0: 0, 2
原始目标 1: 33, 35
原始目标 4: 137, 138
原始目标 5: 145, 147
原始目标 6: 158, 160
原始目标 7: 193, 195
```

每组为 36 tasks × 4 seeds = 144 episodes。

共同设置：

```yaml
num_inference_steps: 10
sigma_shift: 5.0
replan_steps: 8
visualize_future_video: true
offload_text_encoder: true
timing_enabled: true
```

对照组关闭 JEPA；实验组使用 `step 7/8/9` 的两段未来 guidance。

| 范围 | 对照 | 789 guidance |
|---|---:|---:|
| 全部 36 tasks | 80/144（55.56%） | 88/144（61.11%） |
| 机器人初始姿态 | 19/48 | 22/48 |
| 物体布局/干扰物 | 39/48 | 44/48 |
| 背景纹理 | 22/48 | 22/48 |

配对结果：12 次 rescue、4 次 regression、76 对结果不变。

按原始目标：

| 原始目标 | 对照 | 789 guidance |
|---:|---:|---:|
| 0：炉灶与 moka pot | 12/24 | 13/24 |
| 1：抽屉关闭 | 8/24 | 11/24 |
| 4：两个物体放入篮子 | 24/24 | 24/24 |
| 5：汤罐和番茄酱放入篮子 | 16/24 | 16/24 |
| 6：奶酪盒和黄油放入篮子 | 20/24 | 18/24 |
| 7：两个杯子分放左右盘 | 0/24 | 6/24 |

task365 在本轮 Long36 中的结果是：

```text
对照：1/4
789 guidance：2/4
```

## 5. 当前推荐主线配置

如果在另一台服务器继续主实验，建议使用 Long36 的 789 guidance 协议：

```yaml
task: libero_uncond_2cam224_1e-4
EVALUATION.num_trials: 1
EVALUATION.num_inference_steps: 10
EVALUATION.sigma_shift: 5.0
EVALUATION.replan_steps: 8
EVALUATION.visualize_future_video: true
EVALUATION.offload_text_encoder: true
EVALUATION.timing_enabled: true

EVALUATION.vjepa2_ac.enabled: true
EVALUATION.vjepa2_ac.candidate_generation_mode: separate
EVALUATION.vjepa2_ac.replan_steps: 8
EVALUATION.vjepa2_ac.guidance.enabled: true
EVALUATION.vjepa2_ac.guidance.after_flow_steps: null
EVALUATION.vjepa2_ac.guidance.last_flow_steps: 3
EVALUATION.vjepa2_ac.guidance.ac_steps: 2
EVALUATION.vjepa2_ac.guidance.step_size: 0.02
EVALUATION.vjepa2_ac.guidance.verify_descent: false
EVALUATION.vjepa2_ac.dtype: float32
```

当前 YAML 的默认值不是这套主实验配置：默认关闭 JEPA、`replan_steps=10`、`sigma_shift=null`、关闭未来视频。因此必须使用启动脚本或显式 Hydra overrides。

## 6. task365 新相机实验

task365 属于 LIBERO-Plus 的 `libero_10`：

```yaml
EVALUATION.task_suite_name: libero_10
EVALUATION.task_id: 365
EVALUATION.num_trials: 50
```

任务语言：

```text
turn on the stove and put the moka pot on it
```

新相机配置：

```yaml
preset: robot_left_60_high
name: agentview
position: [0.3293, -0.5703643309324312, 1.6104]
quaternion: [0.8715703672034573, 0.4163865954915577,
             0.1115704520011075, 0.23353657603906355]
fovy: 45.0
image_rotation_degrees: 180
```

这个相机是高位、靠近机器人左侧的视角。MuJoCo quaternion 使用 `[w,x,y,z]` 顺序。我们已经完成以下代码改造：

- 每次环境 reset 后重新写入 `agentview` 的位置、四元数和 fovy；
- 对 agentview 和 wrist 图像支持显式的 90°整数倍旋转；
- 保持 `180°` 时与原 FastWAM 历史图像翻转行为一致。

历史 Pose16、Long8、Long36 结果没有使用这套新相机。task365 的新相机 50-trial 评估此前只完成了配置预检，没有生成最终结果。

## 7. 迁移到另一台服务器

当前项目依赖本地未提交改动，不能只重新 clone 原始 FastWAM。建议整体复制当前工作树，或先整理成一个 commit/patch。

### 7.1 代码和环境

```text
/data/chenpengxu/FastWAM
/data/chenpengxu/LIBERO-plus
/data/chenpengxu/JEPA_WAM/.libero_plus_config
/data/chenpengxu/vjepa2
/data/chenpengxu/jepa_wam_py312
/data/chenpengxu/conda_envs/HMoE
```

运行时需要将路径替换成新服务器路径：

```bash
export PYTHONPATH=/new/LIBERO-plus:/new/FastWAM/src:/new/jepa_wam_py312
export LIBERO_CONFIG_PATH=/new/JEPA_WAM/.libero_plus_config
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 7.2 模型和资源

```text
FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt
FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
FastWAM/checkpoints/wan/
jepa_wam_assets/vjepa2-ac-vitg.pt
```

当前资源大小约为：

- FastWAM checkpoint：约 12.0 GB；
- V-JEPA2-AC checkpoint：约 11.8 GB。

### 7.3 GPU 运行方式

单个 worker 使用一张可见 GPU。两张 GPU 的正确用法是启动两个独立 worker，例如分别运行两个 seed；不是让一个 `eval_libero_single.py` worker 自动跨两张卡。

T5 加载约需要 10.6 GiB 峰值显存，因此多 worker 启动时需要错开 T5 加载。T5 释放后，FastWAM + JEPA 通常约 17.5–18.6 GiB，24 GiB GPU 可以运行。

## 8. 当前已知问题

1. 历史 Best-of-8 实验没有保存每次 replan 的 8 条动作、energy、selected candidate ID 和 top-two margin。
2. 8 条候选共享一条 FastWAM 参考未来，不是每条候选分别生成 action-conditioned future。
3. FastWAM 预测错误接触时，JEPA selector 可能放大错误参考。
4. 目前具体 Plus task 通常只有 4 个 seed，task-level 结果适合作为探索性证据。
5. Long36 的提升主要来自部分任务，不能直接宣称对所有 LIBERO-Plus 任务稳定提升。
6. 新的 `robot_left_60_high` 相机只应用于后续新实验，不能与旧实验结果直接混合。

## 9. 关键文件

- 总配置：[`configs/sim_libero.yaml`](configs/sim_libero.yaml)
- LIBERO 单进程评估：[`experiments/libero/eval_libero_single.py`](experiments/libero/eval_libero_single.py)
- JEPA ranker 和动作适配器：[`experiments/libero/vjepa2_ac_ranker.py`](experiments/libero/vjepa2_ac_ranker.py)
- 多 GPU worker：[`experiments/libero/run_libero_manager.py`](experiments/libero/run_libero_manager.py)
- Long36 主实验脚本：[`scripts/run_long36_ten_vs_789_2gpu.sh`](scripts/run_long36_ten_vs_789_2gpu.sh)
- Long36 task list：[`experiments/libero/task_lists/libero_plus_long_36_three_perturbations_two_variants.txt`](experiments/libero/task_lists/libero_plus_long_36_three_perturbations_two_variants.txt)
- 时间线报告：[`experiments/libero/FASTWAM_JEPA_CHRONOLOGICAL_EXPERIMENT_REPORT_20260917.md`](experiments/libero/FASTWAM_JEPA_CHRONOLOGICAL_EXPERIMENT_REPORT_20260917.md)
- JEPA 进度报告：[`FASTWAM_JEPA_PROGRESS.md`](FASTWAM_JEPA_PROGRESS.md)

## 10. 迁移前需要最终确认

当前建议的迁移主线是：

```text
LIBERO-Plus Long36
10-step FastWAM
replan=8
sigma_shift=5.0
789 JEPA action-flow guidance
四个 seed：7, 17, 27, 37
两张 GPU 并行两个 worker
```

task365 的 50-trial、`robot_left_60_high` 相机实验作为独立的新实验，不与历史 Long36 结果混合统计。
