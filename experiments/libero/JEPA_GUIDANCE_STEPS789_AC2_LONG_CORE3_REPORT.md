# JEPA Action-Flow Guidance：LIBERO-Plus Long Core3 简报

## 实验设置

- 任务：`libero_10` 的 Plus task 410、662、457。
- 固定 seeds：7、17、27、37，每个任务运行 4 次，共 12 个 episode。
- FastWAM checkpoint、dataset statistics、任务初始状态、`sigma_shift=5.0`、未来视频生成和 `replan_steps=8` 均与历史对照一致。
- 当前方法：在 action-flow step 7、8、9 加入 JEPA 梯度，`ac_steps=2`，`step_size=0.02`，关闭仅用于诊断的 `verify_descent`。
- 每次只生成一条 action trajectory，不进行 Best-of-8 候选排序；执行 8 个低层动作后重新规划。

## 时间对齐与 Loss

FastWAM 在一次 8-action replan 中提供起始锚点 `f0` 和两张未来帧 `f1`、`f2`。JEPA encoder 每次 replan 只运行一次，批量编码：

```text
current = Encoder([obs_prev, obs_now])
target1 = Encoder([f0, f1])   # 对齐 action 0–3
target2 = Encoder([f1, f2])   # 对齐 action 4–7
```

Predictor 自回归预测两个 transition，使用等权 latent L1 loss：

```text
loss = 0.5 * L1(pred1, target1) + 0.5 * L1(pred2, target2)
```

step 7、8、9 各计算一次上述 loss，因此每次 replan 共执行 6 次 predictor forward。

上一轮 `last_flow_steps=2, ac_steps=1` 实际在 flow step 8、9 引导，只对齐 `[f0,f1]` 与 action 0–3；`f2` 和 action 4–7 未进入 guidance loss。

## 成功率

| 任务 | seed 7 | seed 17 | seed 27 | seed 37 | 当前方法 |
|---|---:|---:|---:|---:|---:|
| 410 | ✓ | ✗ | ✓ | ✓ | **3/4（75%）** |
| 662 | ✗ | ✗ | ✗ | ✗ | **0/4（0%）** |
| 457 | ✓ | ✗ | ✗ | ✓ | **2/4（50%）** |
| **总计** | 2/3 | 0/3 | 1/3 | 2/3 | **5/12（41.7%）** |

## 方法对比

| 方法 | 410 | 662 | 457 | 总计 |
|---|---:|---:|---:|---:|
| 对齐 K=1 baseline | 2/4 | 0/4 | 0/4 | **2/12（16.7%）** |
| step 8/9、单段 guidance | 1/4 | 0/4 | 2/4 | **3/12（25.0%）** |
| 历史 separate K=8 | 3/4 | 0/4 | 1/4 | **4/12（33.3%）** |
| **step 7/8/9、两段 guidance** | **3/4** | **0/4** | **2/4** | **5/12（41.7%）** |

相较上一轮单段 guidance，当前方法出现 3 次 rescue（410/seed7、410/seed37、457/seed37）和 1 次 regression（457/seed17），净增加 2 次成功。

## 结论

step 7/8/9 加两段未来对齐取得当前四种方案中的最高成功率，说明将 `f2` 和后四个动作纳入 JEPA loss 是积极方向。任务 662 仍为 0/4 的地板任务；总样本只有 12 个 episode，因此结果应视为 stress slice 上的积极信号，而不是统计稳定的整体性能结论。

原始结果目录：[`evaluate_results/jepa_guidance_steps789_ac2_long_core3_20260915_192558`](../../evaluate_results/jepa_guidance_steps789_ac2_long_core3_20260915_192558/)

