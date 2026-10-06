# PACT-WAM

基于 [FastWAM](https://github.com/yuantianyuan01/FastWAM) 和 V-JEPA2-AC 的机器人动作修正研究代码。仓库包含 LIBERO / LIBERO-Plus 评测、JEPA guidance 与 trigger 诊断，以及当前的 **latent-action IDM + residual Transformer** 训练实验。

> 更新于 2026-10-06。当前重点是监督学习动作修正模块，不是继续优化梯度 guidance。IDM 与 Transformer 的最新结果来自离线演示窗口验证，**尚不能作为闭环成功率提升的证据**。FastWAM 本体和 JEPA encoder 保持冻结。

## 当前路线

1. FastWAM 生成 imagined future 和原始 action chunk。
2. 冻结 JEPA encoder 保留主视角全部空间 tokens。
3. 共享视觉 IDM 将两个相邻时间段编码为 `LA1`、`LA2`。
4. Residual Transformer 读取原动作与 LA，输出前 8 步动作的修正量。

```text
观测 + 文本 + proprioception
          │
          ▼
    冻结 FastWAM ─────────────► raw action 的前 8×7 步
          │                               │
          ▼                               │
   生成 future frames                     │
          │                               │
          ▼                               │
冻结 JEPA encoder → Z0, Z1, Z2             │
          │                               │
          ▼                               ▼
  IDM(Z0,Z1), IDM(Z1,Z2) → LA1, LA2 → Residual Transformer
                                            │
                                            ▼
                             a'[:, :6] = a[:, :6] + Δa
                             a'[:, 6]  = a[:, 6]
```

这是当前研究模块的链路；现有训练脚本在缓存上验证，尚未把最新 corrector 接入闭环评测。不要把 **FastWAM 发布权重中的 Optional-IDM** 与这里新训练的 **21.28M 视觉 latent-action IDM** 混为一谈。

### 模型与监督

| 模块 | 结构 / 参数量 | 训练方式 |
|---|---|---|
| JEPA encoder | 每个 clip `256×1408`，保留空间 tokens | 冻结；主视角，不加入 wrist |
| 视觉 IDM | 1408→512 token 投影；空间/时间位置编码；motion query cross-attention；6 层 Transformer，8 heads，FFN 2048；输出 128D LA | 本体 **21.28M**；加动作解码头 **21.63M** |
| IDM 解码头 | 128→512→512→24 | 每个 LA 监督对应 4×6 演示动作；IDM 不接收 raw action |
| Residual Transformer | 8 action tokens + 2 LA tokens；4 层，256D，8 heads，FFN 1024；self-attention + gated cross-attention | **4.25M**；监督前 8×6 演示动作，夹爪不修改 |

IDM 首先使用真实 future 训练；随后冻结 IDM，使用 **WAM future 提取的 LA** 训练 Transformer。真实 future oracle 仅用于诊断，在线推理不能读取真实未来。当前修正模块使用 FP32；训练不启用 RMS cap 或残差惩罚，输出头零初始化。历史 guidance 的 `RMS ≤ 0.02` 不适用于这里的训练。

## 数据与坐标口径

- 训练数据为原始 `libero_10`：10 个任务、500 条演示。**不是** LIBERO-Plus 的 2,519 条 long-horizon 测试场景。
- 按轨迹隔离：`demo_index % 10 == 0` 为验证，共 450 条训练轨迹 / 50 条验证轨迹；同一轨迹的窗口不会跨 split。
- 当前 `stride=32`：**3,871 个训练窗口 / 436 个验证窗口，共 4,307 个窗口**。更多窗口不等于更多独立轨迹。
- 当前 clip 使用 `[t-8,t]`；两个 future clip 使用 `[t,t+4]`、`[t+4,t+8]`；标签为 `[t:t+8]`，分成两个 4 步段。
- 7D 标签是可执行的 6D 末端动作命令 + 夹爪。动作标签不等于 `ee_states`：后者的姿态是轴角 / rotvec，进入 AC state 适配时必须显式转换为 Euler，不能直接当作 Euler。
- 当前缓存沿用 `rot180` 契约：真实主视角先旋转 180°，与 FastWAM 输入和生成 future 方向一致。双相机生成帧先取左侧 agentview；额外旋转只应用到所选主视角，避免交换 agent / wrist。
- `encoder_image_rotation_degrees` 是提取主视角后的额外旋转，不是所有实验通用的“正视/倒置”开关。以该实验 config 和数据处理路径为准，不要重复旋转。

## 已核对的最新结果

以下均为**归一化前 8×6 动作的离线验证 MSE，越低越好**。IDM 解码与 Transformer 修正是不同任务；450/50 与全量窗口是不同验证集合，不应跨表直接比较绝对误差。

### 全量 stride-32：Transformer 修正

| 模型 / 输入 | 验证 MSE | 最佳 epoch |
|---|---:|---:|
| raw action，不修正 | 0.01379743 | — |
| Transformer，无 LA 对照 | 0.01326274 | 8 |
| Transformer + 旧 IDM 的 WAM LA | 0.01327799 | 31 |
| **Transformer + 新 IDM（epoch 29）的 WAM LA** | **0.01314435** | **8** |

新 IDM 版本比匹配的无 LA 对照降低约 **0.89%**，比 raw action 降低约 **4.73%**。固定新模型后打乱 LA，误差升到 **0.01471988**；交换 LA1/LA2 后为 **0.01359281**，说明模型使用了未来信息，但收益目前很小。单 seed 结果、验证集 checkpoint 选择及轨迹相关性都限制结论；需要独立测试与闭环验证。

### 全量 stride-32：真实帧 IDM 训练

加载旧 IDM，继续训练 30 epoch / 14,520 step，无早停。下表统一使用真实 future 验证最优的 **epoch 29** 权重：

| IDM 输入 | 训练前验证 MSE | epoch 29 验证 MSE |
|---|---:|---:|
| 真实 future | 0.02384369 | 0.01332424 |
| WAM future，仅作验证 | 0.02858109 | 0.02057321 |

真实帧上的改善部分迁移到了生成帧，但生成帧误差仍较高。不能仅由 latent 差异断言 WAM 画面生成质量差：运动预测偏差与真实/生成特征分布差异尚未分离。

### 450/50 单窗口 pilot（历史诊断）

| 模型 | 无 LA | WAM LA |
|---|---:|---:|
| MLP，约 0.45M | 0.01556739 | 0.01567051 |
| Transformer，4.25M | 0.01583955 | 0.01519595 |

pilot 的 LA 优势未直接延续到使用旧 IDM 的全量窗口实验。这也是先扩大 IDM 训练、再冻结重训 Transformer 的动机。

## 代码导航

| 入口 | 用途 |
|---|---|
| [`scripts/prepare_libero10_dense_windows.py`](scripts/prepare_libero10_dense_windows.py) | 轨迹隔离的 stride 1/4/8/32 窗口索引 |
| [`scripts/cache_libero10_idm_tokens.py`](scripts/cache_libero10_idm_tokens.py) | 冻结 WAM / JEPA 原始输出与完整 token 缓存 |
| [`scripts/run_libero10_idm_cache_shard.sh`](scripts/run_libero10_idm_cache_shard.sh) | 双 shard、可断点缓存准备 |
| [`scripts/train_libero10_idm_two_stage450.py`](scripts/train_libero10_idm_two_stage450.py) | IDM 结构、真实帧预训练、冻结 IDM 后的 MLP pilot |
| [`scripts/train_libero10_idm_dense30.py`](scripts/train_libero10_idm_dense30.py) | 全量真实帧 IDM 继续训练，30 epoch、无早停 |
| [`scripts/train_libero10_transformer_la450.py`](scripts/train_libero10_transformer_la450.py) | 4 层 residual Transformer；pilot / 全量窗口 / 新 IDM 对照 |
| [`scripts/train_libero10_idm_oracle_comparison450.py`](scripts/train_libero10_idm_oracle_comparison450.py) | 真实 future oracle、WAM future、无 LA 匹配对照 |
| [`scripts/audit_idm_real_wam_gap450.py`](scripts/audit_idm_real_wam_gap450.py) | 真实/生成 LA 的维度、时间段及任务差异诊断 |
| [`experiments/libero/eval_libero_single.py`](experiments/libero/eval_libero_single.py) | FastWAM / JEPA 闭环评测；尚未接入最新 Transformer |
| [`experiments/libero/vjepa2_ac_ranker.py`](experiments/libero/vjepa2_ac_ranker.py) | 历史 ranking、guidance 与图像 / AC 适配 |

历史 predictor finetune、反传、ranking、harm/rescue、trigger 和 latency 脚本保留用于追溯。完整的原 README 已移到 [`docs/HISTORICAL_GUIDANCE_README.md`](docs/HISTORICAL_GUIDANCE_README.md)，其中配置和“当前”结论均属于当时实验，不是新 corrector 的默认设置。

## 环境与复现

本次提交是研究快照，**不是一键安装的可移植训练包**。训练脚本目前使用服务器目录 `/data/pxchen/PACT-WAM`、外部 LIBERO / V-JEPA2 checkout 和本地 checkpoint；在其他机器运行前必须修改脚本的 `ROOT`、数据与 checkpoint 路径，安装仓库依赖并准备匹配的外部环境。权重、演示 HDF5 和约 47GB token 缓存不会随 GitHub 下载。

在已配置的服务器环境中，先按代码导航准备窗口与缓存。下述命令均在仓库根目录执行；`PYTHONPATH` 必须包含仓库根目录。`best.pt` 和 `last.pt` 包含模型、优化器、epoch/step 和 RNG 状态，但目前脚本**没有自动加载 last.pt 续训的 CLI**，不要直接重跑覆盖已存在的实验目录。

```bash
# 全量 IDM：固定 30 epoch，加载旧 IDM 权重；先检查 source 和 OUT 路径。
CUDA_VISIBLE_DEVICES=4 PYTHONPATH="$PWD" OMP_NUM_THREADS=4 \
  nohup /data/pxchen/envs/jepa-wam/bin/python -u \
  scripts/train_libero10_idm_dense30.py > idm_dense30.log 2>&1 < /dev/null &

# 新 IDM 的 LA + Transformer：独立输出目录，确保 checkpoint 为第 29 epoch。
CUDA_VISIBLE_DEVICES=4 PYTHONPATH="$PWD" OMP_NUM_THREADS=4 \
  nohup /data/pxchen/envs/jepa-wam/bin/python -u \
  scripts/train_libero10_transformer_la450.py --dense \
  --idm-checkpoint "$PWD/artifacts/libero10_idm_real_stride32_30epoch_20261004/best.pt" \
  --expected-idm-epoch 29 \
  --out "$PWD/artifacts/libero10_transformer_la_stride32_idm29_20261004" \
  > transformer_idm29.log 2>&1 < /dev/null &
```

GPU 编号只是本次运行示例，启动前需确认空闲。Transformer 两组在同一张卡依次运行：`wam_la` 与 `zero_la`，相同初始化 / batch 顺序；batch 8、AdamW LR 3e-4、weight decay 0.01、dropout 0.1、clip 1、最多 50 epoch、patience 10。每 5 epoch 保存，并保留 best / last。IDM 的无早停设置**不等于** Transformer 也禁用早停。

## 历史 guidance / trigger 与 latency

- FastWAM 严格级联 `infer_video() → infer_action_from_video()`；评测 video/action shift 固定 1/1。
- 历史 raw-gradient guidance 在 denoise steps 7/8/9 修正动作，倍率 4×、2 个 AC steps、单次修正 RMS cap 0.02。该设置不用于新 residual Transformer。
- 历史 trigger 使用 ActionExpert future response 的 denoise 相对变化，校准预算统一 30%；固定阈值在线调用率不保证恰好 30%。前 0–3 replan 的漏触发促使采用强制前四次触发策略，但会增加总体调用预算。
- response 是分组 attention 输出贡献的范数信号，不是严格因果归因；JEPA loss 下降也不保证动作更接近真实执行目标。
- 历史 FP32 稳态 latency 示例：no guidance 576ms，完整 decode 串行 1,193ms，前缀异步 + compiled energy 优化 837ms。no guidance 不统计不参与控制输出的 decode。不同硬件 / 编译 / 测量口径不可混用；这些数值不是新 Transformer 的实测 latency。

## 仓库边界

版本管理包含源码、文档以及 [`reports/research_20261004/`](reports/research_20261004/) 下精选的轻量配置、summary 和 epoch 指标。原始 `artifacts/` 中既有历史受控元数据保留；新增实验缓存默认不提交。

不提交 checkpoint、HDF5 / NPZ、raw / token / LA 缓存、rollout 视频、下载包、运行日志、PID / lock、编辑备份和凭据。第三方模型和代码请遵循各自许可证；本仓库的许可证见 [`LICENSE`](LICENSE)。
