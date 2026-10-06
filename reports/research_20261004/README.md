# 离线动作修正研究记录

这里保存 2026-10-04 训练、2026-10-06 核对的精选轻量证据。数据、checkpoint、token 缓存和视频保留在服务器，不随提交上传。JSON 中的绝对路径是实验 provenance，不是可移植的数据地址。

| 目录 | 实验 | 主要文件 |
|---|---|---|
| `idm_two_stage450_20261004` | 450/50 窗口，真实帧 IDM + 冻结 IDM 后 MLP | summary、temporal_diagnostic |
| `idm_oracle_comparison450_20261004` | 真实 future oracle / WAM future / 无 LA 对照 | summary、gap_audit |
| `transformer_la450_20261004` | 450/50 窗口，4 层 Transformer pilot | config、smoke、summary、两组 epoch 指标 |
| `transformer_la_stride32_20261004` | 全量 stride-32，旧 IDM 的 Transformer | config、summary、两组 epoch 指标 |
| `idm_real_stride32_30epoch_20261004` | 全量真实帧 IDM，固定 30 epoch，无早停 | config、initial_evaluation、smoke、summary、epoch 指标 |
| `transformer_la_stride32_idm29_20261004` | 全量数据，冻结新 IDM 的 epoch 29 权重 | config、smoke、summary；本快照未包含两组逐 epoch 日志 |

说明：

- `best_val_mse` 是验证集选出的最佳 checkpoint，不是最后一轮数值；`final` 是训练结束时的统计。
- Transformer 的 `better` / `worse` 是离线窗口相对 raw action 的误差比较，不是闭环 rescue / harm 数量。
- `zero_la` 独立训练对照，与固定 `wam_la` 模型再置零 LA 的消融，不是同一种实验。
- IDM 的 WAM future 验证指标不参与训练，也不用于选择 best；best 按真实 future 验证误差选择。
- 所有收益都是单 seed 的离线证据，需要独立测试、轨迹层面的统计及闭环评测。
