# 2026-10-26 research snapshot

最新主结果：白杯任务全部42个Robot Initial States扰动、seed7，control30/42、unweighted30/42、weighted34/42。参见根README和docs/LATEST_ROBOT42_20261026.md。

- `robot42/`：最新三组的逐episode结果、配置、预检、结构化loss与梯度日志、运行源码。
- `latency/`：独立固定输入性能测试；未完整复现旧837ms异步图，不能与成功率实验混同。
- `archives/`：239个历史文本快照压缩包，每包有`.index.json`逐文件索引。保留原路径，可以`tar -tzf`查看或解压到独立目录。
- `archive_index.json`：每包SHA256、大小和成员数量。
- `publication_manifest.json`：上传范围与排除项。

归档是追溯材料，不是239组有效对照；其中包含smoke、失败/中断实验和旧口径。只有根README明确指定的126个episode用于30/30/34结论。压缩包不包含大模型权重、原始视频或token缓存。原始trace包含在ar2_compiled_latency对应包中，编译二进制缓存仍保留服务器。
