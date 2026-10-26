# Current AR2 compiled latency / bottleneck audit

No guidance median: 571.69 ms
Prefix serial median: 1084.18 ms
Prefix async median: 1027.03 ms
Async encoder wait median: 0.04 ms
Guidance energy forward: 210.46 ms; backward: 129.62 ms

Stage CUDA durations overlap; do not sum them as end-to-end latency. The action span includes guidance and dependency wait. Compare full/prefix serial for decode saving, prefix serial/async for overlap, verify true/false for verification overhead. GPU preprocessing is exploratory unless validation reports equivalence. Cold compile and profiler overhead are excluded from steady samples. This 4090D result does not establish Thor latency or task success.

A low encoder wait does not prove no contention: compare component durations between serial and async. For kernel-level compute-vs-bandwidth attribution use hardware counters; torch.profiler alone cannot establish bandwidth saturation.

## 自动对照结果
Measured dominant span: action_including_guidance_and_wait
Full-to-prefix serial wall reduction: 68.38 ms
Serial-to-async wall reduction: 57.15 ms
Verification wall overhead: 92.32 ms
Encoder CUDA-span slowdown during overlap: 1.107x
Negative reductions indicate this optimization lost time for this fixed input. GPU-spans include queued execution and preemption; do not interpret them as isolated kernel service time.