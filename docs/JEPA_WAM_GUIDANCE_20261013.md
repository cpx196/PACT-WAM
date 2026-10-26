# JEPA-WAM gradient guidance (2026-10-13)

Runtime-only adapter: native policy files and checkpoint weights are unchanged.
No ARB/MLP/IDM is loaded and no optimizer is used.

## Contract

- Frozen released JEPA-WAM LIBERO policy, step040000/epoch37.
- Frozen joint20 Predictor: `predictor21_joint20_train10epoch_bs16_gpu3_20261011/step19690.pt`.
- Current main-view image uses the existing LIBERO upright preprocessing exactly once; wrist remains a native policy input only.
- Predictor input: current `[1,576,1024]`, executable actions `[1,20,7]`, state `[1,8]`.
- Fixed W target: Qwen main-view visual hidden positions 1:577, through the released trained alignment head, `[1,576,1024]`.
- P and W represent the jointly encoded current/t+31 target;20 actions are retained, matching the approved training contract.
- Loss: mean tokenwise `1-cosine(P,W)`. The released alignment head was trained with cosine, so raw W magnitudes are not supervised; raw L1 is not a drop-in objective here.
- Native four flow steps unchanged. Guidance at **between steps3/4, and after step4** (zero-based2,3).
- At flow time t, clean endpoint estimate `x+(1-t)*v` is detached and differentiated through only the Predictor. Correction is applied to the next flow state, not backpropagated through Qwen or Action Expert.
- Raw gradient scale4; per-node first-six normalized-action RMS cap0.02; all20 commands can be corrected. No gradient normalization.
- Gripper gradient is zero. Later flow predictions may still alter gripper because they see a corrected action state; the adapter does not claim final guided gripper equals an independently sampled native trajectory.
- Predictor uses FP32 forward/backward. Policy keeps native BF16; flow state switches to FP32 at first correction to preserve small updates. No torch.compile.
- No trigger: both guidance nodes execute every replan.
- `verify=True` measures post-correction loss but does not reject/backtrack a loss-increasing update, matching historical diagnostic behavior. This can be switched off after correctness validation to reduce latency.

## Smoke and launch

Server scripts:

```text
/data/pxchen/PACT-WAM/scripts/jepawam_gradient_guidance.py
/data/pxchen/PACT-WAM/scripts/smoke_online_jepawam_guidance.py
/data/pxchen/PACT-WAM/scripts/launch_jepawam_guidance_smoke.sh
```

The smoke checks zero-scale exact native output, nonzero finite action gradients,
central finite-difference agreement, cap, zero direct gripper correction and frozen
Predictor weights; then runs task365/seed7 native and guided, saving both videos.
It is locked and refuses duplicate output. It does **not** launch144 tasks.

Final smoke directory:
`/data/pxchen/PACT-WAM/artifacts/jepawam_gradient_guidance2_cosine_smoke_20261013`.

Earlier migration probes using raw L1 or three guidance nodes are retained in
separate directories and must not be reported as the final configuration.

## Final smoke result

- Completed on GPU2, about87 seconds including55 seconds loading; GPU released.
- Zero-scale maximum difference:0. Frozen Predictor parameter gradients remain absent.
- First numerical gradient check: analytic directional derivative0.42738, central difference0.38977 (8.8% discrepancy at epsilon0.002; same sign).
- Native task365/seed7: success,236 controls,12 replans,mean policy192.6ms.
- Two-node cosine guidance: success,230 controls,12 replans,mean policy767.5ms including both post-update verification forwards.
- All24 guidance nodes reduced their checked cosine loss;2 reached the per-node0.02 cap.
- Peak PyTorch allocated memory5451MiB. This is allocated memory, not total nvidia-smi process usage.
- This is one-task integration smoke, not a comparative success-rate evaluation.
