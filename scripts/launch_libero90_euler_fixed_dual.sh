#!/usr/bin/env bash
set -euo pipefail
cd /data/pxchen/PACT-WAM
source scripts/local_libero_env.sh
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
root=/data/pxchen/PACT-WAM/artifacts/jepa_libero90_euler_fixed_rot180_20260928
mkdir -p "$root"
exec 9> "$root/launcher.lock"
flock -n 9 || exit 3
echo $$ > "$root/launcher.pid"
if [[ -e "$root/train_2000/step_000500.pt" ]]; then
    echo "Refuse to overwrite an existing training run; resume must be explicit" >&2
    exit 4
fi
trap 'code=$?; if (( code != 0 && code != 75 )); then echo "FAILED exit=$code" > "$root/status.txt"; fi' EXIT
CUDA_VISIBLE_DEVICES="" "$FASTWAM_PYTHON" scripts/preflight_libero90_euler.py > "$root/preflight.json"
# Never share a GPU with an existing compute job. Graphics-only residual memory
# up to 1GiB is allowed (the existing LIBERO renderer uses GPU2).
deadline=$((SECONDS + 7200))
echo "WAITING_FOR_TWO_IDLE_GPUS" > "$root/status.txt"
while true; do
if gpus=$("$FASTWAM_PYTHON" -c '
import subprocess
busy={s.strip() for s in subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader"],text=True).splitlines()}
rows=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True).splitlines()
free=[]
for row in rows:
    index,uuid,memory,util=[s.strip() for s in row.split(",")]
    if uuid not in busy and int(memory)<1024 and int(util)<10:
        free.append(index)
if len(free)<2:
    raise SystemExit("NEED_TWO_IDLE_GPUS; available="+",".join(free))
print(",".join(free[:2]))
'); then
    break
fi
if (( SECONDS >= deadline )); then
    echo "WAIT_TIMEOUT_NO_GPUS_ALLOCATED" > "$root/status.txt"
    exit 75
fi
sleep 30
done
export CUDA_VISIBLE_DEVICES="$gpus"
echo "$(date '+%F %T') GPU=$gpus" > "$root/gpu_assignment.txt"
common=(--data /data/pxchen/libero-suite/LIBERO/datasets/libero_90
        --repo /data/pxchen/vjepa2
        --checkpoint /data/pxchen/JEPA_WAM/checkpoints/vjepa2/vjepa2-ac-vitg.pt
        --batch-size 8 --accum 1 --lr 1e-5 --warmup 100 --workers 2
        --seed 239 --image-rotation-degrees 180)
echo "DUAL_GPU_SMOKE_RUNNING" > "$root/status.txt"
"$FASTWAM_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
    scripts/train_vjepa2_ac_libero90_euler_fixed.py "${common[@]}" \
    --output "$root/smoke_20" --steps 20 --save-every 20 > "$root/smoke_20.log" 2>&1
# A successful checkpoint save exercises the optimizer and scheduler too.
"$FASTWAM_PYTHON" -c 'import sys,torch; c=torch.load(sys.argv[1],map_location="cpu",weights_only=False,mmap=True); assert c["step"]==20; assert c["config"]["data_format_version"]=="libero_ac_euler_v1"; assert c["optimizer"]["state"]; assert "scheduler" in c; print("SMOKE_CHECKPOINT_PASSED")' \
    "$root/smoke_20/step_000020.pt" > "$root/smoke_checkpoint_check.log"
# Start from the original pretrained model; do not continue the 20-step smoke.
echo "TRAINING_2000_STEPS" > "$root/status.txt"
"$FASTWAM_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
    scripts/train_vjepa2_ac_libero90_euler_fixed.py "${common[@]}" \
    --output "$root/train_2000" --steps 2000 --save-every 500 > "$root/train_2000.log" 2>&1
echo "$(date '+%F %T') COMPLETED" > "$root/completion.txt"
echo "COMPLETED" > "$root/status.txt"
