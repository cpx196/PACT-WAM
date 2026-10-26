#!/usr/bin/env bash
set -euo pipefail
cd /data/pxchen/PACT-WAM
PY=/data/pxchen/envs/jepa-wam/bin/python
root=/data/pxchen/PACT-WAM/artifacts/libero10_calibrated_idm_arb_joint10_20261007
mkdir -p "$root"
exec 9> "$root/launcher.lock"
flock -n 9 || exit 3
echo $$ > "$root/launcher.pid"
trap 'code=$?; if (( code != 0 )); then echo "FAILED exit=$code" > "$root/status.txt"; fi' EXIT
if [[ -e "$root/cache" || -e "$root/joint" || -e "$root/zero_la" ]]; then
    echo 'Refuse to overwrite/duplicate existing run' >&2; exit 4
fi
export OMP_NUM_THREADS=4
export PYTHONPATH=/data/pxchen/PACT-WAM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
idle() {
"$PY" - "$1" <<'PY'
import sys,subprocess
count=int(sys.argv[1]);busy=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).split())
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines();free=[]
for r in rows:
    i,u,m,v=[x.strip() for x in r.split(',')]
    if u not in busy and int(m)<128 and int(v)<5:free.append(i)
if len(free)<count:raise SystemExit(1)
print(','.join(free[:count]))
PY
}
echo WAITING_IDLE_GPU_FOR_CACHE > "$root/status.txt"
deadline=$((SECONDS+7200))
until cache_gpu=$(idle 1); do
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 20
done
echo "$(date '+%F %T %Z') CACHE_GPU=$cache_gpu" > "$root/gpu_assignment.txt"
echo UPDATING_STEP2000_PREDICTOR_CACHE > "$root/status.txt"
CUDA_VISIBLE_DEVICES="$cache_gpu" "$PY" -u scripts/cache_libero10_predictor2000_calibrated.py \
    --out "$root/cache" --batch 8 > "$root/cache.log" 2>&1
echo WAITING_IDLE_GPUS_FOR_TRAIN > "$root/status.txt"
deadline=$((SECONDS+7200));second_deadline=$((SECONDS+300))
parallel=false
while true; do
    if pair=$(idle 2); then
        first=${pair%,*};second=${pair#*,};parallel=true;break
    fi
    if (( SECONDS >= second_deadline )) && first=$(idle 1); then break; fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 20
done
common=(--predictor-cache "$root/cache" --epochs 10)
echo TRAINING_JOINT_AND_ZERO_LA > "$root/status.txt"
echo "$(date '+%F %T %Z') JOINT_GPU=$first PARALLEL=$parallel" >> "$root/gpu_assignment.txt"
CUDA_VISIBLE_DEVICES="$first" "$PY" -u scripts/train_libero10_idm_arb_calibrated_joint10.py \
    --out "$root/joint" --variant joint "${common[@]}" > "$root/joint.log" 2>&1 &
joint_pid=$!;echo "$joint_pid" > "$root/joint_worker.pid"
if [[ "$parallel" == true ]]; then
    echo "$(date '+%F %T %Z') ZERO_LA_GPU=$second" >> "$root/gpu_assignment.txt"
    CUDA_VISIBLE_DEVICES="$second" "$PY" -u scripts/train_libero10_idm_arb_calibrated_joint10.py \
        --out "$root/zero_la" --variant zero_la "${common[@]}" > "$root/zero_la.log" 2>&1 &
    zero_pid=$!;echo "$zero_pid" > "$root/zero_worker.pid"
    # Wait both workers even if one fails; do not leave an unreported sibling.
    a=0;b=0;wait "$joint_pid" || a=$?;wait "$zero_pid" || b=$?
    if (( a != 0 || b != 0 )); then echo "WORKER_FAILED joint=$a zero=$b" >&2;exit 5;fi
else
    wait "$joint_pid"
    until second=$(idle 1); do sleep 20;done
    CUDA_VISIBLE_DEVICES="$second" "$PY" -u scripts/train_libero10_idm_arb_calibrated_joint10.py \
        --out "$root/zero_la" --variant zero_la "${common[@]}" > "$root/zero_la.log" 2>&1
fi
"$PY" - "$root" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);results={}
for variant in ('joint','zero_la'):
    s=json.loads((r/variant/'summary.json').read_text());results[variant]=s['variants'][variant]
    assert s['config']['matched_arb_initialization'] and s['config']['epochs']==10
    assert results[variant]['final']['epoch']==30
report=dict(variants=results,scope='offline normalized action MSE; not robot success',
            matched_arb_initialization=True,trained_full_10_epochs=True)
(r/'summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
PY
date '+%F %T %Z' > "$root/completion.txt"
echo COMPLETED > "$root/status.txt"
