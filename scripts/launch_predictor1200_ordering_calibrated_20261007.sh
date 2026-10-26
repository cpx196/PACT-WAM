#!/usr/bin/env bash
set -euo pipefail
cd /data/pxchen/PACT-WAM
PY=/data/pxchen/envs/jepa-wam/bin/python
root=/data/pxchen/PACT-WAM/artifacts/libero10_predictor1200_calibrated_ordering_20261007
mkdir -p "$root"
exec 9> "$root/launcher.lock"
flock -n 9 || exit 3
echo $$ > "$root/launcher.pid"
trap 'code=$?; if (( code != 0 )); then echo "FAILED exit=$code" > "$root/status.txt"; fi' EXIT
if [[ -e "$root/new1200_calibrated" || -e "$root/old_legacy" || -e "$root/old_calibrated" ]]; then
    echo 'Refuse duplicate diagnostic outputs' >&2; exit 4
fi
export OMP_NUM_THREADS=4
export PYTHONPATH=/data/pxchen/PACT-WAM
gpu=$($PY - <<'PY'
import subprocess
busy=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).split())
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
free=[]
for r in rows:
    i,u,m,v=[x.strip() for x in r.split(',')]
    if u not in busy and int(m)<128 and int(v)<5:free.append(i)
if not free:raise SystemExit('No completely idle GPU; do not share or preempt')
print(free[0])
PY
)
export CUDA_VISIBLE_DEVICES="$gpu"
echo "$(date '+%F %T %Z') GPU=$gpu" > "$root/gpu_assignment.txt"
old=/data/pxchen/PACT-WAM/checkpoints/vjepa2-ac-libero90-euler-fixed-rot180-step2000-inference.pt
new=/data/pxchen/PACT-WAM/artifacts/jepa_libero10_response_calibrated_rot180_20261007/train_2000/step_001200.pt
cal=/data/pxchen/PACT-WAM/artifacts/libero_ac_command_response_calibration_20261007/calibration.json
# Sequential single-GPU runs; no training or simulator evaluation.
for label in new1200_calibrated old_legacy old_calibrated; do
    checkpoint="$old"
    extra=()
    if [[ "$label" == new1200_calibrated ]]; then checkpoint="$new"; fi
    if [[ "$label" != old_legacy ]]; then extra=(--calibration "$cal"); fi
    echo "RUNNING $label" > "$root/status.txt"
    "$PY" -u scripts/diagnose_libero10_predictor_calibrated_ordering.py \
        --label "$label" --checkpoint "$checkpoint" --output "$root/$label" \
        "${extra[@]}" > "$root/$label.log" 2>&1
done
"$PY" - "$root" <<'PY' > "$root/comparison.log"
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);arms={k:json.loads((r/k/'summary.json').read_text()) for k in ('old_legacy','old_calibrated','new1200_calibrated')}
assert len({tuple(s['config']['ids']) for s in arms.values()})==1
comparison={k:dict(overall=s['overall'],seconds=s['seconds'],checkpoint=s['config']['checkpoint'],calibration=s['config']['calibration']) for k,s in arms.items()}
(r/'comparison.json').write_text(json.dumps(comparison,indent=2))
print(json.dumps(comparison,indent=2))
PY
date '+%F %T %Z' > "$root/completion.txt"
echo COMPLETED > "$root/status.txt"
