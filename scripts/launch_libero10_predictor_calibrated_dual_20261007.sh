#!/usr/bin/env bash
set -euo pipefail
cd /data/pxchen/PACT-WAM
PY=/data/pxchen/envs/jepa-wam/bin/python
root=/data/pxchen/PACT-WAM/artifacts/jepa_libero10_response_calibrated_rot180_20261007
mkdir -p "$root"
exec 9> "$root/launcher.lock"
flock -n 9 || exit 3
echo $$ > "$root/launcher.pid"
trap 'code=$?; if (( code != 0 )); then echo "FAILED exit=$code" > "$root/status.txt"; fi' EXIT
if [[ -e "$root/train_2000" || -e "$root/smoke_5" ]]; then
    echo 'Refuse to overwrite an existing run; explicit resume required' >&2
    exit 4
fi
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/data/pxchen/PACT-WAM/scripts:/data/pxchen/PACT-WAM${PYTHONPATH:+:$PYTHONPATH}
common=(--data /data/pxchen/PACT-WAM/artifacts/libero10_guidance_backward_smoke_20260929/data/libero_10
        --repo /data/pxchen/vjepa2
        --checkpoint /data/pxchen/JEPA_WAM/checkpoints/vjepa2/vjepa2-ac-vitg.pt
        --manifest /data/pxchen/PACT-WAM/artifacts/libero10_idm_dense_windows_20261004/windows_stride32.json
        --calibration /data/pxchen/PACT-WAM/artifacts/libero_ac_command_response_calibration_20261007/calibration.json
        --batch-size 8 --accum 1 --lr 1e-5 --warmup 100 --workers 2
        --seed 239 --image-rotation-degrees 180)
echo PREFLIGHT > "$root/status.txt"
CUDA_VISIBLE_DEVICES='' "$PY" - <<'PY' > "$root/preflight.json"
import json, numpy as np, torch
from pathlib import Path
from train_vjepa2_ac_libero10_calibrated import LiberoWindows
from experiments.libero.vjepa2_ac_ranker import LiberoACAdapter
root=Path('/data/pxchen/PACT-WAM')
manifest=root/'artifacts/libero10_idm_dense_windows_20261004/windows_stride32.json'
cal=json.loads((root/'artifacts/libero_ac_command_response_calibration_20261007/calibration.json').read_text())
data=root/'artifacts/libero10_guidance_backward_smoke_20260929/data/libero_10'
datasets={s:LiberoWindows(data,split=s,repeats=1,seed=239,image_rotation_degrees=180,calibration=cal,manifest=manifest) for s in ('train','val')}
assert set((p,d) for p,d,n in datasets['train'].rows).isdisjoint((p,d) for p,d,n in datasets['val'].rows)
for ds in datasets.values():
    frames, actions, states=ds[0]
    assert frames.shape[0]==6 and actions.shape==states.shape==(2,7)
    assert torch.isfinite(actions).all() and torch.isfinite(states).all()
v1=datasets['val'][0];v2=datasets['val'][0]
assert all(torch.equal(a,b) for a,b in zip(v1,v2))
adapter=LiberoACAdapter(osc_position_scale=.05*cal['translation_response_gain'],osc_rotation_scale=.5*cal['rotation_response_gain'])
commands=np.random.default_rng(239).uniform(-.8,.8,(4,8,7)).astype(np.float32)
commands[0]=0
state=np.array([.3,0,.4,.1,-.1,.2,.3],dtype=np.float32)
a=adapter.convert(commands,state)
t=torch.tensor(commands,requires_grad=True)
b=adapter.convert_torch(t,torch.tensor(state))
for key in ('actions','states','terminal_states'):
    np.testing.assert_allclose(getattr(a,key),getattr(b,key).detach().numpy(),rtol=1e-4,atol=2e-6)
sum(getattr(b,k).square().sum() for k in ('actions','states','terminal_states')).backward()
assert torch.isfinite(t.grad).all()
np.testing.assert_allclose(a.actions[:,:,:3],commands[:,:,:3].reshape(4,2,4,3).sum(2)*.05*cal['translation_response_gain'],atol=1e-7)
print(json.dumps(dict(passed=True,train_demos=len(datasets['train'].rows),val_demos=len(datasets['val'].rows),calibration=cal,checks=['split_disjoint','fixed_validation','numpy_torch_match','finite_autograd','calibrated_xyz']),indent=2))
PY
pick_idle() {
"$PY" - <<'PY'
import subprocess
busy=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).split())
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
free=[]
for r in rows:
    i,u,m,v=[x.strip() for x in r.split(',')]
    if u not in busy and int(m)<128 and int(v)<5:free.append(i)
if len(free)<2:raise SystemExit(1)
print(','.join(free[:2]))
PY
}
echo WAITING_FOR_TWO_IDLE_GPUS > "$root/status.txt"
deadline=$((SECONDS+7200))
while true; do
    if first=$(pick_idle); then
        sleep 10
        if second=$(pick_idle) && [[ "$first" == "$second" ]]; then break; fi
    fi
    if (( SECONDS >= deadline )); then echo WAIT_TIMEOUT >&2; exit 75; fi
    sleep 20
done
export CUDA_VISIBLE_DEVICES="$second"
echo "$(date '+%F %T %Z') GPU=$second" > "$root/gpu_assignment.txt"
echo SMOKE_RUNNING > "$root/status.txt"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
    scripts/train_vjepa2_ac_libero10_calibrated.py "${common[@]}" \
    --output "$root/smoke_5" --steps 5 --save-every 5 > "$root/smoke_5.log" 2>&1
"$PY" - "$root/smoke_5/step_000005.pt" <<'PY' > "$root/smoke_checkpoint_check.log"
import sys, torch
c=torch.load(sys.argv[1],map_location='cpu',weights_only=False,mmap=True)
assert c['step']==5 and c['config']['data_format_version']=='libero_ac_euler_response_calibrated_v1'
assert c['optimizer']['state'] and 'scheduler' in c
assert c['config']['calibration_config']['fitted_split']=='train_only'
assert all(torch.isfinite(p).all() for p in c['predictor'].values() if p.is_floating_point())
print('SMOKE_CHECKPOINT_PASSED')
PY
# Formal training starts from the official model, not the smoke checkpoint.
echo TRAINING_2000_STEPS > "$root/status.txt"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
    scripts/train_vjepa2_ac_libero10_calibrated.py "${common[@]}" \
    --output "$root/train_2000" --steps 2000 --save-every 100 > "$root/train_2000.log" 2>&1
date '+%F %T %Z' > "$root/completion.txt"
echo COMPLETED > "$root/status.txt"
