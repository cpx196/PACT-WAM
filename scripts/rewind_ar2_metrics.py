"""Archive unsaved metric tail before resuming our own run at a checkpoint."""
import json,time
from pathlib import Path
p=Path('/data/pxchen/PACT-WAM/artifacts/fastwam_predictor21_ar2_full10epoch_bs16_20261014')
assert (p/'step00250.pt').exists()
active=p/'metrics.jsonl';text=active.read_text();rows=[json.loads(line) for line in text.splitlines() if line.strip()]
archive=p/f'metrics_before_prefetch_{int(time.time())}.jsonl';active.rename(archive)
retained=[r for r in rows if r['step']<=250]
active.write_text(''.join(json.dumps(r)+'\n' for r in retained))
print(json.dumps(dict(archive=str(archive),retained_records=len(retained),archived_tail_records=len(rows)-len(retained))))
