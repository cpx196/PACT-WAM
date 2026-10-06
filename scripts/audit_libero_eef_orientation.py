"""Read-only audit of rotvec-as-Euler interpretation on real LIBERO90 data."""
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

data = Path('/data/pxchen/libero-suite/LIBERO/datasets/libero_90')
samples = []
for path in sorted(data.glob('*.hdf5')):
    with h5py.File(path) as f:
        ee = f['data/demo_0/obs/ee_states']
        for index in np.unique(np.linspace(8, len(ee)-9, 8).astype(int)):
            rotvec = np.asarray(ee[int(index), 3:6])
            actual = Rotation.from_rotvec(rotvec)
            misread = Rotation.from_euler('xyz', rotvec)
            angle = float(np.rad2deg((misread.inv()*actual).magnitude()))
            samples.append({'file': path.name, 'frame': int(index),
                            'rotvec': rotvec.tolist(), 'equivalent_euler': actual.as_euler('xyz').tolist(),
                            'misinterpreted_orientation_error_degrees': angle})
angles = np.array([r['misinterpreted_orientation_error_degrees'] for r in samples])
report = {'files': len(list(data.glob('*.hdf5'))), 'sampled_states': len(samples),
          'sampling': 'demo_0, 8 evenly spaced states per file; descriptive audit, not rollout causality',
          'angular_error_degrees_percentiles': dict(zip(('min','p25','median','p75','p95','max'),
                                                        np.percentile(angles,[0,25,50,75,95,100]).tolist())),
          'samples': samples}
out = Path('/data/pxchen/PACT-WAM/evaluate_results/harm_diagnostic_20260928/eef_orientation_audit.json')
out.write_text(json.dumps(report, indent=2))
print(json.dumps({k:v for k,v in report.items() if k != 'samples'}, indent=2))
