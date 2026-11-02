"""Upload the exact evaluated AR2 checkpoint and its input contract to a private HF model repo."""
import argparse,hashlib,json,tempfile
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--repo-id',required=True,help='Your Hugging Face namespace/repository')
 p.add_argument('--root',type=Path,default=Path('/data/pxchen/PACT-WAM'))
 p.add_argument('--include-weights',action='store_true',help='Also upload the ten fixed task calibration tables')
 p.add_argument('--public',action='store_true',help='Explicitly request a public model repository')
 p.add_argument('--dry-run',action='store_true',help='Validate local payload and print its manifest without network calls')
 args=p.parse_args();root=args.root
 pred=root/'artifacts/fastwam_predictor21_ar2_full10epoch_bs16_20261014/step19690.pt'
 expected='dec6eef937335a04c9fd8e23d7065163d2348919041e548749d8fe132313289e'
 assert pred.is_file(),pred
 assert sha(pred)==expected,'Checkpoint differs from the evaluated predictor'
 files=[(pred,'step19690.pt')]
 required={
  'normalization/train_only_stats.json':root/'artifacts/fastwam_predictor21_ar2_stride20_20261014/train_only_stats.json',
  'input_contract.json':root/'artifacts/predictor21_joint20_stride20_20261011/contract.json',
  'code/fastwam_ar2_eval.py':root/'scripts/fastwam_ar2_eval.py',
  'code/smoke_joint_predictor.py':root/'artifacts/predictor2.1ac1009/smoke_joint_predictor.py',
  'code/vectorized_ac_attention.py':root/'artifacts/predictor2.1ac1009/vectorized_ac_attention.py',
 }
 for target,source in required.items():
  assert source.is_file(),source;files.append((source,target))
 if args.include_weights:
  registry=root/'evaluate_results/feedback_weights_libero10_remaining7_20261031/weights_registry_all10.json'
  d=json.loads(registry.read_text());assert d['complete'] and len(d['tasks'])==10 and d['predictor_sha256']==expected
  for row in d['tasks']:
   source=Path(row['path']);assert sha(source)==row['sha256'];files.append((source,'offline_weights/'+row['task']+'/weights.pt'))
 manifest=dict(checkpoint_sha256=expected,checkpoint_step=19690,precision='original checkpoint, unchanged',files=[dict(path=target,bytes=source.stat().st_size,sha256=sha(source)) for source,target in files])
 if args.dry_run:
  print(json.dumps(manifest,indent=2));return
 from huggingface_hub import HfApi
 api=HfApi();identity=api.whoami();print('Authenticated HF account:',identity['name'])
 api.create_repo(repo_id=args.repo_id,repo_type='model',private=not args.public,exist_ok=True)
 info=api.repo_info(args.repo_id,repo_type='model')
 assert info.private==(not args.public),'Existing repo visibility differs from requested visibility'
 with tempfile.TemporaryDirectory(prefix='pact-hf-metadata-') as td:
  td=Path(td);(td/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
  (td/'README.md').write_text('''---
tags:
- robotics
- world-model
- jepa
- libero
---
# PACT-WAM AR2 action-conditioned predictor

Exact step19690 checkpoint used by the PACT-WAM guidance experiments. Original precision and serialization are retained. WAM and JEPA encoder weights are not included. This is a custom predictor, not a standalone Transformers AutoModel.

Each predictor call consumes a current latent [B,576,1024], four low-level actions [B,4,7], robot state [B,8], and a phase index. Two autoregressive calls predict the two future horizons. Apply the supplied training-only action/state normalization and the camera/action contracts in the PACT-WAM source.

Implementation and evaluation: https://github.com/cpx196/PACT-WAM . Read the experiment reports for limitations: initial positive results did not extend to a broad success-rate improvement in the subsequent seven-task seed7 study. Optional task weights use base-task training demonstrations, not perturbed evaluation trajectories.

Files are indexed with SHA256 in artifact_manifest.json. Code entry point: code/fastwam_ar2_eval.py. Preserve the predictor checkpoint hash when using existing task calibration tables.
''')
  api.upload_folder(repo_id=args.repo_id,repo_type='model',folder_path=td,commit_message='Document evaluated AR2 predictor and reproducibility manifest')
 for source,target in files:
  api.upload_file(repo_id=args.repo_id,repo_type='model',path_or_fileobj=str(source),path_in_repo=target,commit_message='Upload '+target)
 # Verify the committed file set and byte lengths without downloading weights again.
 expected_sizes={x['path']:x['bytes'] for x in manifest['files']}
 actual={x.path:x.size for x in api.list_repo_tree(args.repo_id,repo_type='model',recursive=True) if hasattr(x,'size')}
 assert all(actual.get(name)==size for name,size in expected_sizes.items()),'Remote file size verification failed'
 final=api.repo_info(args.repo_id,repo_type='model');print(json.dumps(dict(repo_id=args.repo_id,revision=final.sha,files_verified=len(expected_sizes))))

if __name__=='__main__':main()
