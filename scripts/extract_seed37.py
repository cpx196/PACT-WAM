from pathlib import Path
import json,numpy as np,imageio.v2 as imageio,ast,collections
ROOT=Path('/data/pxchen/PACT-WAM/evaluate_results/weighted_six_four_seeds_sameT5_20261022')
OUT=Path('/data/pxchen/PACT-WAM/artifacts/seed37_audit_20261022');OUT.mkdir(exist_ok=True)
steps=[0,16,32,48,64,80,96,112,128,144,160,184,208,232,256,280,304,328,352,400,480,560,640,699]
meta={};arrays={}
for arm in ['control','weighted']:
 for tid in [151,559,590]:
  p=next((ROOT/arm/'seed_37/libero_10/videos').glob(f'*task{tid}_trial0*'))
  reader=imageio.get_reader(p);n=reader.count_frames()
  ids=[s for s in steps if s<n]
  frames=np.stack([reader.get_data(i) for i in ids]);reader.close()
  key=f'{arm}_{tid}';arrays[key]=frames;arrays[key+'_steps']=np.array(ids)
  meta[key]=dict(video=str(p),frames=n,shape=list(frames.shape),selected_controls=ids)
 np.savez_compressed(OUT/f'{arm}_frames.npz',**{k:v for k,v in arrays.items() if k.startswith(arm)})
(OUT/'video_metadata.json').write_text(json.dumps(meta,indent=2))
allplans={}
for arm in ['control','weighted']:
 allplans[arm]=[json.loads(l) for l in (ROOT/arm/'plans.jsonl').read_text().splitlines() if json.loads(l)['seed']==37]
 (OUT/f'{arm}_plans.json').write_text(json.dumps(allplans[arm]))
# Per-node diagnostics are ordered in the shared worker stream; associate using actual forward audit losses, not text timestamps.
checks=[json.loads(l) for l in (ROOT/'weighted/gripper_forward_checks.jsonl').read_text().splitlines()]
grads=[json.loads(l) for l in (ROOT/'weighted/gripper_gradient_checks.jsonl').read_text().splitlines()]
checks=[r for r in checks if r.get('seed')==37 and r.get('stage')=='FORMAL'];grads=[r for r in grads if r.get('seed')==37 and r.get('stage')=='FORMAL']
(OUT/'forward_checks_seed37.json').write_text(json.dumps(checks));(OUT/'gradient_checks_seed37.json').write_text(json.dumps(grads))
# First commands compare same-state direct intervention; later deltas also include state divergence.
rows=[]
for tid in [151,559,590]:
 a=next(r for r in allplans['control'] if r['task_id']==tid and r['replan_index']==0)
 b=next(r for r in allplans['weighted'] if r['task_id']==tid and r['replan_index']==0)
 x=np.array(a['first8_commands']);y=np.array(b['first8_commands'])
 gs=[r for r in grads if r['task_id']==tid]
 rows.append(dict(task_id=tid,first_difference_control=int(np.flatnonzero(np.any(x!=y,axis=1))[0]),first8_pose_delta_rms=float(np.sqrt(np.mean((x[:,:6]-y[:,:6])**2))),first8_pose_max_abs=float(np.max(np.abs(x[:,:6]-y[:,:6]))),first8_gripper_control=x[:,6].tolist(),first8_gripper_weighted=y[:,6].tolist(),gripper_gradient_max=max(r['gripper_gradient_max'] for r in gs),guidance_backward_calls=len(gs)))
(OUT/'initial_action_audit.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows),flush=True)
