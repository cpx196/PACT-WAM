"""Audit export/adapter and preserve the exact historical 36x4 task manifest."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.libero.vjepa2_ac_ranker import LiberoACAdapter

ROOT=Path('/data/pxchen/PACT-WAM')
TRAIN=ROOT/'artifacts/jepa_libero10_response_calibrated_rot180_20261007/train_2000/step_002000.pt'
CAL=ROOT/'artifacts/libero_ac_command_response_calibration_20261007/calibration.json'
TASKS=ROOT/'artifacts/evaluate_results/long36_ten_vs_789_20260916_231154_2gpu/ten_guidance_789/seed_7/libero_plus_long_36_three_perturbations_two_variants.txt'
BASE=Path('/data/pxchen/JEPA_WAM/checkpoints/vjepa2/vjepa2-ac-vitg.pt')
def clean(sd):return {k.replace('module.','').replace('backbone.',''):v for k,v in sd.items()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--inference',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    c=torch.load(TRAIN,map_location='cpu',weights_only=False,mmap=True)
    cal=json.loads(CAL.read_text())
    assert c['step']==2000 and c['config']['calibration_config']==cal
    assert c['config']['image_rotation_degrees']==180 and c['config']['data_format_version']=='libero_ac_euler_response_calibrated_v1'
    exported=torch.load(args.inference,map_location='cpu',weights_only=True,mmap=True)
    assert exported['fine_tune_step']==2000 and exported['fine_tune_checkpoint']==str(TRAIN)
    expected=clean(c['predictor']);actual=clean(exported['predictor'])
    assert expected.keys()==actual.keys()
    assert all(torch.equal(expected[k],actual[k]) and torch.isfinite(actual[k]).all() for k in expected)
    base=torch.load(BASE,map_location='cpu',weights_only=True,mmap=True)
    expected_enc=clean(base['encoder']);exported_enc=clean(exported['encoder'])
    assert expected_enc.keys()==exported_enc.keys()
    assert all(torch.equal(expected_enc[k],exported_enc[k]) for k in expected_enc)
    ps=cal['osc_position_scale']*cal['translation_response_gain']
    rs=cal['osc_rotation_scale']*cal['rotation_response_gain']
    adapter=LiberoACAdapter(osc_position_scale=ps,osc_rotation_scale=rs)
    a=np.random.default_rng(239).uniform(-.9,.9,(3,8,7)).astype(np.float32);a[0]=0
    state=np.array([.3,0,.4,.1,-.1,.2,.3],dtype=np.float32)
    ref=adapter.convert(a,state);t=torch.tensor(a,requires_grad=True);result=adapter.convert_torch(t,torch.tensor(state))
    for key in ('actions','states','terminal_states'):
        np.testing.assert_allclose(getattr(ref,key),getattr(result,key).detach().numpy(),atol=2e-6,rtol=1e-4)
    sum(getattr(result,k).square().sum() for k in ('actions','states','terminal_states')).backward()
    assert torch.isfinite(t.grad).all()
    tasks=[x.strip() for x in TASKS.read_text().splitlines() if x.strip()]
    assert len(tasks)==len(set(tasks))==36 and all(x.startswith('libero_10,') for x in tasks)
    (args.out/'tasks36.txt').write_text('\n'.join(tasks)+'\n')
    (args.out/'smoke_task.txt').write_text(tasks[0]+'\n')
    manifest=[dict(suite=x.split(',')[0],task_id=int(x.split(',')[1]),seed=s) for s in (7,17,27,37) for x in tasks]
    config=dict(fastwam_checkpoint=str(ROOT/'checkpoints/fastwam/libero_optional_idm_2cam224.pt'),
                jepa_inference=str(args.inference),predictor_source=str(TRAIN),encoder_source=str(BASE),
                calibration=cal,osc_position_scale=ps,osc_rotation_scale=rs,
                seeds=[7,17,27,37],episodes=144,task_source=str(TASKS),
                fastwam_dtype='bfloat16',jepa_dtype='float32',main_view_only=True,
                wam_image_orientation='rot180',additional_jepa_encoder_rotation=0,
                guidance='full every replan; trigger disabled; raw gradient4x steps7/8/9',
                max_delta_rms=.02,ac_steps=2,low_level_steps_per_ac_step=4,replan_steps=8,
                denoise_steps=10,video_action_shift=[1,1],normalize_gradient=False,
                verify_descent=True,overlap_encoder_with_action=True,compile_action=False,compile_predictor=False,
                max_steps_override=None,gripper='unchanged historical handling',
                arb_module_enabled=False,new_latent_idm_module_enabled=False,
                trials_per_task_seed=1,smoke_max_steps=48)
    (args.out/'run_config.json').write_text(json.dumps(config,indent=2))
    (args.out/'task_manifest144.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'preflight.json').write_text(json.dumps(dict(passed=True,predictor_export_exact=True,encoder_export_exact=True,numpy_torch_adapter_match=True,finite_action_gradient=True,episodes=144),indent=2))
    print(json.dumps(config,indent=2))

if __name__=='__main__':main()
