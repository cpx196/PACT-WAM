"""Require real online guidance in steps7/8/9 before starting full144."""
import argparse
import ast
import json
import math
from pathlib import Path
from omegaconf import OmegaConf

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    expected=json.loads((args.root/'run_config.json').read_text())
    cfg=OmegaConf.load(args.root/'smoke/manager_config.yaml')
    ac=cfg.EVALUATION.vjepa2_ac;g=ac.guidance
    assert cfg.ckpt==expected['fastwam_checkpoint']
    assert ac.checkpoint_path==expected['jepa_inference'] and ac.dtype=='float32'
    assert abs(ac.osc_position_scale-expected['osc_position_scale'])<1e-12
    assert abs(ac.osc_rotation_scale-expected['osc_rotation_scale'])<1e-12
    assert ac.encoder_image_rotation_degrees==0 and ac.camera.image_rotation_degrees==180
    assert g.enabled and not g.trigger.enabled and not g.compile_predictor
    assert list(g.after_flow_steps)==[7,8,9] and g.step_size==4 and not g.normalize_gradient
    assert g.ac_steps==2 and g.max_delta_rms==.02 and g.verify_descent
    assert cfg.EVALUATION.max_steps_override==48 and not cfg.EVALUATION.compile_action_infer
    assert cfg.EVALUATION.video_sigma_shift==cfg.EVALUATION.action_sigma_shift==1
    results=list((args.root/'smoke/libero_10').glob('gpu*_task*_results.json'))
    assert len(results)==1,'Smoke must complete exactly one truncated episode'
    json.loads(results[0].read_text())
    records=[]
    for log in (args.root/'smoke').rglob('*.log'):
        for line in log.read_text(errors='replace').splitlines():
            marker='V-JEPA2-AC action guidance diagnostics='
            if marker in line:
                values=ast.literal_eval(line.split(marker,1)[1])
                assert [r['after_flow_step'] for r in values]==[7,8,9]
                for r in values:
                    assert math.isfinite(r['loss']) and math.isfinite(r['gradient_rms']) and math.isfinite(r['correction_rms'])
                    assert not r['normalize_gradient'] and r['correction_rms']<=.020001
                    if r.get('loss_after_clean_step') is not None:assert math.isfinite(r['loss_after_clean_step'])
                records.extend(values)
    assert len(records)>=3 and any(r['gradient_rms']>1e-8 for r in records)
    report=dict(passed=True,guidance_nodes=len(records),flow_steps=[7,8,9],
                max_correction_rms=max(r['correction_rms'] for r in records),
                mean_gradient_rms=sum(r['gradient_rms'] for r in records)/len(records),
                truncated_episode_result=str(results[0]),smoke_success_not_required=True,
                calibration_verified=True,trigger_disabled=True)
    (args.root/'smoke_check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
