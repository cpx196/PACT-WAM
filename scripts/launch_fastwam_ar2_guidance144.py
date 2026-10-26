"""Single GPU, historical 36x4 manifest, smoke-gated nohup evaluation."""
import os,sys,json,subprocess,fcntl,shutil,time
from pathlib import Path
from omegaconf import OmegaConf
ROOT=Path('/data/pxchen/PACT-WAM')
OLD=ROOT/'evaluate_results/guidance789_rawgrad4x_long36_full144_libero10_calibrated_step2000_20261007'
OUT=ROOT/'evaluate_results/fastwam_guidance789_ar2_epoch10_full144_20261015'
PRED=ROOT/'artifacts/fastwam_predictor21_ar2_full10epoch_bs16_20261014/step19690.pt'
GPU=sys.argv[1]
OUT.mkdir(parents=True,exist_ok=True)
lock=open(OUT/'launcher.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OUT/'smoke').exists(), 'Refuse duplicate evaluation'
for name in ('tasks36.txt','smoke_task.txt','task_manifest144.json'):
    shutil.copy2(OLD/name,OUT/name)
config=json.loads((OLD/'run_config.json').read_text())
config.update(jepa_inference=str(PRED),predictor_source=str(PRED),predictor_epoch=10,predictor_step=19690,
    encoder_source=str(ROOT/'checkpoints/jepa_wam_assets_20261009/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt'),
    jepa_input='raw 8x7 executable commands + XYZ/rotvec/two finger qpos8',
    jepa_prediction='2-step autoregression, pairs [t,t+4] and [t+4,t+8]',
    jepa_normalization='train-only AR2 stats; no extra output layernorm',
    replaced_components=['JEPA Encoder','JEPA Predictor','necessary JEPA-only input adapter'],
    gpu=int(GPU),compile=False)
(OUT/'run_config.json').write_text(json.dumps(config,indent=2))
(OUT/'launcher.pid').write_text(str(os.getpid()))
(OUT/'gpu_assignment.txt').write_text(f'GPU{GPU}; seeds7,17,27,37 sequential; single GPU\n')
def status(s):
    (OUT/'status.txt').write_text(s+'\n');print(time.strftime('%F %T')+' '+s,flush=True)
def run(seed,smoke=False):
    # Preserve all historical overrides verbatim, changing only JEPA checkpoint/interface.
    previous=OmegaConf.load(OLD/'seed_7/manager_config.yaml')
    keys=['EVALUATION','model']
    def flat(v,prefix):
        if isinstance(v,dict):
            for k,x in v.items():yield from flat(x,prefix+'.'+k)
        else:
            yield prefix+'='+json.dumps(v,separators=(',',':'))
    # Use precisely the explicitly configured historical launcher fields.
    fields=['dataset_stats_path','replan_steps','num_inference_steps','compile_action_infer','visualize_future_video','offload_text_encoder','action_infer_mode','video_sigma_shift','action_sigma_shift','sigma_shift']
    overrides=[f'EVALUATION.{k}='+json.dumps(previous.EVALUATION[k]) for k in fields]
    overrides+=list(flat(OmegaConf.to_container(previous.EVALUATION.vjepa2_ac,resolve=True),'EVALUATION.vjepa2_ac'))
    # The additional encoder rotation field is not in the original structured schema.
    overrides=[('++'+s if s.startswith('EVALUATION.vjepa2_ac.encoder_image_rotation_degrees=') else s) for s in overrides]
    overrides=[s for s in overrides if not s.startswith('EVALUATION.vjepa2_ac.checkpoint_path=')]
    overrides += ['EVALUATION.vjepa2_ac.checkpoint_path='+str(PRED),'EVALUATION.trigger_probe.enabled=false',
                  'model.load_text_encoder=true','model.redirect_common_files=false','model.skip_dit_load_from_pretrain=true']
    dest=OUT/('smoke' if smoke else f'seed_{seed}');dest.mkdir()
    cmd=[sys.executable,'scripts/fastwam_ar2_manager.py','task=libero_optional_idm_2cam224_1e-4',
         'ckpt='+str(config['fastwam_checkpoint']),'EVALUATION.output_dir='+str(dest),
         'MULTIRUN.task_file='+str(OUT/('smoke_task.txt' if smoke else 'tasks36.txt')),
         'MULTIRUN.num_gpus=1','EVALUATION.num_trials=1',
         'EVALUATION.max_steps_override='+('48' if smoke else 'null'),'seed='+str(seed),*overrides]
    (dest/'command.json').write_text(json.dumps(cmd,indent=2))
    with (dest/'run.log').open('w') as log:
        subprocess.run(cmd,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,check=True)
try:
    status('SMOKE_RUNNING');run(7,True)
    # Historical verifier checks actual gradients at all 3 nodes and the RMS cap.
    with (OUT/'smoke_check.log').open('w') as log:
        subprocess.run([sys.executable,'scripts/verify_calibrated_guidance144_smoke_20261007.py','--root',str(OUT)],stdout=log,stderr=subprocess.STDOUT,check=True)
    status('FULL144_RUNNING')
    for seed in (7,17,27,37):
        status(f'FULL144_RUNNING seed={seed}');run(seed)
    status('COMPLETED')
except BaseException as exc:
    status('FAILED '+repr(exc));raise
