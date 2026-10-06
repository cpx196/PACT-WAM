"""Three-step differentiable guidance unroll, isolated from production code."""
import argparse, ast, functools, gc, inspect, json, logging, sys, textwrap, types, traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT=Path('/data/pxchen/PACT-WAM')
OUT=ROOT/'artifacts/libero10_guidance_backward_smoke_20260929'
sys.path.insert(0,str(ROOT))

def log(event,**kw): print(json.dumps(dict(event=event,**kw)),flush=True)

def ranker_cpu():
    from experiments.libero.vjepa2_ac_ranker import VJEPA2ACRanker
    return VJEPA2ACRanker(checkpoint_path=str(ROOT/'checkpoints/vjepa2-ac-libero90-euler-fixed-rot180-step2000-inference.pt'),
          vjepa2_repo='/data/pxchen/JEPA_WAM',device='cpu',dtype=torch.float32,max_ac_steps=2)

def encode():
    ranker=ranker_cpu(); ranker.device=torch.device('cuda'); ranker.encoder.cuda()
    rows=torch.load(OUT/'wam_windows.pt',map_location='cpu',weights_only=False)
    for r in rows:
        w=r['wam_future']
        with torch.no_grad():
            r['reps']=ranker.encode_clips([r['context'],[w[0],w[1]],[w[1],w[2]]]).cpu()
        log('JOINT_CONTEXT_ENCODED',demo=r['window']['demo'])
    torch.save(rows,OUT/'joint_encoded_windows.pt')
    log('JOINT_ENCODING_PASSED',windows=len(rows))

def make_train_method(model):
    import fastwam.models.wan22.fastwam_idm as module
    # Optional-IDM runtime's public infer_action is a dispatcher; differentiate
    # the actual IDM implementation called by infer_action_from_video instead.
    tree=ast.parse(textwrap.dedent(inspect.getsource(module.FastWAMIDM.infer_action)))
    tree.body[0].decorator_list=[]
    source=ast.unparse(tree)
    old=').detach().requires_grad_(True)'
    assert source.count(old)==1
    source=source.replace(old,').requires_grad_(True)')
    assert source.count('only_inputs=True')==1
    source=source.replace('only_inputs=True','only_inputs=True, create_graph=True')
    assert source.count('guidance_diagnostics = []')==1
    source=source.replace('guidance_diagnostics = []','guidance_diagnostics = []\n    training_nodes = []\n    training_clean_actions = []')
    marker='horizon = latents_action.shape[1]'
    assert marker in source
    source=source.replace(marker,'training_nodes.append(loss_gradient)\n            training_clean_actions.append(clean_action)\n            '+marker)
    old="latents_action.detach().to(device='cpu', dtype=torch.float32)"
    assert source.count(old)==1
    source=source.replace(old,'latents_action.to(dtype=torch.float32)')
    source=source.replace('return result',"result['training_nodes'] = training_nodes\n    result['training_clean_actions'] = training_clean_actions\n    return result")
    (OUT/'three_step_train_method.txt').write_text(source)
    namespace=dict(module.__dict__)
    exec(compile(source,'<isolated_three_step_train_method>','exec'),namespace)
    return types.MethodType(namespace['infer_action'],model)

def joint():
    from experiments.libero import eval_libero_single as ev
    from experiments.libero.vjepa2_ac_ranker import LiberoACAdapter
    cfg=OmegaConf.load(ROOT/'evaluate_results/guidance789_rawgrad4x_long36_full144_euler_fixed_step2000_20260929/seed_7/manager_config.yaml')
    cfg.EVALUATION.output_dir=str(OUT/'joint_cache')
    cfg.EVALUATION.vjepa2_ac.enabled=False
    cfg.EVALUATION.vjepa2_ac.guidance.enabled=False
    cfg.EVALUATION.vjepa2_ac.num_action_candidates=1
    cfg.EVALUATION.trigger_probe.enabled=False
    cfg.EVALUATION.action_denoise_trace=False
    cfg.MULTIRUN.task_file=None
    description='pick up the book and place it in the back compartment of the caddy'
    ev._evaluation_prompts=lambda cfg:[ev.DEFAULT_PROMPT.format(task=description)]
    rows=torch.load(OUT/'joint_encoded_windows.pt',map_location='cpu',weights_only=False)

    def train(**ctx):
        model=ctx['model']; model.requires_grad_(False)
        differentiable=make_train_method(model)
        capture={}; original=model.infer_action_from_video
        @functools.wraps(original)
        def wrapped(**kw):
            capture['kw']=kw.copy()
            return original(**kw)
        model.infer_action_from_video=wrapped
        ranker=ranker_cpu()
        del ranker.encoder
        ranker.device=torch.device('cuda')
        predictor=ranker.predictor.cuda().eval().requires_grad_(True)
        predictor.extrinsics_encoder.requires_grad_(False)
        predictor.use_activation_checkpointing=True
        parameters=[p for p in predictor.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=1e-6,weight_decay=0.04)
        adapter=LiberoACAdapter(low_level_steps_per_ac_step=4)
        metrics=[]
        tracked=predictor.action_encoder.weight
        before=tracked.detach().clone()
        for i,r in enumerate(rows,1):
            # Frozen WAM conditioning is generated at this exact expert state.
            with torch.no_grad():
                ev._predict_action_chunk(r['obs'],r['context'][0],description,model,ctx['processor'],cfg,
                    action_horizon=ctx['action_horizon'],input_w=ctx['input_w'],input_h=ctx['input_h'],
                    model_device=ctx['model_device'],executed_horizon=8)
            kw=capture['kw'].copy()
            kw['precomputed_video_latents']=kw.pop('video_latents')
            assert kw.pop('action_infer_mode','idm')=='idm'
            kw.pop('compile_action_infer',None)
            scale,offset=r['scale'].cuda(),r['offset'].cuda()
            state=torch.as_tensor(adapter.state_from_observation(r['obs']),device='cuda')
            reps=r['reps'].cuda()
            def energy(a):
                ac=adapter.convert_torch((a[:,:8]-offset)/scale,state)
                return ranker.differentiable_energy(current_rep=reps[0],target_reps=reps[1:],
                       ac_actions=ac.actions,ac_states=ac.states).mean()
            kw.update(action_guidance_fn=energy,action_guidance_after_steps=[7,8,9],
                action_guidance_step_size=4.0,action_guidance_horizon=8,
                action_guidance_normalize_gradient=False,action_guidance_max_delta_rms=0.02,
                action_guidance_verify_descent=False,action_flow_trace=False)
            optimizer.zero_grad(set_to_none=True)
            # Frozen ActionExpert stays differentiable with respect to the action input.
            # CPU saved tensors keep a full three-step chain within a24GiB card.
            with torch.enable_grad(),torch.backends.cuda.sdp_kernel(enable_flash=False,enable_math=True,enable_mem_efficient=False),torch.autograd.graph.save_on_cpu(pin_memory=True):
                result=differentiable(**kw)
                output=result['action']
                if output.ndim==2: output=output[None]
                loss=F.mse_loss(output[:,:8,:6],r['expert'].cuda()[None,:,:6])
                nodes=result['training_nodes']; assert len(nodes)==3
                node_grads=torch.autograd.grad(loss,nodes,retain_graph=True)
                norms=[float(g.float().norm()) for g in node_grads]
                assert all(torch.isfinite(g).all() and g.norm()>0 for g in node_grads)
                loss.backward()
            assert tracked.grad is not None and torch.isfinite(tracked.grad).all() and tracked.grad.norm()>0
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
            row=dict(window=i,action_loss=float(loss.detach()),guidance_node_grad_norms=norms,
                  predictor_action_encoder_grad_norm=float(tracked.grad.norm()),
                  guidance_steps=[d['flow_step'] for d in result['action_guidance']],
                  peak_gib=torch.cuda.max_memory_allocated()/2**30)
            torch.nn.utils.clip_grad_norm_(parameters,1.0)
            optimizer.step();metrics.append(row);log('THREE_STEP_JOINT_UPDATE',**row)
            del result,loss,nodes,node_grads,output
            gc.collect();torch.cuda.empty_cache()
        change=float((tracked.detach()-before).norm()); assert change>0
        report=dict(passed=True,updates=len(metrics),metrics=metrics,parameter_change_norm=change,
                    frozen_fastwam=True,encoder_frozen=True,final_action_dtype='original BF16 output cast then FP32 loss',
                    attention='math SDPA',saved_tensors='CPU',prediction_loss_in_supervision=False)
        (OUT/'three_step_joint_report.json').write_text(json.dumps(report,indent=2))
        log('THREE_STEP_JOINT_PASSED',**report)
        return None,dict(successes=0,duration=0)
    ev._run_task_to_file=train
    ev.eval_single_process(cfg)

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['encode','joint'],required=True)
    args=p.parse_args()
    try: (encode if args.phase=='encode' else joint)()
    except Exception:
        traceback.print_exc();sys.exit(1)
