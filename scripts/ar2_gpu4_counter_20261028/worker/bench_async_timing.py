"""Isolated current-AR2 compile, fixed-input latency and critical-path audit."""
from wand.api import library as _wand_library
import concurrent.futures,copy,functools,hashlib,json,os,statistics,threading,time,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from async_guidance_bridge import PrecomputedActionGradient
from stage_timer import StageTimer
from action_energy import make_action_energy,execution_gripper
import eval_weighted_six as shared
ROOT=Path('/data/pxchen/PACT-WAM');OUT=Path(os.environ['GUIDANCE_SIX_OUT'])
BASE=ROOT/'evaluate_results/white_mug_robot42_seed7_sameT5_20261025'
NS=None;PREENCODE=None
def save(name,x):
 p=OUT/name;q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
def status(s,**kw):save('progress.json',dict(state=s,updated=time.strftime('%F %T'),**kw));print(s,kw,flush=True)
def tensor(x):
 if isinstance(x,torch.Tensor):return x.detach().float().cpu()
 return torch.as_tensor(x).float()
def diff(a,b):
 a=tensor(a);b=tensor(b);return dict(max_abs=float((a-b).abs().max()),rms=float((a-b).square().mean().sqrt()),relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-12)),finite=bool(torch.isfinite(a).all()))
def preencode(cfg,*,device,dtype):
 from experiments.libero.text_context_cache import load_cache,record_audit
 m=json.loads((BASE/'text_cache_manifest.json').read_text());cache,checksum=load_cache(m['path'],m['identity'],NS['_evaluation_prompts'](cfg),m['sha256'])
 record_audit(OUT,cache,m['identity'],'fixed_existing_robot42_T5',m['path'],checksum);return cache
def ev():return torch.cuda.Event(enable_timing=True)

def driver(ns,runner,**kw):
 cfg=kw['cfg'];model=kw['model'];ranker=kw['ac_ranker'];adapter=kw['ac_adapter'];device=ranker.device
 assert not cfg.EVALUATION.compile_action_infer and not cfg.EVALUATION.vjepa2_ac.guidance.compile_predictor
 assert cfg.seed==7 and cfg.EVALUATION.vjepa2_ac.guidance.verify_descent
 bench=ns['benchmark'].get_benchmark_dict()['libero_10'](task_order_index=0)
 ns['set_global_seed'](7,get_worker_init_fn=False);cfg.EVALUATION.task_id=559
 env,desc=ns['get_libero_env'](bench.get_task(559),ns['LIBERO_ENV_RESOLUTION'],7)
 try:
  env.reset();ns['apply_libero_camera'](env,cfg.EVALUATION.vjepa2_ac.camera)
  obs=env.set_init_state(bench.get_task_init_states(559)[0])
  for _ in range(30):obs,_,_,_=env.step(ns['get_libero_dummy_action']())
 finally:env.close()
 pair=json.loads((BASE/'paired_preflight/task559.json').read_text());assert shared.observation_digest(obs)==pair['observation']
 save('input_identity.json',dict(task_id=559,seed=7,observation=pair['observation'],T5=json.loads((BASE/'text_cache_manifest.json').read_text())['sha256'],scope='Fixed initial state; no environment rollout; host camera/preprocessing and recording excluded from standalone model-pipeline measurements.'))
 ranker.active_task_id=559
 capture={};v_orig=model.infer_video;a_orig=model.infer_action_from_video;builder=ns['_build_jepa_action_guidance']
 @functools.wraps(v_orig)
 def capture_video(*a,**k):
  assert not a;capture['video_kwargs']=dict(k);r=v_orig(**k);capture['latents']=r['video_latents'].detach().clone();return r
 @functools.wraps(a_orig)
 def capture_action(*a,**k):
  assert not a;capture['action_kwargs']=dict(k);r=a_orig(**k);capture['raw_action']=r['action'].detach().clone();return r
 @functools.wraps(builder)
 def capture_build(**k):capture['builder_kwargs']=dict(k);return builder(**k)
 model.infer_video=capture_video;model.infer_action_from_video=capture_action;ns['_build_jepa_action_guidance']=capture_build
 args=dict(obs=obs,previous_jepa_image=None,task_description=desc,model=model,processor=kw['processor'],cfg=cfg,action_horizon=32,input_w=kw['input_w'],input_h=kw['input_h'],model_device=kw['model_device'],executed_horizon=8,replan_index=0,ac_ranker=ranker,ac_adapter=adapter)
 status('EAGER_REFERENCE')
 try:
  reference_result=ns['_predict_action_chunk'](**args);reference=reference_result[0]
  save('eager_native_reference_timing.json',reference_result[3])
 finally:model.infer_video=v_orig;model.infer_action_from_video=a_orig;ns['_build_jepa_action_guidance']=builder
 old=ROOT/'evaluate_results/feedback_weights_6task_seed7_sameT5_20261021/preflight_task559_guided_actions.npy'
 assert np.array_equal(reference,np.load(old)),'Eager historical guided first plan drift'
 np.save(OUT/'eager_reference_commands.npy',reference)
 vk=capture['video_kwargs'];ak=capture['action_kwargs'];bk=capture['builder_kwargs'];latents=capture['latents']
 for k in ['video_latents','action_guidance_fn']:ak.pop(k,None)
 vk['compile_video_infer']=True;vk['decode_video']=False;ak['compile_action_infer']=True
 ranker.active_task_id=559
 save('shapes.json',dict(video_latents=list(latents.shape),current_transform=str(ranker.encoder.image_transform),guidance=dict(nodes=[7,8,9],scale=4,cap=.02,verify_descent=True,weights='original fixed white-mug table',ar_chain='P1 then P2 without detach')))
 # Keep all compiler state isolated, and disable CUDA graph pools for multi-stream
 # correctness and predictable memory use; tensor kernels are still compiled.
 options={'triton.cudagraphs':False}
 def compile_fn(fn,fullgraph=True):return torch.compile(fn,fullgraph=fullgraph,dynamic=False,options=options)
 model._encode_input_image_latents_tensor=compile_fn(model._encode_input_image_latents_tensor,False)
 # The legacy public decode copies latent data through CPU. Bypass that wrapper
 # for the GPU-native pipeline, retaining the identical neural decoder/scale.
 # encode/decode clear shared feature caches: use a distinct state owner with
 # shared read-only weights so concurrent action-image encode cannot reset it.
 decode_model=copy.copy(model.vae.model);decode_model.clear_cache()
 assert decode_model._feat_map is not model.vae.model._feat_map
 assert decode_model._enc_feat_map is not model.vae.model._enc_feat_map
 def decode_core(z):return decode_model.decode(z,model.vae.scale).clamp(-1,1)
 decode_compiled=compile_fn(decode_core,False)
 prefix_count=1+(3-1+3)//4
 # Preserve the full-shape conv2 call: slicing before it selects different
 # BF16 convolution kernels and changes pixels. Only stop decoder chunks early.
 def decode_prefix_core(z):
  from fastwam.models.wan22.wan_video_vae import unpatchify
  decode_model.clear_cache()
  scale=[v.to(dtype=z.dtype,device=z.device) for v in model.vae.scale]
  z=z/scale[1].view(1,decode_model.z_dim,1,1,1)+scale[0].view(1,decode_model.z_dim,1,1,1)
  x=decode_model.conv2(z)
  for i in range(prefix_count):
   decode_model._conv_idx=[0]
   if i==0:
    out,decode_model._feat_map,decode_model._conv_idx=decode_model.decoder(x[:,:,i:i+1],feat_cache=decode_model._feat_map,feat_idx=decode_model._conv_idx,first_chunk=True)
   else:
    extra,decode_model._feat_map,decode_model._conv_idx=decode_model.decoder(x[:,:,i:i+1],feat_cache=decode_model._feat_map,feat_idx=decode_model._conv_idx)
    out=torch.cat([out,extra],2)
  out=unpatchify(out,patch_size=2);decode_model.clear_cache()
  return out.clamp(-1,1)
 prefix_compiled=compile_fn(decode_prefix_core,False)
 def decode(z,prefix,compiled):
  fn=(prefix_compiled if compiled else decode_prefix_core) if prefix else (decode_compiled if compiled else decode_core)
  with torch.no_grad():r=fn(z)
  return r[:,:,:3]
 full=decode(latents,False,False);prefix=decode(latents,True,False)
 prefix_error=diff(prefix,full);save('prefix_decode_validation.json',prefix_error)
 assert prefix_error['max_abs']==0,'Prefix VAE does not preserve causal first3 frames'
 # Match the original uint8 conversion before canonical PIL bicubic processing.
 def as_pil(rgb):
  y=((rgb[0].float().clamp(-1,1)+1)*127.5).to(torch.uint8).cpu()
  return [Image.fromarray(y[:,i].permute(1,2,0).numpy()) for i in range(3)]
 original_frames=bk['predicted_future_frames'][:3]
 original_pixel_error=max(int(np.abs(np.asarray(a).astype(np.int16)-np.asarray(b).astype(np.int16)).max()) for a,b in zip(as_pil(full),original_frames))
 save('gpu_decode_original_validation.json',dict(max_uint8_difference=original_pixel_error,independent_mutable_cache_owner=True,shared_read_only_decoder_weights=True))
 assert original_pixel_error==0,'GPU-native full decode differs from original decoder pixels'
 current=ranker.pixels(bk['current_jepa_image'])[None].to(device,torch.float32)
 def canonical_pixels(rgb):
  frames=as_pil(rgb);v=torch.stack([ranker.pixels(f) for f in frames]).to(device,torch.float32)
  return torch.stack([v[:2],v[1:3]])
 mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
 std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
 def gpu_pixels(rgb):
  y=((rgb[0].float().clamp(-1,1)+1)*127.5).to(torch.uint8).permute(1,0,2,3)
  if y.shape[-1]==2*y.shape[-2]:y=y[...,:y.shape[-2]]
  # Tensor bicubic is checked against original PIL; it is never silently
  # substituted into the exact-comparison pipeline if it changes the result.
  y=F.interpolate(y.float(),size=(384,384),mode='bicubic',align_corners=False,antialias=True).round().clamp(0,255)/255
  y=(y-mean)/std
  return torch.stack([y[:2],y[1:3]])
 # The deprecated generator wrapper is unsupported by Dynamo fullgraph.
 # Its installed implementation delegates to this same modern context with
 # exactly these flags/order. Change only this benchmark process.
 def compatible_sdp_kernel(enable_flash=True,enable_math=True,enable_mem_efficient=True,enable_cudnn=True):
  backends=[];B=torch.nn.attention.SDPBackend
  if enable_flash:backends.append(B.FLASH_ATTENTION)
  if enable_mem_efficient:backends.append(B.EFFICIENT_ATTENTION)
  if enable_math:backends.append(B.MATH)
  if enable_cudnn:backends.append(B.CUDNN_ATTENTION)
  return torch.nn.attention.sdpa_kernel(backends)
 with torch.no_grad():old_encoder=ranker.encoder(current).detach().clone()
 torch.backends.cuda.sdp_kernel=compatible_sdp_kernel
 with torch.no_grad():new_encoder=ranker.encoder(current).detach().clone()
 context_error=diff(new_encoder,old_encoder);save('attention_context_validation.json',context_error)
 assert context_error['max_abs']==0,'Modern SDPA context changes encoder output'
 def encoder_core(c,p):return ranker.encoder(c).detach(),ranker.encoder.encode_pair(p).reshape(2,576,1024).detach()
 def current_core(c):return ranker.encoder(c).detach()
 def future_core(p):return ranker.encoder.encode_pair(p).reshape(2,576,1024).detach()
 joint_compiled=compile_fn(encoder_core);current_compiled=compile_fn(current_core);future_compiled=compile_fn(future_core)
 am,ast=ranker.stats['action'];sm,sst=ranker.stats['state'];weights=ranker.task_weights[559]
 def energy_core(z,target,actions,states,w):
  actions=(actions-am)/ast;state=(states-sm)/sst
  p1=ranker.predictor(z.expand(len(actions),-1,-1),actions[:,:4],state,0)
  p2=ranker.predictor(p1,actions[:,4:8],state,1)
  return torch.stack([((p1-target[0]).abs()*w[0]).mean((1,2)),((p2-target[1]).abs()*w[1]).mean((1,2))],1).mean(1)
 energy_compiled=compile_fn(energy_core)
 energy_eager=lambda z,t,a,s:energy_core(z,t,a,s,weights)
 normalizer=kw['processor'].normalizer.normalizers['action'][kw['processor'].shape_meta['action'][0]['key']]
 scale=normalizer.scale.to(device=device,dtype=torch.float32);offset=normalizer.offset.to(device=device,dtype=torch.float32)
 raw_scale=normalizer.scale[...,6:7].to(device);raw_offset=normalizer.offset[...,6:7].to(device)
 if raw_scale.ndim>1:raw_scale=raw_scale[:8];raw_offset=raw_offset[:8]
 initial=torch.as_tensor(adapter.state_from_observation(obs),device=device,dtype=torch.float32)
 binarize=bool(cfg.EVALUATION.binarize_gripper)
 # Exercise the discrete boundary independently of the three warmup actions.
 scale_cpu=normalizer.scale[...,6:7].cpu();offset_cpu=normalizer.offset[...,6:7].cpu()
 center=(offset_cpu+scale_cpu*.5).float();centers=center.expand(8,1) if center.ndim==1 else center[:8]
 cases=torch.zeros(260,8,7);cases[:257,:,6]=centers[:,0][None]+torch.linspace(-1e-5,1e-5,257)[:,None]
 cases[257,:,6]=centers[:,0];cases[258,:,6]=torch.nextafter(centers[:,0],torch.full_like(centers[:,0],float('inf')));cases[259,:,6]=torch.nextafter(centers[:,0],torch.full_like(centers[:,0],-float('inf')))
 expected=ns['_denormalize_action'](cases,kw['processor'])[...,6:7];expected=-(expected*2-1);expected=np.sign(expected) if binarize else expected
 grip_boundary_compiled=compile_fn(lambda c:execution_gripper(c,raw_scale,raw_offset,binarize))
 boundary_error=diff(grip_boundary_compiled(cases.to(device)),expected);save('gripper_boundary_validation.json',dict(cases=2080,**boundary_error));assert boundary_error['max_abs']==0
 action_energy_eager=make_action_energy(energy_eager,scale,offset,raw_scale,raw_offset,initial,binarize)
 action_energy_compiled=compile_fn(action_energy_eager)
 gpu_preprocess_compiled=compile_fn(gpu_pixels)
 # Quantify preprocessing drift explicitly. GPU resizing is a separate,
 # authorized treatment; never call it bitwise equivalent to canonical PIL.
 pixels_ref=canonical_pixels(prefix);pixels_gpu=gpu_pixels(prefix)
 preprocess_error=diff(pixels_gpu,pixels_ref)
 save('gpu_preprocess_validation.json',dict(tensors=preprocess_error,exact_equivalence=preprocess_error['max_abs']==0,canonical='PIL resize via ranker.pixels',candidate='GPU bicubic antialias at 384px; unchanged encoder/checkpoint'))
 active=None;phase='';captured_gradient=None;action_image_ready=None;prefill_ready=None;full_checks={};gripper_checks=[]
 def measured_energy(current_rep,target_reps,ac_actions,ac_states):
  fn=lambda:energy_compiled(current_rep,target_reps,ac_actions,ac_states,weights)
  kind='predictor_forward_gradient' if torch.is_grad_enabled() and ac_actions.requires_grad else 'predictor_forward_verification'
  return active.call(kind,fn) if active is not None else fn()
 ranker.differentiable_energy=measured_energy
 # WAM subcomponents are timed around their existing compiled calls.
 def install_stage(attribute,fn,name):
  compiled=compile_fn(fn)
  @functools.wraps(fn)
  def measured(*a,**k):
   nonlocal prefill_ready
   call=lambda:compiled(*a,**k)
   value=active.call(name,call) if active is not None else call()
   if name=='video_cache_prefill' and prefill_ready is not None:
    callback=prefill_ready;prefill_ready=None;callback()
   return value
  setattr(model,attribute,measured)
 image_encode=model._encode_input_image_latents_tensor
 def timed_image(*a,**k):
  nonlocal action_image_ready
  if phase=='action' and action_image_ready is not None:
   callback=action_image_ready;action_image_ready=None;callback()
  call=lambda:image_encode(*a,**k)
  return active.call('wam_input_vae_encode_'+phase,call) if active is not None else call()
 model._encode_input_image_latents_tensor=timed_image
 install_stage('_denoise_video_compiled',model._denoise_video,'video_denoise_step')
 install_stage('_prefill_video_cache_compiled',model.mot.prefill_video_cache_tensor,'video_cache_prefill')
 install_stage('_denoise_action_with_video_cache_compiled',model._denoise_action_with_video_cache,'action_denoise_step')
 step_with_guidance=model.infer_action_scheduler.step_with_loss_guidance
 def timed_correction(*a,**k):return active.call('scheduler_guidance_update',lambda:step_with_guidance(*a,**k))
 model.infer_action_scheduler.step_with_loss_guidance=timed_correction
 original_encode=ranker.encode_guidance_context
 auxiliary=torch.cuda.Stream(device=device)
 pool=concurrent.futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix='jepa-aux')
 roles={'main':threading.get_native_id()};save('thread_roles.json',roles)
 # Dynamo/FX tracing must finish before another Python thread calls compiled
 # functions. Bootstrap each complete path on this host thread; CUDA stream
 # selection is unchanged. Only bootstrap is serial, never measured samples.
 serial_warmup=True
 def submit_branch(fn):
  if not serial_warmup:return pool.submit(fn)
  future=concurrent.futures.Future()
  try:future.set_result(fn())
  except BaseException as error:future.set_exception(error)
  return future
 MODES=['no_guidance','full_gpu_verify0_serial','full_gpu_verify0_async','full_gpu_verify0_prefill_first']
 samples=[]
 def iteration(mode,record=False):
  nonlocal active,phase,captured_gradient,action_image_ready,prefill_ready
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();timer=StageTimer(device);active=timer;start=time.perf_counter();main=torch.cuda.current_stream(device);auxiliary.wait_event(timer.origin)
  guided=mode!='no_guidance';serial=mode.endswith('_serial');early=guided and not serial;aux_predictor=guided;full_path=mode.startswith('full_');gpu_images=mode.startswith('full_gpu_');verify='verify0' not in mode;joint=False;priority=mode.endswith('_prefill_first');release_event=None;first_guidance_seen=False
  def prepare_guidance():
   local_bk=dict(bk);local_bk.update(predicted_future_frames=[None,None,None],overlap_encoder_with_action=False,lazy_encoder=True)
   return builder(**local_bk)[0]
  # The native builder copies the observation state to CUDA synchronously.
  # Do this before launching either branch, so the copy cannot serialize the
  # newly queued VAE decode with Action Gen. Keep its cost inside wall timing.
  prepared=timer.call('guidance_setup',prepare_guidance) if guided else None
  def current_input():
   c=timer.call('current_cpu_preprocess',lambda:ranker.pixels(bk['current_jepa_image'])[None],gpu=False)
   return timer.call('current_h2d',lambda:c.to(device,torch.float32))
  def current_branch():
   if not serial_warmup and 'auxiliary' not in roles:roles['auxiliary']=threading.get_native_id();save('thread_roles.json',roles)
   with torch.cuda.device(device),torch.cuda.stream(auxiliary),torch.no_grad():
    c=current_input();value=timer.call('current_jepa_encoder',lambda:current_compiled(c));done=ev();done.record();return value,done
  current_future=submit_branch(current_branch) if guided and early else None
  phase='video'
  with torch.no_grad():pred=timer.call('video_total',lambda:v_orig(**dict(vk,compile_video_infer=True)))
  z=pred['video_latents'];video_ready=ev();video_ready.record()
  def future_branch():
   with torch.cuda.device(device),torch.cuda.stream(auxiliary),torch.no_grad():
    auxiliary.wait_event(video_ready);z.record_stream(auxiliary)
    if priority:
     assert release_event is not None
     token=timer.begin('aux_prefill_dependency_wait');auxiliary.wait_event(release_event);timer.end(token)
    rgb=timer.call('prefix_decode',lambda:decode(z,True,True))
    if gpu_images:pairs=timer.call('future_gpu_preprocess',lambda:gpu_preprocess_compiled(rgb))
    else:
     y=timer.call('future_uint8_conversion',lambda:((rgb[0].float().clamp(-1,1)+1)*127.5).to(torch.uint8))
     y=timer.call('future_d2h',lambda:y.cpu())
     def cpu_pixels():
      frames=[Image.fromarray(y[:,i].permute(1,2,0).numpy()) for i in range(3)]
      return torch.stack([ranker.pixels(f) for f in frames])
     pixels_cpu=timer.call('future_cpu_preprocess',cpu_pixels,gpu=False)
     pixels=timer.call('future_h2d',lambda:pixels_cpu.to(device,torch.float32))
     pairs=timer.call('future_pair_assembly',lambda:torch.stack([pixels[:2],pixels[1:3]]))
    if joint:
     c=current_input();encoded=timer.call('joint_jepa_encoder',lambda:joint_compiled(c,pairs))
    else:
     if early:
      current_rep,current_done=current_future.result();auxiliary.wait_event(current_done)
     else:
      c=current_input();current_rep=timer.call('current_jepa_encoder',lambda:current_compiled(c))
     target=timer.call('future_jepa_encoder',lambda:future_compiled(pairs));encoded=current_rep,target
    timer.call('encoded_context_ready',lambda:None)
    done=ev();done.record();return encoded,done
  future=submit_branch(future_branch) if guided and serial else None
  if future is not None:
   _,done=future.result();main.wait_event(done)
  def submit_when_action_inputs_ready():
   nonlocal future
   # Native Action Gen has now completed its CPU-random-noise H2D copy.
   # Queue Decode before its first VAE encode call, so the two actual GPU
   # workloads can start together without an intervening blocking host copy.
   future=timer.call('future_branch_submit',lambda:submit_branch(future_branch),gpu=False)
  def submit_after_prefill():
   nonlocal release_event
   release_event=ev();release_event.record(main)
   submit_when_action_inputs_ready()
  action_image_ready=submit_when_action_inputs_ready if guided and not serial and not priority else None
  prefill_ready=submit_after_prefill if guided and priority else None
  try:
   if guided:
    def resolve(**unused):
     token=timer.begin('encoder_dependency_wait');encoded,done=future.result();torch.cuda.current_stream(device).wait_event(done);timer.end(token);return encoded
    ranker.encode_guidance_context=resolve
    native_guidance=prepared if prepared is not None else timer.call('guidance_setup',prepare_guidance)
    def full_guidance(clean):
     token=timer.begin('full_context_dependency_wait')
     encoded,done=future.result();torch.cuda.current_stream(device).wait_event(done)
     timer.end(token)
     return timer.call('compiled_action_to_energy',lambda:action_energy_compiled(clean,*encoded))
    selected_guidance=full_guidance if full_path else native_guidance
    def guidance(clean):
     nonlocal captured_gradient,first_guidance_seen
     needs_gradient=torch.is_grad_enabled() and clean.requires_grad
     if needs_gradient and not first_guidance_seen:
      timer.call('first_guidance_action_ready',lambda:None);first_guidance_seen=True
     if not aux_predictor:
      name='guidance_forward_total' if needs_gradient else 'guidance_verification_total'
      loss=timer.call(name,lambda:selected_guidance(clean))
      if needs_gradient:
       pending={}
       def begin_backward(g):pending['token']=timer.begin('guidance_backward_total');return g
       def end_backward(g):timer.end(pending['token']);return g
       loss.register_hook(begin_backward);clean.register_hook(end_backward)
      return loss
     ready=ev();ready.record(main);clean.record_stream(auxiliary)
     wait=timer.begin('guidance_aux_dependency_wait')
     def evaluate(source):
      nonlocal captured_gradient
      with torch.cuda.device(device),torch.cuda.stream(auxiliary):
       auxiliary.wait_event(ready)
       with torch.enable_grad():
        separate=source.detach().clone().requires_grad_(True)
        loss=timer.call('guidance_forward_total',lambda:selected_guidance(separate))
        gradient=timer.call('guidance_backward_total',lambda:torch.autograd.grad(loss,separate,only_inputs=True)[0])
       if full_path and serial_warmup:
        cpu_g=ns['_denormalize_action'](separate.detach()[:,:8],kw['processor'])[...,6:7];cpu_g=-(cpu_g*2-1);cpu_g=np.sign(cpu_g) if binarize else cpu_g
        gpu_g=execution_gripper(separate,raw_scale,raw_offset,binarize)
        error=diff(gpu_g,cpu_g);gripper_checks.append(dict(mode=mode,**error));assert error['max_abs']==0,'GPU execution gripper differs from CPU execution'
        if mode not in full_checks:
         encoded,_=future.result()
         with torch.enable_grad():
          ref=separate.detach().clone().requires_grad_(True);ref_loss=native_guidance(ref);ref_grad=torch.autograd.grad(ref_loss,ref)[0]
          eager=separate.detach().clone().requires_grad_(True);eager_loss=action_energy_eager(eager,*encoded);eager_grad=torch.autograd.grad(eager_loss,eager)[0]
         check=dict(native_loss=diff(loss,ref_loss),native_gradient=diff(gradient,ref_grad),eager_loss=diff(loss,eager_loss),eager_gradient=diff(gradient,eager_grad),gripper_gradient_max=float(gradient[...,6].abs().max()))
         full_checks[mode]=check;save('full_path_validation.json',full_checks)
         assert all(check[k]['max_abs']<1e-5 for k in ['native_loss','native_gradient','eager_loss','eager_gradient']) and check['gripper_gradient_max']==0
       if captured_gradient is None:captured_gradient=(separate.detach().clone(),loss.detach().clone(),gradient.detach().clone(),native_guidance)
       done=ev();done.record();gradient.record_stream(main)
      main.wait_event(done);timer.end(wait);return loss,gradient
     if needs_gradient:return PrecomputedActionGradient.apply(clean,evaluate)
     with torch.cuda.device(device),torch.cuda.stream(auxiliary),torch.no_grad():
      auxiliary.wait_event(ready);loss=timer.call('guidance_verification_total',lambda:selected_guidance(clean));done=ev();done.record()
     main.wait_event(done);timer.end(wait);return loss
   local_ak=dict(ak);local_ak.update(compile_action_infer=True,action_guidance_verify_descent=verify,action_guidance_fn=guidance if guided else None)
   if not guided:local_ak.update(action_guidance_last_steps=0,action_guidance_after_steps=None)
   phase='action'
   with torch.no_grad():result=timer.call('action_total',lambda:a_orig(video_latents=z,**local_ak))
   if future is not None:future.result()
   if current_future is not None:current_future.result()
   torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000
   collected=timer.collect()
   markers={x['name']:x for x in collected['intervals']}
   encoder_wait=max(0,markers['encoded_context_ready']['gpu_end_ms']-markers['first_guidance_action_ready']['gpu_start_ms']) if guided else 0
   row=dict(mode=mode,wall_ms=wall,started_unix_s=time.time()-wall/1000,finished_unix_s=time.time(),verify_descent=verify,gpu_images=gpu_images,full_action_energy=full_path,serial_components=serial,prefill_priority=priority,first_guidance_encoder_wait_ms=encoder_wait,**collected,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
   # The hash transfer follows the synchronized wall-time boundary. It does
   # not enter latency and verifies every measured action, not just warmups.
   row['raw_action_sha256']=hashlib.sha256(result['action'].detach().float().cpu().numpy().tobytes()).hexdigest()
   if record:
    samples.append(row)
    with (OUT/'latency_samples.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   return row,result['action'].detach().clone()
  finally:
   # Complete pending auxiliary work before restoring shared callback state.
   if future is not None:
    try:future.result()
    except BaseException:pass
   if current_future is not None:
    try:current_future.result()
    except BaseException:pass
   ranker.encode_guidance_context=original_encode;active=None;action_image_ready=None;prefill_ready=None
 # Keep compilation serial until both encoders and gradient paths are warm.
 cold=[];actions={};status('WARMING_COMPONENTS_FROM_EXISTING_CACHE')
 for mode in MODES:
  t=time.perf_counter();row,action=iteration(mode);actions[mode]=action;cold.append(dict(mode=mode,elapsed_s=time.perf_counter()-t,metrics=row));save('warmup_costs.json',cold);status('PATH_READY',mode=mode,elapsed_s=cold[-1]['elapsed_s'])
 serial_warmup=False
 for mode,value in actions.items():np.save(OUT/('warmup_'+mode+'_raw.npy'),value.detach().float().cpu().numpy())
 comparisons={mode:diff(value,actions['full_gpu_verify0_async']) for mode,value in actions.items() if mode!='no_guidance'}
 save('scheduling_validation.json',comparisons)
 assert all(v['finite'] for v in comparisons.values())
 same_gpu_actions=diff(actions['full_gpu_verify0_prefill_first'],actions['full_gpu_verify0_async']);save('prefill_priority_action_validation.json',same_gpu_actions);assert same_gpu_actions['max_abs']<1e-5
 save('gripper_execution_validation.json',gripper_checks)
 clean,loss,gradient,base_guidance=captured_gradient
 with torch.enable_grad():
  clean_ref=clean.detach().clone().requires_grad_(True);ref_loss=base_guidance(clean_ref);ref_grad=torch.autograd.grad(ref_loss,clean_ref,only_inputs=True)[0]
 gradient_check=dict(loss=diff(loss,ref_loss),gradient=diff(gradient,ref_grad),gripper_gradient_max=float(gradient[...,6].abs().max()))
 save('aux_gradient_validation.json',gradient_check)
 assert gradient_check['gradient']['max_abs']<1e-5 and gradient_check['loss']['max_abs']<1e-5 and gradient_check['gripper_gradient_max']==0
 status('WARMING_STEADY_PATHS')
 for mode in MODES:
  for _ in range(3):iteration(mode)
 from torch._dynamo.utils import counters
 save('compiler_counters_before_measure.json',{str(k):{str(a):int(b) for a,b in v.items()} for k,v in counters.items()})
 status('MEASURING',samples_per_mode=20)
 for i in range(20):
  for mode in MODES:iteration(mode,True)
  if i%5==0:status('MEASURING',completed_per_mode=i+1,total=20)
 summary={}
 for mode in MODES:
  rs=[r for r in samples if r['mode']==mode];names=set().union(*(r['gpu_span_ms'] for r in rs));hostnames=set().union(*(r['host_call_ms'] for r in rs))
  summary[mode]=dict(n=len(rs),median_ms=float(np.median([r['wall_ms'] for r in rs])),p95_ms=float(np.percentile([r['wall_ms'] for r in rs],95)),first_guidance_encoder_wait_median_ms=float(np.median([r['first_guidance_encoder_wait_ms'] for r in rs])),gpu_span_median_ms={k:float(np.median([r['gpu_span_ms'].get(k,0) for r in rs])) for k in names},host_call_median_ms={k:float(np.median([r['host_call_ms'].get(k,0) for r in rs])) for k in hostnames})
 hashes={mode:sorted({r['raw_action_sha256'] for r in samples if r['mode']==mode}) for mode in MODES};save('measured_action_hashes.json',hashes)
 assert hashes['full_gpu_verify0_async']==hashes['full_gpu_verify0_prefill_first'] and len(hashes['full_gpu_verify0_async'])==1,'Measured actions changed across scheduling modes'
 save('latency_summary.json',dict(variants=summary,scheduling_validation=comparisons,gradient_validation=gradient_check,full_path_validation=full_checks,preprocess_validation=preprocess_error,requested_mode='full_gpu_verify0_prefill_first',guidance_setup='Before video generation, identical in both guided modes',future_branch_launch='Control: native action input VAE entry. Treatment: after compiled prefill returns to host, auxiliary stream waits on explicit prefill-complete CUDA event.',current_encoder_launch='Before video generation in both guided modes; unchanged',compiler_scope='Same fully compiled action-to-energy, FP32 JEPA/AR2, BF16 WAM, GPU preprocessing and verification disabled as v5',cache='Reuse existing compiler cache; warm all modes serially before concurrency',timing='CUDA-event and host spans include waits; nested/overlapping spans must not be summed',scope='Fixed559 seed7 same T5/weights/input; only Decode+future branch release point changes between guided modes; no success-rate rollouts'))
 save('compiler_counters_after_measure.json',{str(k):{str(a):int(b) for a,b in v.items()} for k,v in counters.items()})
 assert json.loads((OUT/'compiler_counters_before_measure.json').read_text())==json.loads((OUT/'compiler_counters_after_measure.json').read_text()),'Compilation during steady measurements'
 status('HARDWARE_COUNTER_CAPTURE')
 profile_rows=[]
 torch.cuda.synchronize()
 torch.cuda.cudart().cudaProfilerStart()
 try:
  for repeat in range(5):
   for mode in MODES:
    torch.cuda.nvtx.range_push('iteration/'+mode)
    try:row,_=iteration(mode)
    finally:torch.cuda.nvtx.range_pop()
    profile_rows.append(row)
 finally:
  torch.cuda.synchronize()
  torch.cuda.cudart().cudaProfilerStop()
 save('counter_iterations.json',profile_rows)
 pool.shutdown(wait=True);status('COMPLETED');return None,dict(successes=0,duration=0)

def main():
 native=ROOT/'experiments/libero/eval_libero_single.py';manifest=json.loads((BASE/'manifest.json').read_text());assert hashlib.sha256(native.read_bytes()).hexdigest()==manifest['native_sha256']
 code=native.read_text();entry='if __name__ == "__main__":\n    eval_single_process()';assert code.count(entry)==1
 def install(ns):
  global NS
  NS=ns;shared.NS=ns;shared.OUT=OUT
  ns['VJEPA2ACRanker']=shared.ARRanker;ns['LiberoACAdapter']=shared.RawAdapter
  ns['_evaluation_task_keys']=lambda cfg:[('libero_10',559)];ns['_preencode_prompt_contexts']=preencode
  original=ns['_run_task_to_file']
  if os.environ.get('ASYNC_PRELOAD_CPU')=='1':
   ns['_resolve_eval_device']=lambda cfg:'cpu'
   def run_ready(**kw):
    from cpu_model_ready import acquire
    return driver(ns,original,**acquire(kw))
   ns['_run_task_to_file']=run_ready
  else:ns['_run_task_to_file']=lambda **kw:driver(ns,original,**kw)
 code=code.replace(entry,'if __name__ == "__main__":\n    install_benchmark(globals())\n    eval_single_process()')
 exec(compile(code,str(native),'exec'),dict(__name__='__main__',__file__=str(native),install_benchmark=install))
if __name__=='__main__':
 try:main()
 except BaseException:save('failure.json',dict(traceback=traceback.format_exc()));status('FAILED');raise
