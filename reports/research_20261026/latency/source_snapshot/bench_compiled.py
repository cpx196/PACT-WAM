"""Isolated current-AR2 compile, fixed-input latency and critical-path audit."""
from wand.api import library as _wand_library
import concurrent.futures,copy,functools,hashlib,json,os,statistics,threading,time,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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
 gpu_pixels=compile_fn(gpu_pixels)
 encoder_compiled=compile_fn(encoder_core)
 am,ast=ranker.stats['action'];sm,sst=ranker.stats['state'];weights=ranker.task_weights[559]
 def energy_core(z,target,actions,states,w):
  actions=(actions-am)/ast;state=(states-sm)/sst
  p1=ranker.predictor(z.expand(len(actions),-1,-1),actions[:,:4],state,0)
  p2=ranker.predictor(p1,actions[:,4:8],state,1)
  return torch.stack([((p1-target[0]).abs()*w[0]).mean((1,2)),((p2-target[1]).abs()*w[1]).mean((1,2))],1).mean(1)
 energy_compiled=compile_fn(energy_core)
 energy_events=[]
 def compiled_energy(current_rep,target_reps,ac_actions,ac_states):
  fs,fe=ev(),ev();fs.record();loss=energy_compiled(current_rep,target_reps,ac_actions,ac_states,weights);fe.record()
  rec={'forward_start':fs,'forward_end':fe}
  if ac_actions.requires_grad and loss.requires_grad:
   bs,be=ev(),ev();rec.update(backward_start=bs,backward_end=be)
   loss.register_hook(lambda grad:(bs.record(),grad)[1]);ac_actions.register_hook(lambda grad:(be.record(),grad)[1])
  energy_events.append(rec);ranker.calls+=1;return loss
 ranker.differentiable_energy=compiled_energy
 original_encode=ranker.encode_guidance_context
 stream=torch.cuda.Stream(device=device)
 samples=[]
 def iteration(mode,compiled=True,record=True,gpu_preprocess=False,verify=True):
  nonlocal energy_events
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();energy_events=[]
  stages={};stage_wall={};start=time.perf_counter()
  def mark(name,fn):
   a,b=ev(),ev();a.record();t=time.perf_counter()
   with torch.profiler.record_function(name):value=fn()
   b.record();stages[name]=(a,b);stage_wall[name]=time.perf_counter()-t;return value
  local_vk=dict(vk);local_vk['compile_video_infer']=compiled
  with torch.no_grad():pred=mark('video',lambda:v_orig(**local_vk))
  z=pred['video_latents'];main=torch.cuda.current_stream();ready=ev();ready.record()
  future=concurrent.futures.Future();waits=[];thread=None
  def encode_branch():
   try:
    with torch.cuda.device(device),torch.cuda.stream(stream if mode=='async' else main),torch.no_grad():
     if mode=='async':stream.wait_event(ready)
     rgb=mark('decode_prefix' if mode!='full_serial' else 'decode_full',lambda:decode(z,mode!='full_serial',compiled))
     pixels=mark('future_preprocess',lambda:gpu_pixels(rgb) if gpu_preprocess else canonical_pixels(rgb))
     encoded=mark('jepa_encoder',lambda:(encoder_compiled if compiled else encoder_core)(current,pixels))
     done=ev();done.record();future.set_result((encoded,done))
   except BaseException as e:future.set_exception(e)
  guided=mode!='no_guidance'
  if guided:
   if mode=='async':thread=threading.Thread(target=encode_branch);thread.start()
   else:encode_branch()
   def resolve(**unused):
    ws,we=ev(),ev();ws.record();t=time.perf_counter();value,done=future.result();main.wait_event(done);we.record();waits.append((ws,we,time.perf_counter()-t));return value
   ranker.encode_guidance_context=resolve
   local_bk=dict(bk);local_bk.update(predicted_future_frames=[None,None,None],overlap_encoder_with_action=False,lazy_encoder=True)
   try:guidance,horizon=builder(**local_bk)
   finally:pass
  local_ak=dict(ak);local_ak['compile_action_infer']=compiled
  local_ak['action_guidance_verify_descent']=verify
  local_ak['action_guidance_fn']=guidance if guided else None
  if not guided:
   local_ak['action_guidance_last_steps']=0;local_ak['action_guidance_after_steps']=None
  with torch.no_grad():result=mark('action_including_guidance_and_wait',lambda:a_orig(video_latents=z,**local_ak))
  if thread is not None:thread.join()
  torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000
  ranker.encode_guidance_context=original_encode
  row=dict(mode=mode,compiled=compiled,gpu_preprocess=gpu_preprocess,verify_descent=verify,wall_ms=wall,cuda_ms={k:a.elapsed_time(b) for k,(a,b) in stages.items()},host_dispatch_ms={k:v*1000 for k,v in stage_wall.items()},encoder_wait_gpu_ms=sum(a.elapsed_time(b) for a,b,_ in waits),encoder_wait_host_ms=sum(t*1000 for _,_,t in waits),energy_forward_ms=sum(r['forward_start'].elapsed_time(r['forward_end']) for r in energy_events),energy_backward_ms=sum(r['backward_start'].elapsed_time(r['backward_end']) for r in energy_events if 'backward_start' in r),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
  if record:
   samples.append(row)
   with (OUT/'latency_samples.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  return row,result['action'].detach().clone()
 # Compile WAM kernels, prefix/full VAE, encoder and energy+AOT backward via
 # complete production-shaped calls. Record cold-start separately.
 cold=[];status('COMPILING_CURRENT_CORES')
 for mode in ['no_guidance','full_serial','prefix_serial','async']:
  t=time.perf_counter();row,action=iteration(mode,record=False);cold.append(dict(mode=mode,wall_s=time.perf_counter()-t,metrics=row));save('cold_start.json',cold)
  status('COMPILE_PATH_READY',mode=mode,cold_s=cold[-1]['wall_s'])
 # Same compiled program and fixed RNG inputs: asynchronous scheduling must
 # preserve actions, otherwise it cannot be reported as equivalent speedup.
 _,serial=iteration('prefix_serial',record=False);_,asynchronous=iteration('async',record=False)
 scheduling=diff(asynchronous,serial);save('scheduling_validation.json',scheduling)
 assert scheduling['max_abs']<1e-5,'Async schedule changes deterministic compiled action'
 # Record compiled/eager output drift, rather than hiding it behind cache reuse.
 eager_raw=capture['raw_action'];drift=diff(serial,eager_raw);save('compiled_vs_eager.json',drift)
 assert drift['finite'] and drift['max_abs']<.05,'Compiled action drift exceeds 0.05 normalized-action tolerance'
 cp=canonical_pixels(prefix);gp=gpu_pixels(prefix);pixel_drift=diff(gp,cp)
 _,canonical_action=iteration('async',record=False,gpu_preprocess=False)
 _,gpu_action=iteration('async',record=False,gpu_preprocess=True)
 gpu_drift=diff(gpu_action,canonical_action);gpu_equivalent=gpu_drift['max_abs']<1e-5
 save('gpu_preprocess_validation.json',dict(pixels=pixel_drift,actions=gpu_drift,equivalent_at_1e_5=gpu_equivalent,note='GPU/PIL bicubic rounding may differ; GPU path is exploratory unless action-equivalent.'))
 modes=[('no_guidance',False,True),('full_serial',False,True),('prefix_serial',False,True),('async',False,True),('async',True,True),('async',False,False)]
 status('WARMING_UP')
 for mode,gpu,verify in modes:
  for _ in range(3):iteration(mode,record=False,gpu_preprocess=gpu,verify=verify)
 status('MEASURING_STEADY_STATE',samples_per_variant=20)
 # Interleave variants to reduce ordering effects from shared host/clock drift.
 for repeat in range(20):
  for mode,gpu,verify in modes:iteration(mode,gpu_preprocess=gpu,verify=verify)
  if repeat%5==0:status('MEASURING_STEADY_STATE',completed_per_variant=repeat+1,total=20)
 grouped={}
 for mode,gpu,verify in modes:
  rows=[r for r in samples if r['mode']==mode and r['gpu_preprocess']==gpu and r['verify_descent']==verify]
  key=f'{mode}_gpu{int(gpu)}_verify{int(verify)}';values=[r['wall_ms'] for r in rows]
  grouped[key]=dict(n=len(rows),median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),mean_ms=float(np.mean(values)),stage_cuda_median_ms={k:float(np.median([r['cuda_ms'].get(k,0) for r in rows])) for k in set().union(*(r['cuda_ms'] for r in rows))},wait_median_ms=float(np.median([r['encoder_wait_gpu_ms'] for r in rows])),energy_forward_median_ms=float(np.median([r['energy_forward_ms'] for r in rows])),energy_backward_median_ms=float(np.median([r['energy_backward_ms'] for r in rows])))
 save('latency_summary.json',dict(variants=grouped,scope='Fixed initial559/seed7, same cached T5, warmed compiled current WAM+JEPA2.1+AR2+weighted L1 forward/backward; no task success evaluation.',gpu_preprocess_equivalent=gpu_equivalent,scheduling_validation=scheduling,compiled_vs_eager=drift))
 from torch._dynamo.utils import counters
 save('compiler_counters.json',{str(k):{str(a):int(b) for a,b in v.items()} for k,v in counters.items()})
 status('PROFILING_TRACE')
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True,with_stack=False) as prof:
  for _ in range(2):iteration('async',record=False)
 prof.export_chrome_trace(str(OUT/'compiled_async_trace.json'))
 (OUT/'operator_profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
 async_result=grouped['async_gpu0_verify1'];serial_result=grouped['prefix_serial_gpu0_verify1'];base=grouped['no_guidance_gpu0_verify1']
 full_result=grouped['full_serial_gpu0_verify1'];noverify=grouped['async_gpu0_verify0']
 bottleneck=dict(video_cuda_ms=async_result['stage_cuda_median_ms']['video'],action_span_cuda_ms=async_result['stage_cuda_median_ms']['action_including_guidance_and_wait'],encoder_dependency_wait_ms=async_result['wait_median_ms'],energy_forward_ms=async_result['energy_forward_median_ms'],energy_backward_ms=async_result['energy_backward_median_ms'],prefix_decode_wall_saving_ms=full_result['median_ms']-serial_result['median_ms'],overlap_wall_saving_ms=serial_result['median_ms']-async_result['median_ms'],verification_wall_overhead_ms=async_result['median_ms']-noverify['median_ms'],encoder_slowdown_in_overlap=async_result['stage_cuda_median_ms']['jepa_encoder']/max(serial_result['stage_cuda_median_ms']['jepa_encoder'],1e-9))
 bottleneck['dominant_measured_span']='action_including_guidance_and_wait' if bottleneck['action_span_cuda_ms']>bottleneck['video_cuda_ms'] else 'video'
 bottleneck['limitation']='Component slowdown is timing evidence of overlap interference, not proof of compute or DRAM saturation. Profiler trace distinguishes kernels and synchronization; bandwidth cause requires hardware counters.'
 save('bottleneck_metrics.json',bottleneck)
 note=['# Current AR2 compiled latency / bottleneck audit','',f"No guidance median: {base['median_ms']:.2f} ms",f"Prefix serial median: {serial_result['median_ms']:.2f} ms",f"Prefix async median: {async_result['median_ms']:.2f} ms",f"Async encoder wait median: {async_result['wait_median_ms']:.2f} ms",f"Guidance energy forward: {async_result['energy_forward_median_ms']:.2f} ms; backward: {async_result['energy_backward_median_ms']:.2f} ms",'', 'Stage CUDA durations overlap; do not sum them as end-to-end latency. The action span includes guidance and dependency wait. Compare full/prefix serial for decode saving, prefix serial/async for overlap, verify true/false for verification overhead. GPU preprocessing is exploratory unless validation reports equivalence. Cold compile and profiler overhead are excluded from steady samples. This 4090D result does not establish Thor latency or task success.','', 'A low encoder wait does not prove no contention: compare component durations between serial and async. For kernel-level compute-vs-bandwidth attribution use hardware counters; torch.profiler alone cannot establish bandwidth saturation.']
 note+=['', '## 自动对照结果',f"Measured dominant span: {bottleneck['dominant_measured_span']}",f"Full-to-prefix serial wall reduction: {bottleneck['prefix_decode_wall_saving_ms']:.2f} ms",f"Serial-to-async wall reduction: {bottleneck['overlap_wall_saving_ms']:.2f} ms",f"Verification wall overhead: {bottleneck['verification_wall_overhead_ms']:.2f} ms",f"Encoder CUDA-span slowdown during overlap: {bottleneck['encoder_slowdown_in_overlap']:.3f}x",'Negative reductions indicate this optimization lost time for this fixed input. GPU-spans include queued execution and preemption; do not interpret them as isolated kernel service time.']
 (OUT/'bottleneck_report.md').write_text('\n'.join(note));status('COMPLETED')
 return None,dict(successes=0,duration=0)

def main():
 native=ROOT/'experiments/libero/eval_libero_single.py';manifest=json.loads((BASE/'manifest.json').read_text());assert hashlib.sha256(native.read_bytes()).hexdigest()==manifest['native_sha256']
 code=native.read_text();entry='if __name__ == "__main__":\n    eval_single_process()';assert code.count(entry)==1
 def install(ns):
  global NS,PREENCODE
  NS=ns;shared.NS=ns;shared.OUT=OUT
  ns['VJEPA2ACRanker']=shared.ARRanker;ns['LiberoACAdapter']=shared.RawAdapter
  ns['_evaluation_task_keys']=lambda cfg:[('libero_10',559)]
  ns['_preencode_prompt_contexts']=preencode
  original=ns['_run_task_to_file'];ns['_run_task_to_file']=lambda **kw:driver(ns,original,**kw)
 code=code.replace(entry,'if __name__ == "__main__":\n    install_benchmark(globals())\n    eval_single_process()')
 exec(compile(code,str(native),'exec'),dict(__name__='__main__',__file__=str(native),install_benchmark=install))
if __name__=='__main__':
 try:main()
 except BaseException:save('failure.json',dict(traceback=traceback.format_exc()));status('FAILED');raise
