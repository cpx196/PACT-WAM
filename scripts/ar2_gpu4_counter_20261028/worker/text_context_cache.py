"""Immutable, fingerprinted CPU T5 contexts for reproducible LIBERO evaluation."""
from contextlib import contextmanager
import fcntl,hashlib,json,os
from pathlib import Path
import torch

FORMAT=1

def tensor_sha256(value):
 return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def file_sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def cache_identity(text_path,tokenizer_path,context_len,dtype):
 text=Path(text_path).resolve();st=text.stat();tokenizer=Path(tokenizer_path).resolve()
 token_files={p.name:file_sha256(p) for p in sorted(tokenizer.iterdir()) if p.is_file() and p.name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','spiece.model','config.json')}
 return dict(text_checkpoint=str(text),text_checkpoint_bytes=st.st_size,text_checkpoint_mtime_ns=st.st_mtime_ns,tokenizer=str(tokenizer),tokenizer_files=token_files,context_len=int(context_len),dtype=str(dtype),encoding_batch_size=1,torch_version=str(torch.__version__))
def entry_fingerprints(cache):
 return {prompt:dict(context=tensor_sha256(context),mask=tensor_sha256(mask),context_shape=list(context.shape),mask_shape=list(mask.shape)) for prompt,(context,mask) in sorted(cache.items())}
def validate_cache(cache,identity,fingerprints):
 if not isinstance(cache,dict) or not cache:raise ValueError('Empty or invalid text context cache')
 for prompt,pair in cache.items():
  if not isinstance(prompt,str) or not isinstance(pair,(tuple,list)) or len(pair)!=2:raise ValueError('Invalid cache entry')
  context,mask=pair
  if context.device.type!='cpu' or mask.device.type!='cpu':raise ValueError('Text cache must contain CPU tensors')
  if context.ndim!=3 or context.shape[0]!=1 or mask.shape!=context.shape[:2]:raise ValueError('Invalid context/mask shape')
  if str(context.dtype)!=identity['dtype'] or mask.dtype!=torch.bool:raise ValueError('Invalid context/mask dtype')
  if not torch.isfinite(context).all():raise ValueError('Nonfinite context')
 if entry_fingerprints(cache)!=fingerprints:raise ValueError('Text cache tensor fingerprint mismatch')
@contextmanager
def locked(path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.with_suffix(path.suffix+'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);yield path

def save_cache(cache,path,identity):
 cache={key:(a.detach().cpu().contiguous(),b.detach().cpu().contiguous()) for key,(a,b) in cache.items()}
 fingerprints=entry_fingerprints(cache);validate_cache(cache,identity,fingerprints)
 with locked(path) as p:
  if p.exists():raise FileExistsError(f'Refuse to overwrite text context cache: {p}')
  temp=p.with_name(p.name+f'.{os.getpid()}.tmp')
  try:
   torch.save(dict(format=FORMAT,identity=identity,fingerprints=fingerprints,contexts=cache),temp);temp.replace(p)
  finally:
   if temp.exists():temp.unlink()
  checksum=file_sha256(p)
  manifest=p.with_suffix(p.suffix+'.json');manifest.write_text(json.dumps(dict(format=FORMAT,identity=identity,fingerprints=fingerprints,sha256=checksum),ensure_ascii=False,indent=2))
  return checksum

def load_cache(path,identity,prompts,expected_sha256=None):
 with locked(path) as p:
  if not p.is_file():raise FileNotFoundError(f'Pinned text context cache missing: {p}')
  checksum=file_sha256(p)
  if expected_sha256 and checksum!=expected_sha256:raise ValueError('Pinned text context cache file hash mismatch')
  payload=torch.load(p,map_location='cpu',weights_only=True)
 if payload.get('format')!=FORMAT or payload.get('identity')!=identity:raise ValueError('Text cache identity mismatch: assets/config/runtime changed')
 cache=payload['contexts'];validate_cache(cache,identity,payload['fingerprints'])
 missing=set(prompts)-cache.keys()
 if missing:raise ValueError(f'Text cache missing {len(missing)} requested prompts')
 return {prompt:cache[prompt] for prompt in prompts},checksum

def record_audit(output_dir,cache,identity,mode,path=None,checksum=None):
 folder=Path(os.path.expanduser(os.path.expandvars(str(output_dir))));folder.mkdir(parents=True,exist_ok=True)
 artifact=folder/f'text_context_fingerprints_{os.getpid()}.json'
 artifact.write_text(json.dumps(dict(mode=mode,cache_path=str(path) if path else None,cache_sha256=checksum,identity=identity,entries=entry_fingerprints(cache)),ensure_ascii=False,indent=2))
 return artifact
