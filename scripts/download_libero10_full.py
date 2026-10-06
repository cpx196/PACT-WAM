"""Download official LIBERO10 demonstrations only; no GPU or training work."""
import concurrent.futures
import datetime
import json
from pathlib import Path
import h5py
from huggingface_hub import HfApi, hf_hub_download

ROOT=Path('/data/pxchen/PACT-WAM/artifacts/libero10_guidance_backward_smoke_20260929')
DEST=ROOT/'data'
REPO='yifengzhu-hf/LIBERO-datasets'

def emit(event,**kw):
    print(json.dumps(dict(time=datetime.datetime.now().isoformat(),event=event,**kw)),flush=True)

def inspect_file(path,expected_size):
    assert path.stat().st_size==expected_size,(str(path),path.stat().st_size,expected_size)
    with h5py.File(path,'r') as h:
        demos=list(h['data'])
        lengths=[]
        for demo in demos:
            g=h['data'][demo]
            n=len(g['actions'])
            assert g['actions'].shape==(n,7),(path,demo,g['actions'].shape)
            for key in ['agentview_rgb','eye_in_hand_rgb','ee_states','gripper_states']:
                assert key in g['obs'],(path,demo,key)
                assert len(g['obs'][key])==n,(path,demo,key)
            assert g['obs']['ee_states'].shape[-1]==6
            lengths.append(n)
    return dict(path=str(path),bytes=expected_size,demos=len(demos),total_actions=sum(lengths),
                min_length=min(lengths),max_length=max(lengths))

def main():
    api=HfApi()
    revision=api.repo_info(REPO,repo_type='dataset').sha
    files=[x for x in api.list_repo_tree(REPO,repo_type='dataset',revision=revision,
           path_in_repo='libero_10') if x.path.endswith('.hdf5')]
    assert len(files)==10,len(files)
    emit('DOWNLOAD_PLAN',repo=REPO,revision=revision,files=10,total_bytes=sum(x.size for x in files),workers=2)
    def download(item):
        parts=Path(item.path).parts
        assert len(parts)==2 and parts[0]=='libero_10' and '..' not in parts
        path=DEST/item.path
        if path.exists() and path.stat().st_size==item.size:
            report=inspect_file(path,item.size)
            emit('EXISTING_FILE_VERIFIED',**report)
            return report
        emit('DOWNLOAD_START',file=item.path,bytes=item.size)
        result=Path(hf_hub_download(repo_id=REPO,repo_type='dataset',revision=revision,
                    filename=item.path,local_dir=str(DEST)))
        report=inspect_file(result,item.size)
        emit('FILE_VERIFIED',**report)
        return report
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        reports=list(pool.map(download,files))
    manifest=dict(repo=REPO,revision=revision,files=sorted(reports,key=lambda x:x['path']),
                  task_files=len(reports),demonstrations=sum(x['demos'] for x in reports),
                  total_actions=sum(x['total_actions'] for x in reports),
                  total_bytes=sum(x['bytes'] for x in reports),
                  completed_at=datetime.datetime.now().isoformat(),training_started=False)
    (ROOT/'libero10_full_manifest.json').write_text(json.dumps(manifest,indent=2))
    emit('LIBERO10_DOWNLOAD_COMPLETED',task_files=manifest['task_files'],demonstrations=manifest['demonstrations'],total_bytes=manifest['total_bytes'])

if __name__=='__main__':
    main()
