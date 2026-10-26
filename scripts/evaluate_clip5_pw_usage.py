import sys,json,time,argparse,fcntl
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/data/pxchen/PACT-WAM/scripts')
import train_jepawam_arb_pilot100 as t

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default=str(t.ROOT/'jepawam_arb_clip5_pilot100_20261013'))
    parser.add_argument('--checkpoint',default='step100.pt')
    args=parser.parse_args()
    torch.set_num_threads(2)
    root=Path(args.root)
    lock=open(root/'pw_usage_evaluation.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'pw_usage_evaluation.json').exists():raise RuntimeError('Evaluation result already exists')
    t.NO_PW=False
    model=t.ARB(json.load(open(t.BASE/'train_only_stats.json'))).cuda().eval()
    ck=torch.load(root/args.checkpoint,map_location='cpu',weights_only=False)
    assert not ck['config'].get('no_pw',False)
    model.load_state_dict(ck['model'],strict=True)
    del ck
    ds=t.Windows('val')
    dl=t.loader(ds,list(range(len(ds))))
    modes=['full','zero_P','zero_W','zero_both','shuffle_both']
    sums={k:np.zeros(6) for k in modes}; changes={k:0.0 for k in modes}
    order=np.random.default_rng(714).permutation(len(ds)).tolist()
    donor=t.loader(ds,order)
    count=0;start=time.time()
    with torch.inference_mode():
        for i,(cpu,dcpu) in enumerate(zip(dl,donor)):
            b=t.device(cpu);d=t.device(dcpu);reference=None
            for mode in modes:
                v=dict(b)
                if mode in ('zero_P','zero_both'):v['P']=torch.zeros_like(b['P'])
                if mode in ('zero_W','zero_both'):v['W']=torch.zeros_like(b['W'])
                if mode=='shuffle_both':v['P']=d['P'];v['W']=d['W']
                with torch.autocast('cuda',dtype=torch.bfloat16):out,_=model(v)
                e=(out-b['target'][...,:6])/model.action_std[:6]
                sums[mode]+=e.square().sum((0,1)).cpu().numpy()
                if reference is None:reference=out
                changes[mode]+=float(((out-reference)/model.action_std[:6]).square().sum())
            count+=b['action'].shape[0]*20
            if i%100==0:print(json.dumps({'batches':i+1,'windows':count//20,'seconds':time.time()-start}),flush=True)
    assert count//20==len(ds)==3500
    result={'windows':count//20,'checkpoint':str(root/args.checkpoint),'evaluation_only':True,
            'shuffle':'P/W jointly replaced by a globally permuted validation donor; state/action/target unchanged',
            'results':{k:{'normalized_mse':float(sums[k].mean()/count),'per_dim':(sums[k]/count).tolist(),
                          'normalized_output_change_rms':float((changes[k]/(count*6))**.5)} for k in modes},
            'seconds':time.time()-start}
    expected=json.load(open(root/'summary.json'))['validation'][-1]['normalized_mse']
    assert abs(result['results']['full']['normalized_mse']-expected)<1e-6,'Checkpoint baseline mismatch'
    temp=root/'pw_usage_evaluation.tmp'
    temp.write_text(json.dumps(result,indent=2));temp.replace(root/'pw_usage_evaluation.json')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
