"""Portable fixed-recipe reproduction, independent of historical fitted heads/run IDs.

check: metadata only; extract: explicit CUDA encoder inference; fit: CPU full-label
regression refit; predict: CPU test inference. Nothing runs when imported.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
if not (ROOT/'final_recipe.json').exists():
    ROOT=ROOT.parents[1]/'submission/English/code'
sys.path.insert(0,str(ROOT/'scripts/experiments'))


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def unit(x):
    x=np.asarray(x,dtype=np.float64);norm=np.linalg.norm(x,axis=1,keepdims=True)
    if not np.isfinite(x).all() or (norm<=1e-12).any():raise ValueError('Invalid feature rows')
    return x/norm
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check():
    recipe=read(ROOT/'final_recipe.json')
    split=pd.read_csv(ROOT/'inputs/split_v1.csv')
    assert len(split)==8000 and split.imageid.is_unique
    assert split.partition.eq('train').sum()==6399 and split.partition.eq('validation').sum()==1601
    assert not set(split[split.partition.eq('train')].group_id)&set(split[split.partition.eq('validation')].group_id)
    assert abs(sum(recipe['weights'])-1)<1e-12
    for c in recipe['components']:
        path=ROOT/'.local/cache'/c['cache_family']/c['cache_key']/'manifest.json'
        m=read(path);assert m['feature_dim']==c['feature_dim']
        source='pe_core_multiview.py' if c['cache_family']=='pe_core_multiview' else 'pe_giant_whole_view.py'
        if c['cache_family']!='siglip2_naflex_features':
            assert digest(ROOT/'scripts/experiments'/source)==m['source_sha256']
        if 'original_cache_key' in m:
            family='pe_core_features' if c['cache_family']=='pe_core_multiview' else 'pe_core_giant_features'
            p=ROOT/'.local/cache'/family/m['original_cache_key']/'manifest.json'
            assert digest(p)==m['original_manifest_sha256']
    print('Portable recipe/split/encoder metadata verified; no weights or GPU loaded.')


def extract(args):
    import torch
    from PIL import Image,ImageOps
    from common.local_paths import local_path
    from visual_kernel_inference import encoder
    if not torch.cuda.is_available():raise RuntimeError('The original encoder precision/streaming contract requires CUDA')
    recipe=read(ROOT/'final_recipe.json');c=recipe['components'][args.component]
    manifest=read(ROOT/'.local/cache'/c['cache_family']/c['cache_key']/'manifest.json')
    rows=pd.read_csv(ROOT/('inputs/split_v1.csv' if args.partition=='train' else 'inputs/test.csv'))
    data=args.data_root.resolve() if args.data_root else local_path('data_root')
    folder=args.output/'features';folder.mkdir(parents=True,exist_ok=True)
    dest=folder/f'component_{args.component}_{args.partition}.npy'
    if dest.exists():raise FileExistsError(dest)
    torch.set_num_threads(2);torch.manual_seed(2026)
    encode=encoder(c['cache_family'],manifest);batch=int(manifest['batch_size']);parts=[]
    for offset in range(0,len(rows),batch):
        images=[]
        for name in rows.imageid.iloc[offset:offset+batch]:
            with Image.open(data/args.partition/name) as im:images.append(ImageOps.exif_transpose(im).convert('RGB'))
        values=encode(images)
        # Training caches were stored in F32 before F64 normalization by the heads.
        if args.partition=='train':values=values.astype(np.float32)
        parts.append(unit(values))
        if offset%400==0:print(args.component,args.partition,offset,len(rows),flush=True)
    values=np.concatenate(parts)
    assert values.shape==(len(rows),c['feature_dim'])
    with dest.open('xb') as stream:np.save(stream,values)
    write(dest.with_suffix('.json'),{'imageids':rows.imageid.tolist(),'component':args.component,'partition':args.partition,'sha256':digest(dest)})


def features(args,index,partition):
    path=args.output/'features'/f'component_{index}_{partition}.npy'
    metadata=read(path.with_suffix('.json'))
    ids=pd.read_csv(ROOT/('inputs/split_v1.csv' if partition=='train' else 'inputs/test.csv')).imageid.tolist()
    assert metadata['imageids']==ids and metadata['sha256']==digest(path)
    x=unit(np.load(path,allow_pickle=False))
    assert len(x)==len(ids)
    return x


def fit(args):
    import joblib
    from scipy.linalg import solve
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler
    from visual_kernel_components import kernel_for,scores
    from pe_whole_pca_bandwidth_kernel import eigensystem,calibrate_gamma,transform
    recipe=read(ROOT/'final_recipe.json');out=args.output/'fitted'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    y=pd.read_csv(ROOT/'inputs/split_v1.csv').price.to_numpy()/1000
    for i,c in enumerate(recipe['components']):
        x=features(args,i,'train');metric=None
        if i==2:
            mean,vectors,variance=eigensystem(x)
            bank=transform(x,mean,vectors,variance,256,0.)
            gamma,_,_=calibrate_gamma(x,bank)
            np.savez_compressed(out/f'component_{i}_pca.npz',mean=mean,vectors=vectors[:,:256],variance=variance[:256],whitening_power=0.)
            head={'train_features':bank,'gamma':gamma,'alpha':1.}
        elif i==1:
            pls=PLSRegression(n_components=16,scale=True,max_iter=500).fit(x,y)
            metric={'pls':pls,'scaler':StandardScaler().fit(pls.transform(x))}
            joblib.dump(metric,out/f'component_{i}_metric.joblib')
            head={'train_features':x,'image_gamma':1.,'pls_gamma':.1,'components':4,'alpha':1.,'train_pls_scores':scores(metric,x,4)}
        else:head={'train_features':x,'gamma':1.,'alpha':.1}
        kernel=kernel_for(head,head['train_features'],metric)
        kernel.flat[::len(y)+1]+=head['alpha']
        head.update(coefficients=solve(kernel,y-y.mean(),assume_a='pos',check_finite=False,overwrite_a=True),target_mean=float(y.mean()))
        np.savez_compressed(out/f'component_{i}_head.npz',**head)
        del kernel,x,head
        print('Refitted component',i,flush=True)
    write(out/'recipe.json',recipe)


def predict(args):
    import joblib
    from visual_kernel_components import kernel_for
    from pca_kernel_component import apply_transform
    folder=args.output/'fitted';recipe=read(folder/'recipe.json');parts=[]
    target=args.output/'predictions.csv'
    if target.exists():raise FileExistsError(target)
    for i in range(3):
        x=features(args,i,'test')
        if i==2:
            with np.load(folder/f'component_{i}_pca.npz',allow_pickle=False) as z:x=apply_transform(x,{k:z[k] for k in z.files})
        metric=joblib.load(folder/f'component_{i}_metric.joblib') if i==1 else None
        with np.load(folder/f'component_{i}_head.npz',allow_pickle=False) as z:head={k:z[k] for k in z.files}
        result=np.concatenate([(kernel_for(head,x[j:j+128],metric)@head['coefficients']+float(head['target_mean']))*1000 for j in range(0,len(x),128)])
        parts.append(np.maximum(result,0))
    p=np.column_stack(parts)@np.asarray(recipe['weights'])
    ids=pd.read_csv(ROOT/'inputs/test.csv').imageid
    pd.DataFrame({'imageid':ids,'price':p}).to_csv(target,index=False)
    original=pd.read_csv(ROOT/'predictions.csv').price.to_numpy()
    write(args.output/'comparison.json',{'n':len(p),'max_abs_difference_from_archived_submission':float(np.max(abs(p-original))),
        'note':'Fresh encoder computation can differ with hardware/library precision; inspect the difference, not CSV byte identity.'})
    print(target)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['check','extract','fit','predict'])
    parser.add_argument('--component',type=int,choices=[0,1,2])
    parser.add_argument('--partition',choices=['train','test'])
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'.local/reproduction')
    args=parser.parse_args()
    if args.phase=='extract' and (args.component is None or args.partition is None):parser.error('extract requires --component and --partition')
    args.output=args.output.resolve()
    with threadpool_limits(limits=2):
        if args.phase=='check':check()
        elif args.phase=='extract':extract(args)
        elif args.phase=='fit':fit(args)
        else:predict(args)

if __name__=='__main__':main()
