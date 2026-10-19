"""Retain existing CLIP reference; add full-image letterbox and fixed center detail views."""
import json,hashlib,time,uuid,os,argparse,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch,joblib,open_clip
from PIL import Image,ImageOps
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV,GroupKFold
from threadpoolctl import threadpool_limits
from common.local_paths import REPOSITORY,local_path
from common.config import digest,write_json
from common.metrics import save_predictions

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['ViT-B-32','ViT-L-14','ViT-L-14-336'],default='ViT-B-32')
    parser.add_argument('--reference-manifest',type=Path);parser.add_argument('--batch-size',type=int,default=16);parser.add_argument('--kernel-followup',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);torch.manual_seed(2026)
    ref=args.reference_manifest or local_path('records_root')/'03_clip_reference/runs/20261013T043141Z_c9f899f2/data_manifest.json'
    manifest=json.loads(ref.read_text());split=local_path('split_path');assert digest(split)==manifest['split_sha256']
    assert (manifest.get('model') or manifest['protocol']['model'])==args.model
    frame=pd.read_csv(split,dtype={'imageid':str,'group_id':str});assert frame.imageid.tolist()==manifest['imageids']
    key=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    original=np.concatenate([np.load(p) for p in sorted((local_path('cache_root')/'clip_features'/key).glob('*.npy'))]);assert len(original)==8000 and original.ndim==2
    dimension=original.shape[1]
    expected=open_clip.get_pretrained_cfg(args.model,'openai');checkpoint=local_path('cache_root')/'clip_weights'/f'{args.model}.pt'
    assert digest(checkpoint)==expected['url'].split('/')[-2]
    model,_,preprocess=open_clip.create_model_and_transforms(args.model,pretrained=str(checkpoint),force_quick_gelu=True,weights_only=False,
        image_mean=expected['mean'],image_std=expected['std'],image_interpolation=expected['interpolation'],image_resize_mode=expected['resize_mode'])
    visual=model.visual.cuda().float().eval().requires_grad_(False);del model
    shape=visual.image_size if isinstance(visual.image_size,tuple) else (visual.image_size,visual.image_size)
    assert shape[0]==shape[1];view_size=int(shape[0])
    run=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
    label='clip_multiview' if args.model=='ViT-B-32' else args.model.replace('-','_')+'_multiview'
    out=local_path('records_root')/'13_clip_optimization'/label/run;out.mkdir(parents=True)
    public=REPOSITORY/'output/experiment_record/13_clip_optimization'/label/run;public.mkdir(parents=True)
    registration={'views':['original official center-crop reference',f'whole EXIF image letterboxed to {view_size} square, gray RGB128 border','fixed center 60% width and height, official preprocessing'],
        'view_size':view_size,
        'combinations':['original+whole','original+center','original+whole+center'],'head':'fold-specific StandardScaler+Ridge',
        'alphas':[10,100,1000,10000,100000],'selection':'training-only 3-fold GroupKFold','split_sha256':digest(split),
        'original_manifest_sha256':digest(ref),'checkpoint_sha256':digest(checkpoint),'source_sha256':digest(__file__),
        'imageids':frame.imageid.tolist(),'model':args.model,'feature_dim':int(dimension),'preprocess':repr(preprocess),'precision':'float32','test_images_used':False,
        'course_scope':'CLIP exploratory; eligibility unresolved; no text prompts'}
    write_json(out/'registration.json',registration);write_json(public/'registration.json',{k:v for k,v in registration.items() if k!='imageids'})
    ck=hashlib.sha256(json.dumps(registration,sort_keys=True).encode()).hexdigest();cache=local_path('cache_root')/'clip_multiview'/ck;cache.mkdir(parents=True,exist_ok=True)
    write_json(cache/'manifest.json',registration)
    def status(state,**kw):write_json(public/'status.json',{'status':state,'pid':os.getpid(),**kw})
    status('extracting');start=time.perf_counter()
    try:
        chunks=[];batch_size=args.batch_size
        for offset in range(0,len(frame),batch_size):
            rows=frame.iloc[offset:offset+batch_size];path=cache/f'{offset:05d}_{len(rows)}.npy'
            if path.exists():values=np.load(path)
            else:
                images=[]
                for name in rows.imageid:
                    with Image.open(local_path('data_root')/'train'/name) as im:
                        im=ImageOps.exif_transpose(im).convert('RGB');w,h=im.size
                        full=ImageOps.pad(im,(view_size,view_size),method=Image.Resampling.BICUBIC,color=(128,128,128))
                        center=im.crop((round(w*.2),round(h*.2),round(w*.8),round(h*.8)))
                        images.extend([preprocess(full),preprocess(center)])
                with torch.inference_mode():
                    values=visual(torch.stack(images).cuda()).float();values=values/values.norm(dim=-1,keepdim=True).clamp_min(1e-12)
                    values=values.cpu().numpy().reshape(len(rows),2,dimension)
                temporary=path.with_suffix('.tmp')
                with temporary.open('wb') as stream:np.save(stream,values)
                temporary.replace(path)
            assert values.shape==(len(rows),2,dimension) and np.isfinite(values).all();chunks.append(values)
            if offset%400==0:status('extracting',images=offset+len(rows),total=8000);print(f'CLIP multiview {offset+len(rows)}/8000',flush=True)
        extra=np.concatenate(chunks);del visual;torch.cuda.empty_cache()
        train=frame.partition.eq('train').to_numpy();valid=frame.partition.eq('validation').to_numpy();assert train.sum()==6399 and valid.sum()==1601
        groups=frame.loc[train,'group_id'];assert not set(groups)&set(frame.loc[valid,'group_id'])
        candidates={'original_whole':np.concatenate([original,extra[:,0]],axis=1),'original_center':np.concatenate([original,extra[:,1]],axis=1),
            'original_whole_center':np.concatenate([original,extra.reshape(8000,dimension*2)],axis=1)}
        results=[]
        with threadpool_limits(limits=2):
            for name,x in candidates.items():
                status('fitting',head=name);x=x.astype(np.float64)
                search=GridSearchCV(make_pipeline(StandardScaler(),Ridge(solver='cholesky')),{'ridge__alpha':registration['alphas']},cv=GroupKFold(3),scoring='neg_mean_squared_error',n_jobs=1,error_score='raise')
                search.fit(x[train],frame.loc[train,'price']/1000,groups=groups)
                pd.DataFrame(search.cv_results_).to_csv(public/f'{name}_training_cv.csv',index=False)
                folder=out/name;folder.mkdir();joblib.dump(search.best_estimator_,folder/'head.joblib')
                metrics=save_predictions(folder,frame.loc[valid],search.predict(x[valid])*1000,1300)
                row={'head':name,'alpha':search.best_params_['ridge__alpha'],'training_cv_mse':-search.best_score_*1e6,**metrics};results.append(row)
                write_json(public/'results.json',results);print(json.dumps(row),flush=True)
        write_json(public/'cost.json',{'seconds':time.perf_counter()-start,'feature_cache_key':ck});status('completed')
        if args.kernel_followup:
            write_json(public/'kernel_queue.json',{'status':'starting','cache_key':ck,'gammas':[1,4,16],'alphas':[.01,.1,1,10]})
            job=subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).with_name('frozen_kernel_search.py')),
                '--name',label+'_kernel','--cache-family','clip_multiview','--cache-key',ck,'--original-manifest',str(ref)],check=False)
            write_json(public/'kernel_queue.json',{'status':'completed' if job.returncode==0 else 'failed','exit_code':job.returncode,'cache_key':ck})
            if job.returncode:raise RuntimeError('Kernel followup failed; Ridge results preserved')
    except Exception as exc:status('failed',error=repr(exc));raise

if __name__=='__main__':main()
