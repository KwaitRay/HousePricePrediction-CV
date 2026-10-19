"""Independent frozen official CLIP image-tower variants with train-only Ridge CV."""
import argparse,json,hashlib,time,uuid,os,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd,torch,joblib,open_clip
from PIL import Image,ImageOps
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV,GroupKFold
from threadpoolctl import threadpool_limits
from common.local_paths import local_path,REPOSITORY
from common.config import digest,write_json
from common.metrics import save_predictions

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['ViT-B-16','ViT-L-14','ViT-L-14-336'],default='ViT-B-16');p.add_argument('--batch-size',type=int,default=8);p.add_argument('--kernel-followup',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(2026)
    split=local_path('split_path');frame=pd.read_csv(split,dtype={'imageid':str,'group_id':str})
    assert len(frame)==8000 and frame.imageid.is_unique
    train=frame.partition.eq('train').to_numpy();valid=frame.partition.eq('validation').to_numpy()
    assert train.sum()==6399 and valid.sum()==1601 and not set(frame.loc[train,'group_id'])&set(frame.loc[valid,'group_id'])
    checkpoint=local_path('cache_root')/'clip_weights'/f'{args.model}.pt';expected=open_clip.get_pretrained_cfg(args.model,'openai')
    assert checkpoint.exists() and digest(checkpoint)==expected['url'].split('/')[-2]
    clip,_,preprocess=open_clip.create_model_and_transforms(args.model,pretrained=str(checkpoint),force_quick_gelu=True,weights_only=False,
        image_mean=expected['mean'],image_std=expected['std'],image_interpolation=expected['interpolation'],image_resize_mode=expected['resize_mode'])
    model=clip.visual.float().cuda().eval().requires_grad_(False);del clip
    name=args.model.replace('-','_');run=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
    out=local_path('records_root')/'13_clip_optimization'/name/run;out.mkdir(parents=True)
    public=REPOSITORY/'output/experiment_record/13_clip_optimization'/name/run;public.mkdir(parents=True)
    manifest={'model':args.model,'pretrained':'openai','checkpoint_sha256':digest(checkpoint),'split_sha256':digest(split),
        'imageids':frame.imageid.tolist(),'pixel_hashes':frame.pixel_hash.tolist(),'preprocess':repr(preprocess),'precision':'float32',
        'source_sha256':digest(__file__),'course_scope':'CLIP exploratory, eligibility unresolved','seed':2026,'frozen':True,'test_images_used':False}
    key=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest();cache=local_path('cache_root')/'clip_features'/key;cache.mkdir(parents=True,exist_ok=True)
    write_json(cache/'manifest.json',manifest);write_json(out/'data_manifest.json',manifest)
    write_json(public/'registration.json',{**{k:v for k,v in manifest.items() if k not in ['imageids','pixel_hashes']},'feature_cache_key':key,'alphas':[1,10,100,1000,10000],'selection':'3-fold train-only GroupKFold'})
    def status(state,**kw):write_json(public/'status.json',{'status':state,'pid':os.getpid(),**kw})
    status('extracting');chunks=[];start=time.perf_counter()
    try:
        for offset in range(0,8000,args.batch_size):
            rows=frame.iloc[offset:offset+args.batch_size];path=cache/f'{offset:05d}_{len(rows)}.npy'
            if path.exists():values=np.load(path)
            else:
                images=[]
                for imageid in rows.imageid:
                    with Image.open(local_path('data_root')/'train'/imageid) as im:images.append(preprocess(ImageOps.exif_transpose(im).convert('RGB')))
                with torch.inference_mode():values=model(torch.stack(images).cuda()).float();values=values/values.norm(dim=-1,keepdim=True).clamp_min(1e-12);values=values.cpu().numpy()
                temporary=path.with_suffix('.tmp')
                with temporary.open('wb') as stream:np.save(stream,values)
                temporary.replace(path)
            assert len(values)==len(rows) and values.ndim==2 and np.isfinite(values).all();chunks.append(values)
            if offset%400==0:status('extracting',images=offset+len(rows),total=8000,cache_key=key);print(f'{args.model} features {offset+len(rows)}/8000',flush=True)
        x=np.concatenate(chunks).astype(np.float64);del model;torch.cuda.empty_cache();status('fitting',head='ridge')
        search=GridSearchCV(make_pipeline(StandardScaler(),Ridge(solver='cholesky')),{'ridge__alpha':[1,10,100,1000,10000]},cv=GroupKFold(3),scoring='neg_mean_squared_error',error_score='raise',n_jobs=1)
        with threadpool_limits(limits=2):search.fit(x[train],frame.loc[train,'price']/1000,groups=frame.loc[train,'group_id'])
        pd.DataFrame(search.cv_results_).to_csv(public/'training_cv.csv',index=False);joblib.dump(search.best_estimator_,out/'head.joblib')
        metrics=save_predictions(out,frame.loc[valid],search.predict(x[valid])*1000,1300)
        result={'head':'ridge','alpha':search.best_params_['ridge__alpha'],'training_cv_mse':-search.best_score_*1e6,**metrics,'seconds':time.perf_counter()-start}
        write_json(public/'results.json',result);print(json.dumps(result),flush=True);status('completed')
        if args.kernel_followup:
            write_json(public/'kernel_queue.json',{'status':'starting','cache_key':key,'gammas':[1,4,16],'alphas':[.01,.1,1,10]})
            followup=subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).with_name('frozen_kernel_search.py')),
                '--name',name+'_kernel','--cache-family','clip_features','--cache-key',key],check=False)
            write_json(public/'kernel_queue.json',{'status':'completed' if followup.returncode==0 else 'failed','exit_code':followup.returncode,'cache_key':key})
            if followup.returncode:raise RuntimeError('Kernel followup failed; Ridge result preserved')
    except Exception as exc:status('failed',error=repr(exc));raise

if __name__=='__main__':main()
