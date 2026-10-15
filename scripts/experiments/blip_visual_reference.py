"""BLIP-base image tower only; no text encoder, decoder, captions or prompts."""
import argparse,json,time,uuid,hashlib,os
from pathlib import Path
import numpy as np,pandas as pd,torch,joblib
from PIL import Image,ImageOps
from transformers import BlipVisionModel,BlipVisionConfig,BlipImageProcessor
from safetensors.torch import load_file
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV,GroupKFold
from threadpoolctl import threadpool_limits
from common.local_paths import local_path,REPOSITORY
from common.config import digest,write_json
from common.metrics import save_predictions

def main():
    p=argparse.ArgumentParser();p.add_argument('--weights-dir',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(2026)
    provenance=json.loads((args.weights_dir/'download.json').read_text());weight=args.weights_dir/provenance['weight_file']
    assert digest(weight)==provenance['files'][weight.name]['sha256'] and provenance['files'][weight.name]['publisher_lfs_verified']
    config=json.loads((args.weights_dir/'config.json').read_text());processor=BlipImageProcessor.from_pretrained(args.weights_dir,local_files_only=True)
    model=BlipVisionModel(BlipVisionConfig(**config['vision_config']))
    state=load_file(str(weight)) if weight.suffix=='.safetensors' else torch.load(weight,map_location='cpu',weights_only=True)
    vision={k[len('vision_model.'):]:v for k,v in state.items() if k.startswith('vision_model.')}
    model.load_state_dict(vision,strict=True);del state,vision
    model=model.float().cuda().eval().requires_grad_(False)
    split=local_path('split_path');frame=pd.read_csv(split,dtype={'imageid':str,'group_id':str})
    train=frame.partition.eq('train').to_numpy();valid=frame.partition.eq('validation').to_numpy()
    assert len(frame)==8000 and train.sum()==6399 and valid.sum()==1601 and not set(frame.loc[train,'group_id'])&set(frame.loc[valid,'group_id'])
    manifest={'model':provenance['repo'],'revision':provenance['revision'],'checkpoint_sha256':digest(weight),
        'imageids':frame.imageid.tolist(),'split_sha256':digest(split),'pixel_hashes':frame.pixel_hash.tolist(),'preprocess':processor.to_dict(),
        'feature':'L2-normalized CLS pooled image token','source_sha256':digest(__file__),'frozen':True,'precision':'float32',
        'text_modules_used':False,'test_images_used':False,'course_scope':'Original BLIP visual tower exploratory; course eligibility unresolved; no BLIP-2 or LLM'}
    key=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest();cache=local_path('cache_root')/'blip_features'/key;cache.mkdir(parents=True,exist_ok=True)
    write_json(cache/'manifest.json',manifest)
    run=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
    out=local_path('records_root')/'14_blip_reference/blip_base'/run;out.mkdir(parents=True)
    public=REPOSITORY/'output/experiment_record/14_blip_reference/blip_base'/run;public.mkdir(parents=True)
    write_json(out/'data_manifest.json',manifest);write_json(public/'registration.json',{**{k:v for k,v in manifest.items() if k not in ['imageids','pixel_hashes']},'cache_key':key,'alphas':[1,10,100,1000,10000],'selection':'train-only 3-fold GroupKFold'})
    def status(state,**kw):write_json(public/'status.json',{'status':state,'pid':os.getpid(),**kw})
    status('extracting');chunks=[];start=time.perf_counter()
    try:
        for offset in range(0,8000,4):
            rows=frame.iloc[offset:offset+4];path=cache/f'{offset:05d}_{len(rows)}.npy'
            if path.exists():values=np.load(path)
            else:
                images=[]
                for name in rows.imageid:
                    with Image.open(local_path('data_root')/'train'/name) as im:images.append(ImageOps.exif_transpose(im).convert('RGB'))
                pixels=processor(images=images,return_tensors='pt')['pixel_values'].cuda()
                with torch.inference_mode():values=model(pixel_values=pixels).pooler_output.float();values=values/values.norm(dim=-1,keepdim=True).clamp_min(1e-12);values=values.cpu().numpy()
                temporary=path.with_suffix('.tmp')
                with temporary.open('wb') as stream:np.save(stream,values)
                temporary.replace(path)
            assert values.shape==(len(rows),768) and np.isfinite(values).all();chunks.append(values)
            if offset%400==0:status('extracting',images=offset+len(rows),total=8000);print(f'BLIP visual features {offset+len(rows)}/8000',flush=True)
        x=np.concatenate(chunks).astype(np.float64);del model;torch.cuda.empty_cache();status('fitting',head='ridge')
        search=GridSearchCV(make_pipeline(StandardScaler(),Ridge(solver='cholesky')),{'ridge__alpha':[1,10,100,1000,10000]},cv=GroupKFold(3),scoring='neg_mean_squared_error',n_jobs=1,error_score='raise')
        with threadpool_limits(limits=2):search.fit(x[train],frame.loc[train,'price']/1000,groups=frame.loc[train,'group_id'])
        pd.DataFrame(search.cv_results_).to_csv(public/'training_cv.csv',index=False);joblib.dump(search.best_estimator_,out/'head.joblib')
        metrics=save_predictions(out,frame.loc[valid],search.predict(x[valid])*1000,1300)
        result={'head':'ridge','alpha':search.best_params_['ridge__alpha'],'training_cv_mse':-search.best_score_*1e6,**metrics,'seconds':time.perf_counter()-start}
        write_json(public/'results.json',result);print(json.dumps(result),flush=True);status('completed')
    except Exception as exc:status('failed',error=repr(exc));raise

if __name__=='__main__':main()
