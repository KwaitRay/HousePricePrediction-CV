"""StreetCLIP image tower + visual projection only; no text or geolocation lookup."""
import argparse,json,time,uuid,hashlib,os,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd,torch,joblib
from torch import nn
from PIL import Image,ImageOps
from transformers import CLIPVisionModel,CLIPVisionConfig,CLIPImageProcessor
from safetensors.torch import load_file
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV,GroupKFold
from threadpoolctl import threadpool_limits
from common.local_paths import local_path,REPOSITORY
from common.config import digest,write_json
from common.metrics import save_predictions

def main():
    p=argparse.ArgumentParser();p.add_argument('--weights-dir',type=Path,required=True);args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(2026)
    provenance=json.loads((args.weights_dir/'download.json').read_text());weight=args.weights_dir/provenance['weight_file']
    assert provenance['repo']=='geolocal/StreetCLIP' and digest(weight)==provenance['files'][weight.name]['sha256'] and provenance['files'][weight.name]['publisher_lfs_verified']
    config=json.loads((args.weights_dir/'config.json').read_text());vision=CLIPVisionModel(CLIPVisionConfig(**config['vision_config']))
    state=load_file(str(weight)) if weight.suffix=='.safetensors' else torch.load(weight,map_location='cpu',weights_only=True)
    tensors={k:v for k,v in state.items() if k.startswith('vision_model.')}
    dropped=[]
    if 'vision_model.embeddings.position_ids' in tensors and 'vision_model.embeddings.position_ids' not in vision.state_dict():
        position=tensors.pop('vision_model.embeddings.position_ids');assert torch.equal(position,vision.vision_model.embeddings.position_ids);dropped.append('legacy fixed position_ids buffer')
    vision.load_state_dict(tensors,strict=True)
    projection=nn.Linear(config['vision_config']['hidden_size'],config['projection_dim'],bias=False)
    projection.load_state_dict({'weight':state['visual_projection.weight']},strict=True);del state,tensors
    vision=vision.float().cuda().eval().requires_grad_(False);projection=projection.float().cuda().eval().requires_grad_(False)
    processor=CLIPImageProcessor.from_pretrained(args.weights_dir,local_files_only=True)
    split=local_path('split_path');frame=pd.read_csv(split,dtype={'imageid':str,'group_id':str});train=frame.partition.eq('train').to_numpy();valid=frame.partition.eq('validation').to_numpy()
    assert len(frame)==8000 and train.sum()==6399 and valid.sum()==1601 and not set(frame.loc[train,'group_id'])&set(frame.loc[valid,'group_id'])
    manifest={'model':provenance['repo'],'revision':provenance['revision'],'license':provenance['license'],'checkpoint_sha256':digest(weight),
        'imageids':frame.imageid.tolist(),'pixel_hashes':frame.pixel_hash.tolist(),'split_sha256':digest(split),'source_sha256':digest(__file__),'preprocess':processor.to_dict(),
        'feature':'L2-normalized projected visual CLS embedding','dimension':config['projection_dim'],'frozen':True,'precision':'float32',
        'text_modules_used':False,'test_images_used':False,'legacy_buffers_verified':dropped,
        'course_scope':'CLIP vision-only geo-pretrained reference, not house-price task; overall course eligibility pending confirmation'}
    key=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest();cache=local_path('cache_root')/'streetclip_features'/key;cache.mkdir(parents=True,exist_ok=True);write_json(cache/'manifest.json',manifest)
    run=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
    out=local_path('records_root')/'17_geo_visual_reference/streetclip'/run;out.mkdir(parents=True)
    public=REPOSITORY/'output/experiment_record/17_geo_visual_reference/streetclip'/run;public.mkdir(parents=True)
    write_json(out/'data_manifest.json',manifest);write_json(public/'registration.json',{**{k:v for k,v in manifest.items() if k not in ['imageids','pixel_hashes']},'cache_key':key,
        'alphas':[1,10,100,1000,10000],'selection':'3-fold training GroupKFold, fold-specific StandardScaler; original unweighted MSE'})
    def status(s,**kw):write_json(public/'status.json',{'status':s,'pid':os.getpid(),**kw})
    status('extracting');chunks=[];start=time.perf_counter()
    try:
        for first in range(0,8000,4):
            rows=frame.iloc[first:first+4];path=cache/f'{first:05d}_{len(rows)}.npy'
            if path.exists():features=np.load(path)
            else:
                images=[]
                for name in rows.imageid:
                    with Image.open(local_path('data_root')/'train'/name) as im:images.append(ImageOps.exif_transpose(im).convert('RGB'))
                pixels=processor(images=images,return_tensors='pt')['pixel_values'].cuda()
                with torch.inference_mode():features=projection(vision(pixel_values=pixels).pooler_output).float();features=features/features.norm(dim=-1,keepdim=True).clamp_min(1e-12);features=features.cpu().numpy()
                temporary=path.with_suffix('.tmp')
                with temporary.open('wb') as stream:np.save(stream,features)
                temporary.replace(path)
            assert features.shape==(len(rows),config['projection_dim']) and np.isfinite(features).all();chunks.append(features)
            if first%400==0:status('extracting',images=first+len(rows),total=8000);print(f'StreetCLIP features {first+len(rows)}/8000',flush=True)
        x=np.concatenate(chunks).astype(np.float64);del vision,projection;torch.cuda.empty_cache();status('fitting',head='ridge')
        search=GridSearchCV(make_pipeline(StandardScaler(),Ridge(solver='cholesky')),{'ridge__alpha':[1,10,100,1000,10000]},cv=GroupKFold(3),scoring='neg_mean_squared_error',n_jobs=1,error_score='raise')
        with threadpool_limits(limits=2):search.fit(x[train],frame.loc[train,'price']/1000,groups=frame.loc[train,'group_id'])
        pd.DataFrame(search.cv_results_).to_csv(public/'training_cv.csv',index=False);joblib.dump(search.best_estimator_,out/'head.joblib')
        metrics=save_predictions(out,frame.loc[valid],search.predict(x[valid])*1000,1300)
        result={'head':'ridge','alpha':search.best_params_['ridge__alpha'],'training_cv_mse':-search.best_score_*1e6,**metrics,'seconds':time.perf_counter()-start}
        write_json(public/'results.json',result);print(json.dumps(result),flush=True);status('completed')
        write_json(public/'kernel_queue.json',{'status':'starting','cache_key':key,'gammas':[1,4,16],'alphas':[.01,.1,1,10]})
        followup=subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).with_name('frozen_kernel_search.py')),
            '--name','streetclip_kernel','--cache-family','streetclip_features','--cache-key',key],check=False)
        write_json(public/'kernel_queue.json',{'status':'completed' if followup.returncode==0 else 'failed','exit_code':followup.returncode,'cache_key':key})
        if followup.returncode:raise RuntimeError('StreetCLIP kernel followup failed; Ridge result preserved')
    except Exception as exc:status('failed',error=repr(exc));raise

if __name__=='__main__':main()
