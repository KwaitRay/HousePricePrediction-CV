"""Evaluate a supervised fine-tuned image encoder with fixed, previously selected KRR."""
import argparse,json,hashlib,os,time,uuid
from pathlib import Path
import numpy as np,pandas as pd,torch,open_clip
from torch import nn
from PIL import Image,ImageOps
from scipy.linalg import solve
from threadpoolctl import threadpool_limits
from finetune_clip import PriceModel
from common.local_paths import local_path,REPOSITORY
from common.config import digest,write_json
from common.metrics import save_predictions

def main():
    p=argparse.ArgumentParser();p.add_argument('--fine-dir',required=True);args=p.parse_args();torch.set_num_threads(2)
    source=local_path('records_root')/args.fine_dir;config=json.loads((source/'registration.json').read_text())
    split=local_path('split_path');assert digest(split)==config['split_sha256'];frame=pd.read_csv(split,dtype={'imageid':str,'group_id':str})
    model_name=config['model'].split()[0];expected=open_clip.get_pretrained_cfg(model_name,'openai');checkpoint=local_path('cache_root')/'clip_weights'/f'{model_name}.pt'
    assert digest(checkpoint)==config['checkpoint_sha256']
    clip,_,preprocess=open_clip.create_model_and_transforms(model_name,pretrained=str(checkpoint),force_quick_gelu=True,weights_only=False,
        image_mean=expected['mean'],image_std=expected['std'],image_interpolation=expected['interpolation'],image_resize_mode=expected['resize_mode'])
    state=torch.load(source/'best.pt',map_location='cpu',weights_only=True);dimension=state['head.weight'].shape[1]
    model=PriceModel(clip.visual.float(),nn.Linear(dimension,1));model.load_state_dict(state,strict=True);del clip,state
    model=model.cuda().eval().requires_grad_(False)
    run=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
    out=local_path('records_root')/'13_clip_optimization/l14_finetuned_fixed_kernel'/run;out.mkdir(parents=True)
    public=REPOSITORY/'output/experiment_record/13_clip_optimization/l14_finetuned_fixed_kernel'/run;public.mkdir(parents=True)
    manifest={'imageids':frame.imageid.tolist(),'split_sha256':digest(split),'fine_source':args.fine_dir,'fine_checkpoint_sha256':digest(source/'best.pt'),
        'fine_registration_sha256':digest(source/'registration.json'),'supervised_encoder':True,'preprocess':repr(preprocess),'precision':'float32','source_sha256':digest(__file__)}
    key=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest();cache=local_path('cache_root')/'finetuned_clip_features'/key;cache.mkdir(parents=True,exist_ok=True);write_json(cache/'manifest.json',manifest)
    write_json(public/'registration.json',{**{k:v for k,v in manifest.items() if k!='imageids'},'cache_key':key,'gamma':4,'alpha':1,
        'head_selection':'Fixed previously training-CV selected pretrained L14 gamma4 alpha1; no new head CV on supervised encoder features',
        'caveat':'Encoder trained on6399 labels and checkpoint selected on original validation; final validation is development reuse, not independent test',
        'course_scope':'Exploratory CLIP visual model, eligibility unresolved','test_images_used':False})
    def status(s,**kw):write_json(public/'status.json',{'status':s,'pid':os.getpid(),**kw})
    status('extracting');start=time.perf_counter();chunks=[];linear_values=[]
    try:
        for first in range(0,8000,8):
            rows=frame.iloc[first:first+8];path=cache/f'{first:05d}_{len(rows)}.npy'
            if path.exists():features=np.load(path)
            else:
                images=[]
                for name in rows.imageid:
                    with Image.open(local_path('data_root')/'train'/name) as im:images.append(preprocess(ImageOps.exif_transpose(im).convert('RGB')))
                with torch.inference_mode():features=model.visual(torch.stack(images).cuda()).float();features=features/features.norm(dim=-1,keepdim=True).clamp_min(1e-12);features=features.cpu().numpy()
                temporary=path.with_suffix('.tmp')
                with temporary.open('wb') as stream:np.save(stream,features)
                temporary.replace(path)
            assert features.shape==(len(rows),dimension) and np.isfinite(features).all();chunks.append(features)
            with torch.inference_mode():linear_values.extend(model.head(torch.from_numpy(features).cuda()).reshape(-1).cpu().numpy()*1000)
            if first%400==0:status('extracting',images=first+len(rows),total=8000);print(f'Fine-tuned CLIP features {first+len(rows)}/8000',flush=True)
        x=np.concatenate(chunks).astype(np.float64);del model;torch.cuda.empty_cache()
        train=frame.partition.eq('train').to_numpy();valid=frame.partition.eq('validation').to_numpy();assert train.sum()==6399 and valid.sum()==1601
        validation_linear=np.maximum(np.asarray(linear_values)[valid],0);saved=json.loads((source/'metrics.json').read_text())['mse']
        reproduced=float(np.mean((validation_linear-frame.loc[valid,'price'].to_numpy())**2));assert abs(reproduced-saved)/saved<.001,'Fine checkpoint reproduction mismatch'
        write_json(public/'checkpoint_reproduction.json',{'reproduced_mse':reproduced,'saved_mse':saved})
        x/=np.linalg.norm(x,axis=1,keepdims=True);xt=x[train];xv=x[valid];y=frame.loc[train,'price'].to_numpy()/1000;mean=float(y.mean());status('fitting')
        with threadpool_limits(limits=4):
            k=np.exp(-4*np.maximum(2-2*xt@xt.T,0));coef=solve(k+np.eye(len(y)),y-mean,assume_a='pos',check_finite=False)
            raw=(np.exp(-4*np.maximum(2-2*xv@xt.T,0))@coef+mean)*1000
        metrics=save_predictions(out,frame.loc[valid],raw,1300);np.savez_compressed(out/'head.npz',train_features=xt,coefficients=coef,gamma=4,alpha=1,target_mean=mean)
        write_json(public/'results.json',{'head':'fixed_kernel_on_supervised_encoder',**metrics,'seconds':time.perf_counter()-start})
        print(json.dumps(metrics),flush=True);status('completed')
    except Exception as exc:status('failed',error=repr(exc));raise

if __name__=='__main__':main()
