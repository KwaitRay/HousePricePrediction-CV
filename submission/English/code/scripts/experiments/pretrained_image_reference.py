"""ImageNet-only frozen CNN features; training-group CV selects regression heads."""
import argparse, hashlib, json, os, time, uuid
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision import models
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits
from common.local_paths import REPOSITORY, local_path
from common.config import digest, write_json
from common.metrics import save_predictions

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['resnet50','convnext_tiny','efficientnet_b0'],default='resnet50')
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(2026)
    out=local_path('records_root')/'11_pretrained_cnn'/args.model/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True)
    public=REPOSITORY/'output/experiment_record/11_pretrained_cnn'/args.model/out.name;public.mkdir(parents=True)
    def status(value,**kw):
        record={'status':value,'pid':os.getpid(),'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw}
        write_json(out/'status.json',record);write_json(public/'status.json',record)
    start=time.perf_counter();status('starting')
    try:
        torch.hub.set_dir(str(local_path('cache_root')/'torch_hub'))
        enums={'resnet50':models.ResNet50_Weights.IMAGENET1K_V2,'convnext_tiny':models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,'efficientnet_b0':models.EfficientNet_B0_Weights.IMAGENET1K_V1}
        weights=enums[args.model]
        checkpoint=local_path('cache_root')/'torch_hub/checkpoints'/Path(weights.url).name
        if not checkpoint.exists():raise ValueError(f'Download official checkpoint first: {weights.url} -> {checkpoint}')
        if not digest(checkpoint).startswith(checkpoint.stem.split('-')[-1]):raise ValueError('Official checkpoint hash prefix mismatch')
        model=getattr(models,args.model)(weights=weights).eval().requires_grad_(False)
        if args.model=='resnet50':model.fc=torch.nn.Identity()
        elif args.model=='convnext_tiny':model.classifier[2]=torch.nn.Identity()
        else:model.classifier=torch.nn.Identity()
        device='cuda' if torch.cuda.is_available() else 'cpu';model=model.to(device)
        preprocess=weights.transforms()
        split=local_path('split_path');frame=pd.read_csv(split,dtype={'imageid':str,'group_id':str})
        assert frame.imageid.is_unique and len(frame)==8000
        train=frame.partition.eq('train').to_numpy();valid=frame.partition.eq('validation').to_numpy()
        assert train.sum()==6399 and valid.sum()==1601
        assert not set(frame.loc[train,'group_id'])&set(frame.loc[valid,'group_id'])
        registration={'model':args.model,'weights':str(weights),'checkpoint_sha256':digest(checkpoint),'checkpoint_url':weights.url,
          'pretraining':'ImageNet-1K image classification, not house price task or SoCal dataset; no MLLM',
          'split_sha256':digest(split),'preprocess':repr(preprocess),'imageids':frame.imageid.tolist(),
          'pixel_hashes':frame.pixel_hash.tolist(),'source_sha256':digest(__file__),'seed':2026,
          'scope':'Only development training/validation images; parameters selected with 3-fold training GroupKFold; no test images',
          'course_evidence':'COMP90086_2026_Project.pdf page 2 explicitly permits pretrained features; excludes task/dataset-pretrained models, extra SoCal data, pretrained MLLMs',
          'heads':{'ridge_alpha':[0.1,1,10,100,1000,10000], 'svr_C':[1,10,100], 'svr_gamma':[1,4,16], 'svr_epsilon':0.05},
          'precision':'float32','torch':torch.__version__}
        write_json(out/'registration.json',registration)
        write_json(public/'registration.json',{k:v for k,v in registration.items() if k not in ['imageids','pixel_hashes']})
        # Cache includes complete input ordering, split and preprocessing provenance.
        key=hashlib.sha256(json.dumps(registration,sort_keys=True).encode()).hexdigest()
        cache=local_path('cache_root')/'imagenet_features'/key;cache.mkdir(parents=True,exist_ok=True)
        write_json(cache/'manifest.json',registration)
        status('extracting',cache_key=key)
        features=[]
        for first in range(0,len(frame),args.batch_size):
            subset=frame.iloc[first:first+args.batch_size];path=cache/f'{first:05d}_{len(subset)}.npy'
            if path.exists():batch=np.load(path)
            else:
                tensors=[]
                for name in subset.imageid:
                    with Image.open(local_path('data_root')/'train'/name) as im:tensors.append(preprocess(ImageOps.exif_transpose(im).convert('RGB')))
                with torch.inference_mode():batch=model(torch.stack(tensors).to(device)).float().cpu().numpy()
                temporary=path.with_suffix('.tmp')
                with temporary.open('wb') as stream:np.save(stream,batch)
                temporary.replace(path)
            if batch.ndim!=2 or len(batch)!=len(subset) or not np.isfinite(batch).all():raise ValueError('Invalid cached features')
            features.append(batch)
            if first%(args.batch_size*25)==0:
                status('extracting',completed_images=first+len(subset),total_images=len(frame),cache_key=key)
                print(f'{args.model}: extracted {first+len(subset)}/{len(frame)}',flush=True)
        x=np.concatenate(features);y=frame.price.to_numpy()/1000
        del model
        if device=='cuda':torch.cuda.empty_cache()
        candidates=[('ridge',make_pipeline(StandardScaler(),Ridge(solver='cholesky')),{'ridge__alpha':registration['heads']['ridge_alpha']}),
                    ('svr',make_pipeline(Normalizer(),SVR(epsilon=0.05,cache_size=512)),{'svr__C':registration['heads']['svr_C'],'svr__gamma':registration['heads']['svr_gamma']})]
        scores=[]
        with threadpool_limits(limits=4):
            for name,head,grid in candidates:
                status('fitting',head=name,feature_dim=x.shape[1]);print(f'{args.model}: fitting {name} using training-only CV',flush=True)
                search=GridSearchCV(head,grid,cv=GroupKFold(3),scoring='neg_mean_squared_error',n_jobs=1,return_train_score=True,error_score='raise')
                search.fit(x[train],y[train],groups=frame.loc[train,'group_id'])
                pd.DataFrame(search.cv_results_).to_csv(public/f'{name}_training_cv.csv',index=False)
                joblib.dump(search.best_estimator_,out/f'{name}_head.joblib')
                destination=out/name;destination.mkdir()
                metrics=save_predictions(destination,frame.loc[valid],search.predict(x[valid])*1000,1300)
                write_json(public/f'{name}_metrics.json',metrics)
                result={'head':name,'selected_params':search.best_params_,'training_cv_mse':float(-search.best_score_*1e6),**metrics}
                actual=frame.loc[valid,'price'].to_numpy();pred=np.maximum(search.predict(x[valid])*1000,0)
                slices={}
                for label,mask in [('low',actual<=300),('middle',(actual>300)&(actual<=1300)),('high',actual>1300)]:
                    error=pred[mask]-actual[mask]
                    slices[label]={'n':int(mask.sum()),'mse':float(np.mean(error**2)) if mask.any() else None,
                        'mae':float(np.mean(np.abs(error))) if mask.any() else None,'bias':float(np.mean(error)) if mask.any() else None,
                        'mean_positive_overestimate':float(np.mean(np.maximum(error,0))) if mask.any() else None}
                write_json(public/f'{name}_price_slices.json',slices)
                scores.append(result);write_json(public/'results.json',scores)
                print(json.dumps(result),flush=True)
        write_json(out/'feature_cache.json',{'cache_key':key,'shape':list(x.shape)})
        write_json(public/'cost.json',{'seconds':time.perf_counter()-start,'device':device,'feature_shape':list(x.shape)})
        report='# ImageNet预训练CNN特征参照\n\n冻结图像编码器；使用官方确定性预处理。仅训练分区三折GroupKFold选择回归参数，原始未加权验证MSE评估，无测试图片、无价格/场景人工输入。课程PDF第2页允许预训练特征，本模型为ImageNet分类权重，不是房价任务模型或MLLM。\n\n| 方法 | 训练CV MSE | 验证MSE | MAE | 高价MSE |\n| --- | ---: | ---: | ---: | ---: |\n'
        for s in scores:report+=f"| {args.model}+{s['head']} | {s['training_cv_mse']:.2f} | {s['mse']:.2f} | {s['mae']:.2f} | {s['tail_mse']:.2f} |\n"
        report+='\n超参数与所有CV尝试保存在registration.json及training_cv.csv；完整回归头、逐图验证预测和图像特征仅留.local。验证不是独立测试，跨方案选择仍有选择偏差；本地<50000不能代替Kaggle公榜<50000。\n'
        (public/'实验报告.md').write_text(report,encoding='utf-8');status('completed')
    except Exception as exc:
        status('failed',error=repr(exc));raise

if __name__=='__main__':main()
