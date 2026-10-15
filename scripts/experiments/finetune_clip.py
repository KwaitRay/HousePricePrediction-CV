"""Warm-start CLIP price regression from the train-only Ridge reference."""
import os,time,uuid,json,random,argparse
import numpy as np,pandas as pd,joblib,torch,open_clip
from torch import nn
from torch.utils.data import DataLoader
from finetune_imagenet import Images
from common.local_paths import local_path,REPOSITORY
from common.config import digest,write_json
from common.metrics import save_predictions

class PriceModel(nn.Module):
    def __init__(self,visual,head):super().__init__();self.visual=visual;self.head=head
    def forward(self,x):
        features=self.visual(x).float();features=features/features.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        return self.head(features).reshape(-1)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['ViT-B-32','ViT-L-14'],default='ViT-B-32')
    parser.add_argument('--reference-dir');parser.add_argument('--batch-size',type=int,default=16);parser.add_argument('--epochs',type=int,default=20);args=parser.parse_args()
    torch.set_num_threads(2);torch.manual_seed(2026);np.random.seed(2026);random.seed(2026)
    split=local_path('split_path');frame=pd.read_csv(split,dtype={'imageid':str,'group_id':str})
    train=frame[frame.partition=='train'];val=frame[frame.partition=='validation']
    assert len(train)==6399 and len(val)==1601 and not set(train.group_id)&set(val.group_id)
    ref=local_path('records_root')/args.reference_dir if args.reference_dir else local_path('records_root')/'03_clip_reference/runs/20261013T043141Z_c9f899f2'
    manifest=json.loads((ref/'data_manifest.json').read_text());assert digest(split)==manifest['split_sha256']
    assert (manifest.get('model') or manifest['protocol']['model'])==args.model
    original=joblib.load(ref/'head.joblib');scaler=original.named_steps['standardscaler'];ridge=original.named_steps['ridge']
    weight=ridge.coef_/scaler.scale_;bias=float(ridge.intercept_-np.dot(weight,scaler.mean_))
    dimension=len(weight);head=nn.Linear(dimension,1)
    with torch.no_grad():head.weight.copy_(torch.tensor(weight,dtype=torch.float32).reshape(1,-1));head.bias.fill_(bias)
    import hashlib
    key=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest();cache=local_path('cache_root')/'clip_features'/key
    features=np.concatenate([np.load(p) for p in sorted(cache.glob('*.npy'))]);assert features.shape==(8000,dimension)
    with torch.inference_mode():converted=head(torch.from_numpy(features)).reshape(-1).numpy()
    expected_prices=original.predict(features)
    error=float(np.max(np.abs(converted-expected_prices))*1000);assert error<0.1,(error,'Ridge conversion mismatch')
    expected=open_clip.get_pretrained_cfg(args.model,'openai');checkpoint=local_path('cache_root')/'clip_weights'/f'{args.model}.pt'
    assert digest(checkpoint)==expected['url'].split('/')[-2]
    clip,_,preprocess=open_clip.create_model_and_transforms(args.model,pretrained=str(checkpoint),force_quick_gelu=True,weights_only=False,
        image_mean=expected['mean'],image_std=expected['std'],image_interpolation=expected['interpolation'],image_resize_mode=expected['resize_mode'])
    model=PriceModel(clip.visual.float(),head).cuda();del clip
    for param in model.visual.parameters():param.requires_grad_(False)
    last=model.visual.transformer.resblocks[-2:]
    optimizer=torch.optim.AdamW([{'params':model.head.parameters(),'lr':1e-4},{'params':last.parameters(),'lr':1e-6}],weight_decay=0.01)
    amp=torch.amp.GradScaler('cuda')
    loader=DataLoader(Images(train,preprocess,True),batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(2026),num_workers=2,persistent_workers=True,pin_memory=True)
    validation=DataLoader(Images(val,preprocess),batch_size=args.batch_size,shuffle=False,num_workers=2,persistent_workers=True,pin_memory=True)
    run=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
    label='clip_finetune' if args.model=='ViT-B-32' else args.model.replace('-','_')+'_finetune'
    out=local_path('records_root')/'13_clip_optimization'/label/run;out.mkdir(parents=True)
    public=REPOSITORY/'output/experiment_record/13_clip_optimization'/label/run;public.mkdir(parents=True)
    config={'model':args.model+' image tower only','seed':2026,'price_head':f'Linear equivalent of train-only StandardScaler+Ridge alpha{ridge.alpha}',
        'converted_max_prediction_difference_thousand_usd':error,'head_lr':1e-4,'backbone_lr':1e-6,'warmup_epochs':1,
        'fine_tune':'last 2 visual transformer blocks only; all earlier layers, projection and norms frozen','epochs':args.epochs,'min_epochs':5,'patience':4,
        'batch_size':args.batch_size,'augmentation':'horizontal flip p=0.5','loss':'unweighted price/1000 MSE','amp':'float16 training; float32 validation',
        'checkpoint_sha256':digest(checkpoint),'split_sha256':digest(split),'source_sha256':digest(__file__),'head_sha256':digest(ref/'head.joblib'),
        'course_scope':'Exploratory CLIP; eligibility unresolved','test_images_used':False,'selection':'lowest original-price full validation MSE, including epoch0'}
    write_json(public/'registration.json',config);write_json(out/'registration.json',config)
    def status(state,**kw):write_json(public/'status.json',{'status':state,'pid':os.getpid(),**kw})
    def predict():
        model.eval();values=[]
        with torch.inference_mode():
            for x,_ in validation:values.extend(model(x.cuda(non_blocking=True)).cpu().numpy()*1000)
        assert len(values)==1601 and np.isfinite(values).all();return np.asarray(values)
    status('validating_initial');start=time.perf_counter();history=[]
    try:
        raw=predict();initial=save_predictions(out,val,raw,1300);best=initial['mse'];best_epoch=0
        reference=json.loads((ref/'metrics.json').read_text())['mse'];assert abs(best-reference)/reference<0.001,'Initial model deviates from Ridge reference'
        torch.save(model.state_dict(),out/'best.pt');write_json(public/'initial_metrics.json',initial)
        print(f'CLIP initial MSE={best:.2f}; head conversion max error={error:.6f}',flush=True)
        for epoch in range(1,args.epochs+1):
            if epoch==2:
                for param in last.parameters():param.requires_grad_(True)
            model.eval();model.head.train()
            if epoch>=2:last.train()
            total=0.;count=0;epoch_start=time.perf_counter();status('training',epoch=epoch,best_mse=best,best_epoch=best_epoch)
            for x,y in loader:
                x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):loss=((model(x).float()-y)**2).mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                amp.scale(loss).backward();amp.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),1);amp.step(optimizer);amp.update()
                total+=loss.item()*len(y);count+=len(y)
            assert count==6399;raw=predict();mse=float(np.mean((np.maximum(raw,0)-val.price.to_numpy())**2))
            row={'epoch':epoch,'train_mse':total/count*1e6,'validation_mse':mse,'seconds':time.perf_counter()-epoch_start};history.append(row)
            pd.DataFrame(history).to_csv(public/'history.csv',index=False)
            if mse<best:
                best=mse;best_epoch=epoch;torch.save(model.state_dict(),out/'best.pt');save_predictions(out,val,raw,1300)
            status('training',epoch=epoch,best_mse=best,best_epoch=best_epoch);print(json.dumps(row),flush=True)
            if epoch>=5 and epoch-best_epoch>=4:break
        write_json(public/'results.json',{'best_mse':best,'best_epoch':best_epoch,'initial_mse':initial['mse'],'epochs':epoch,'seconds':time.perf_counter()-start})
        status('completed',best_mse=best,best_epoch=best_epoch)
    except Exception as exc:status('failed',error=repr(exc));raise

if __name__=='__main__':main()
