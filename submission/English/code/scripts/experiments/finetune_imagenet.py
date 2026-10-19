"""ImageNet ConvNeXt last-stage fine-tuning for house-price regression."""
import argparse,json,os,time,uuid,random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision.models import convnext_tiny,ConvNeXt_Tiny_Weights
from PIL import Image,ImageOps
from common.local_paths import local_path,REPOSITORY
from common.config import digest,write_json
from common.metrics import save_predictions

class Images(Dataset):
    def __init__(self,frame,transform,augment=False):self.frame=frame.reset_index(drop=True);self.transform=transform;self.augment=augment
    def __len__(self):return len(self.frame)
    def __getitem__(self,i):
        row=self.frame.iloc[i]
        with Image.open(local_path('data_root')/'train'/row.imageid) as im:
            im=ImageOps.exif_transpose(im).convert('RGB')
            if self.augment and torch.rand(())<0.5:im=ImageOps.mirror(im)
            image=self.transform(im)
        return image,torch.tensor(row.price/1000,dtype=torch.float32)

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=2026);p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for this experiment')
    split=local_path('split_path');frame=pd.read_csv(split,dtype={'imageid':str,'group_id':str})
    train=frame[frame.partition=='train'];val=frame[frame.partition=='validation']
    assert len(train)==6399 and len(val)==1601 and not set(train.group_id)&set(val.group_id)
    torch.hub.set_dir(str(local_path('cache_root')/'torch_hub'))
    weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    checkpoint=local_path('cache_root')/'torch_hub/checkpoints'/Path(weights.url).name
    assert digest(checkpoint).startswith('983f1562')
    model=convnext_tiny(weights=weights).cuda()
    model.classifier[2]=nn.Sequential(nn.Linear(768,128),nn.GELU(),nn.Dropout(0.2),nn.Linear(128,1)).cuda()
    with torch.no_grad():model.classifier[2][-1].bias.fill_(float(train.price.mean()/1000))
    for param in model.parameters():param.requires_grad_(False)
    for param in model.classifier.parameters():param.requires_grad_(True)
    optimizer=torch.optim.AdamW([{'params':model.classifier.parameters(),'lr':1e-3},{'params':model.features[-1].parameters(),'lr':1e-5}],weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda')
    generator=torch.Generator().manual_seed(args.seed)
    kw={'batch_size':args.batch_size,'num_workers':args.workers,'pin_memory':True,'persistent_workers':args.workers>0}
    loader=DataLoader(Images(train,weights.transforms(),True),shuffle=True,generator=generator,**kw)
    validation=DataLoader(Images(val,weights.transforms()),shuffle=False,**kw)
    run=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
    out=local_path('records_root')/'12_finetune/convnext_tiny'/run;out.mkdir(parents=True)
    public=REPOSITORY/'output/experiment_record/12_finetune/convnext_tiny'/run;public.mkdir(parents=True)
    config={'seed':args.seed,'epochs':args.epochs,'batch_size':args.batch_size,'workers':args.workers,'pretrained':'ImageNet-1K ConvNeXt-Tiny',
        'checkpoint_sha256':digest(checkpoint),'split_sha256':digest(split),'source_sha256':digest(__file__),'preprocess':repr(weights.transforms()),
        'train_only':6399,'validation':1601,'test_images_used':False,'loss':'unweighted MSE on price/1000',
        'augmentation':'horizontal flip p=0.5 only','warmup_epochs':3,'fine_tune':'last features stage and classifier; earlier stages frozen and eval',
        'head_lr':1e-3,'backbone_lr':1e-5,'weight_decay':1e-4,'amp':'float16','gradient_clip':5,
        'early_stop_min_epochs':10,'patience':8,'selection':'strict lowest original-price unweighted validation MSE',
        'scope':'Full-method comparison; architecture, head and transfer learning differ from scratch baseline; not a single-factor scratch ablation'}
    write_json(public/'registration.json',config);write_json(out/'registration.json',config)
    history=[];best=float('inf');best_epoch=0;start=time.perf_counter()
    def status(state,**kw):
        record={'status':state,'pid':os.getpid(),**kw};write_json(public/'status.json',record);write_json(out/'status.json',record)
    status('running')
    try:
        for epoch in range(1,args.epochs+1):
            if epoch==4:
                for param in model.features[-1].parameters():param.requires_grad_(True)
            model.eval();model.classifier.train()
            if epoch>=4:model.features[-1].train()
            total=0.;count=0;epoch_start=time.perf_counter()
            for images,y in loader:
                images=images.cuda(non_blocking=True);y=y.cuda(non_blocking=True);optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):pred=model(images).reshape(-1);loss=((pred.float()-y)**2).mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(optimizer);scaler.update()
                total+=loss.item()*len(y);count+=len(y)
            assert count==6399
            model.eval();values=[]
            with torch.inference_mode():
                for images,_ in validation:
                    with torch.autocast('cuda',dtype=torch.float16):prediction=model(images.cuda(non_blocking=True)).reshape(-1)
                    values.extend(prediction.float().cpu().numpy()*1000)
            raw=np.asarray(values);pred=np.maximum(raw,0);mse=float(np.mean((pred-val.price.to_numpy())**2))
            if not np.isfinite(mse) or len(pred)!=1601:raise ValueError('Invalid validation predictions')
            row={'epoch':epoch,'train_mse':total/count*1e6,'validation_mse':mse,'seconds':time.perf_counter()-epoch_start};history.append(row)
            pd.DataFrame(history).to_csv(public/'history.csv',index=False)
            if mse<best:
                best=mse;best_epoch=epoch
                torch.save(model.state_dict(),out/'best.pt');save_predictions(out,val,raw,1300)
            status('running',epoch=epoch,best_epoch=best_epoch,best_mse=best)
            print(json.dumps(row),flush=True)
            if epoch>=10 and epoch-best_epoch>=8:break
        write_json(public/'results.json',{'best_mse':best,'best_epoch':best_epoch,'epochs':epoch,'seconds':time.perf_counter()-start})
        status('completed',best_mse=best,best_epoch=best_epoch)
    except Exception as exc:status('failed',error=repr(exc));raise

if __name__=='__main__':main()
