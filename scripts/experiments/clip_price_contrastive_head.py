"""Controlled price contrastive auxiliary loss on frozen CLIP L14 image features."""
import argparse,json,time,uuid,os
import numpy as np,pandas as pd,torch,joblib
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from common.local_paths import local_path,REPOSITORY
from common.config import digest,write_json
from common.metrics import save_predictions

class Head(nn.Module):
    def __init__(self,dimension):
        super().__init__();self.hidden=nn.Sequential(nn.Linear(dimension,64),nn.GELU(),nn.Dropout(.25));self.price=nn.Linear(64,1);self.project=nn.Linear(64,32)
        nn.init.zeros_(self.price.weight);nn.init.zeros_(self.price.bias)
    def forward(self,x):
        h=self.hidden(x);z=self.project(h);z=z/z.norm(dim=1,keepdim=True).clamp_min(1e-12)
        return self.price(h).reshape(-1),z

def pair_loss(z,raw,price,groups,mode,epoch):
    delta=(torch.log(price[:,None]+1)-torch.log(price[None,:]+1)).abs()
    distinct=groups[:,None]!=groups[None,:]
    positive=(delta<=np.log(1.15))&distinct;negative=(delta>=np.log(2))&distinct
    visual=(2-2*raw@raw.T).clamp_min(0);distance2=(2-2*z@z.T).clamp_min(1e-8)
    hard_fraction=.25*min(max((epoch-10)/10,0),1) if mode=='curriculum' else 0
    losses=[];counts=[]
    for mask,loss in [(positive,distance2),(negative,torch.relu(.5-distance2.sqrt())**2)]:
        values=visual[mask]
        if not len(values):counts.append(0);continue
        low=torch.quantile(values,.25);high=torch.quantile(values,.75)
        easy=mask&(visual<=low if mask is positive else visual>=high)
        hard=mask&(visual>=high if mask is positive else visual<=low)
        term=loss[easy].mean() if easy.any() else loss[mask].mean()
        if hard_fraction and hard.any():term=(1-hard_fraction)*term+hard_fraction*loss[hard].mean()
        losses.append(term);counts.append(int(mask.sum()))
    return sum(losses) if losses else z.sum()*0,counts

def fit(x,y,groups,held,epochs,mode,baseline_alpha=10000):
    torch.manual_seed(2026);generator=torch.Generator().manual_seed(2026)
    scaler=StandardScaler().fit(x);sx=scaler.transform(x).astype(np.float32);sh=scaler.transform(held).astype(np.float32)
    with threadpool_limits(limits=2):base=Ridge(alpha=baseline_alpha,solver='cholesky').fit(sx.astype(np.float64),y)
    baseline=base.predict(sx).astype(np.float32);hb=base.predict(sh).astype(np.float32)
    model=Head(x.shape[1]);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    tx=torch.from_numpy(sx);raw=torch.from_numpy(x.astype(np.float32));raw=raw/raw.norm(dim=1,keepdim=True).clamp_min(1e-12)
    target=torch.from_numpy((y-baseline).astype(np.float32));price=torch.from_numpy((y*1000).astype(np.float32));group=torch.from_numpy(pd.factorize(groups)[0])
    predictions={};history=[];lam=0 if mode=='regression' else .01
    for epoch in range(1,max(epochs)+1):
        model.train();total=0.;aux=0.;pairs=[0,0]
        for ids in torch.randperm(len(tx),generator=generator).split(256):
            optimizer.zero_grad(set_to_none=True);pred,z=model(tx[ids]);mse=((pred-target[ids])**2).mean()
            contrast,counts=(z.sum()*0,[0,0]) if mode=='regression' else pair_loss(z,raw[ids],price[ids],group[ids],mode,epoch);loss=mse+lam*contrast
            if not torch.isfinite(loss):raise ValueError('Nonfinite contrastive loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            total+=mse.item()*len(ids);aux+=contrast.item()*len(ids);pairs=[a+b for a,b in zip(pairs,counts)]
        history.append({'epoch':epoch,'residual_train_mse':total/len(tx)*1e6,'contrastive_loss':aux/len(tx),'positive_pairs':pairs[0],'negative_pairs':pairs[1]})
        if epoch in epochs:
            model.eval()
            with torch.inference_mode():predictions[epoch]=hb+model(torch.from_numpy(sh))[0].numpy()
    return predictions,model,scaler,base,history

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['regression','easy','curriculum'],required=True)
    p.add_argument('--reference-kernel');p.add_argument('--label-prefix',default='l14_head');p.add_argument('--baseline-alpha',type=float,default=10000)
    args=p.parse_args();torch.set_num_threads(2)
    source_registration=None
    if args.reference_kernel:
        source_registration=json.loads((REPOSITORY/'output/experiment_record'/args.reference_kernel/'registration.json').read_text())
        cache=local_path('cache_root')/source_registration['cache_family']/source_registration['feature_cache_key']
    else:
        cache=local_path('cache_root')/'clip_features/6cc6ea162d96836a66ec8b9b53757f40f2853e0029fdfd1dcf48ef23f98c3529'
    manifest=json.loads((cache/'manifest.json').read_text())
    split=local_path('split_path');frame=pd.read_csv(split,dtype={'imageid':str,'group_id':str});assert digest(split)==manifest['split_sha256'] and frame.imageid.tolist()==manifest['imageids']
    if source_registration:
        from visual_kernel_submission import features_for
        x=features_for(source_registration,frame)
    else:x=np.concatenate([np.load(p) for p in sorted(cache.glob('*.npy'))])
    assert x.ndim==2 and len(x)==8000 and np.isfinite(x).all()
    train=frame.partition.eq('train').to_numpy();valid=frame.partition.eq('validation').to_numpy();assert train.sum()==6399 and valid.sum()==1601
    xt=x[train];y=frame.loc[train,'price'].to_numpy()/1000;groups=frame.loc[train,'group_id'].to_numpy();epochs=[10,30,60]
    run=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8];label=args.label_prefix+'_'+args.mode
    out=local_path('records_root')/'16_price_contrastive'/label/run;out.mkdir(parents=True)
    public=REPOSITORY/'output/experiment_record/16_price_contrastive'/label/run;public.mkdir(parents=True)
    config={'mode':args.mode,'lambda':0 if args.mode=='regression' else .01,'epochs':epochs,'margin':.5,'head':f'Fixed fit-fold Ridge alpha{args.baseline_alpha} + residual {x.shape[1]}->64->1, contrast projection64->32',
        'positive':'log price distance <= log1.15; different group; visual near easy / far hard','negative':'log price distance >= log2; different group; visual far easy / near hard',
        'visual_difficulty':'Lower/upper quartile among price-eligible pairs in training microbatch, computed from original frozen unit CLIP embeddings',
        'curriculum':'Hard pair fraction 0 until epoch10, linear rise to25% by epoch20; positive and negative separately normalized',
        'lr':.001,'weight_decay':.01,'dropout':.25,'batch_size':256,'seed':2026,'backbone_frozen':True,'source_sha256':digest(__file__),'split_sha256':digest(split),
        'feature_manifest_sha256':digest(cache/'manifest.json'),'selection':'Epoch chosen on 3-fold training GroupKFold unweighted price MSE; validation once',
        'scope':'Frozen-feature head experiment; does not claim image-encoder attention changed; source checkpoint scope retained','test_images_used':False,
        'reference_kernel':args.reference_kernel,'feature_dim':x.shape[1],'baseline_alpha':args.baseline_alpha,
        'source_course_scope':source_registration['course_scope'] if source_registration else 'CLIP course eligibility unresolved'}
    write_json(public/'registration.json',config)
    def status(s,**kw):write_json(public/'status.json',{'status':s,'pid':os.getpid(),**kw})
    status('fitting');start=time.perf_counter();oof={e:np.full(len(y),np.nan) for e in epochs}
    try:
        for n,(fit_ids,held_ids) in enumerate(GroupKFold(3).split(xt,y,groups)):
            assert not set(groups[fit_ids])&set(groups[held_ids]);status('fitting',fold=n+1)
            values,_,_,_,history=fit(xt[fit_ids],y[fit_ids],groups[fit_ids],xt[held_ids],epochs,args.mode,args.baseline_alpha)
            for e in epochs:oof[e][held_ids]=values[e]
            pd.DataFrame(history).to_csv(public/f'fold_{n+1}_history.csv',index=False);print(f'{args.mode}: fold {n+1}/3',flush=True)
        scores=[{'epochs':e,'training_oof_mse':float(np.mean((np.maximum(oof[e],0)-y)**2)*1e6)} for e in epochs];pd.DataFrame(scores).to_csv(public/'training_cv.csv',index=False)
        best=min(scores,key=lambda r:r['training_oof_mse']);status('refitting',epochs=best['epochs'])
        values,model,scaler,base,history=fit(xt,y,groups,x[valid],[best['epochs']],args.mode,args.baseline_alpha)
        metrics=save_predictions(out,frame.loc[valid],values[best['epochs']]*1000,1300)
        torch.save(model.state_dict(),out/'head.pt');joblib.dump({'scaler':scaler,'ridge':base},out/'base.joblib');pd.DataFrame(history).to_csv(public/'refit_history.csv',index=False)
        pd.DataFrame({'imageid':frame.loc[train,'imageid'],'raw_oof_prediction':oof[best['epochs']]*1000,'oof_prediction':np.maximum(oof[best['epochs']],0)*1000}).to_csv(out/'selected_training_oof.csv',index=False)
        write_json(public/'results.json',{'head':args.mode,**best,**metrics,'seconds':time.perf_counter()-start});print(json.dumps({**best,**metrics}),flush=True);status('completed')
    except Exception as exc:status('failed',error=repr(exc));raise

if __name__=='__main__':main()
