"""Frozen CLIP Ridge baseline plus a small learned nonlinear residual; train-only CV."""
import json,os,time,uuid
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import joblib
from common.local_paths import local_path,REPOSITORY
from common.config import digest,write_json
from common.metrics import save_predictions

def fit(x,y,groups,held,epochs,seed=2026):
    torch.manual_seed(seed)
    scaler=StandardScaler().fit(x);x=scaler.transform(x).astype(np.float32);held=scaler.transform(held).astype(np.float32)
    with threadpool_limits(limits=2):base=Ridge(alpha=1000,solver='svd').fit(x,y)
    baseline=base.predict(x).astype(np.float32);held_baseline=base.predict(held).astype(np.float32)
    model=nn.Sequential(nn.Linear(x.shape[1],64),nn.GELU(),nn.Dropout(0.25),nn.Linear(64,1))
    nn.init.zeros_(model[-1].weight);nn.init.zeros_(model[-1].bias)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-2)
    tx=torch.from_numpy(x);target=torch.from_numpy((y-baseline).astype(np.float32));vh=torch.from_numpy(held)
    predictions={};losses=[]
    for epoch in range(1,max(epochs)+1):
        model.train();order=torch.randperm(len(tx));total=0
        for ids in order.split(256):
            optimizer.zero_grad(set_to_none=True);pred=model(tx[ids]).reshape(-1);loss=((pred-target[ids])**2).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite residual loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();total+=loss.item()*len(ids)
        losses.append({'epoch':epoch,'residual_train_mse':total/len(tx)*1e6})
        if epoch in epochs:
            model.eval()
            with torch.inference_mode():predictions[epoch]=held_baseline+model(vh).reshape(-1).numpy()
    return predictions,model,scaler,base,losses

def main():
    torch.set_num_threads(2)
    manifest_path=local_path('records_root')/'03_clip_reference/runs/20261013T043141Z_c9f899f2/data_manifest.json'
    manifest=json.loads(manifest_path.read_text());split=local_path('split_path');assert digest(split)==manifest['split_sha256']
    frame=pd.read_csv(split,dtype={'imageid':str,'group_id':str});assert frame.imageid.tolist()==manifest['imageids']
    import hashlib
    key=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    cache=local_path('cache_root')/'clip_features'/key
    x=np.concatenate([np.load(p,allow_pickle=False) for p in sorted(cache.glob('*.npy'))]);assert x.shape==(8000,512)
    train=frame.partition.eq('train').to_numpy();valid=frame.partition.eq('validation').to_numpy()
    assert train.sum()==6399 and valid.sum()==1601
    xt=x[train];y=frame.loc[train,'price'].to_numpy()/1000;groups=frame.loc[train,'group_id'].to_numpy()
    epochs=[10,30,60];run=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
    out=local_path('records_root')/'13_clip_optimization/clip_residual'/run;out.mkdir(parents=True)
    public=REPOSITORY/'output/experiment_record/13_clip_optimization/clip_residual'/run;public.mkdir(parents=True)
    write_json(public/'registration.json',{'method':'Fixed Ridge alpha1000 plus MLP residual 512->64->1','epochs':epochs,'dropout':0.25,'weight_decay':0.01,
        'lr':0.001,'batch':256,'seed':2026,'selection':'3-fold train-only GroupKFold original-price MSE; validation once after epoch selection',
        'normalization':'each fold train-only StandardScaler','backbone_frozen':True,'course_scope':'CLIP exploratory; eligibility unresolved',
        'split_sha256':digest(split),'manifest_sha256':digest(manifest_path),'source_sha256':digest(__file__),'cache_key':key})
    def status(state,**kw):write_json(public/'status.json',{'status':state,'pid':os.getpid(),**kw})
    status('running');start=time.perf_counter()
    try:
        oof={e:np.full(len(y),np.nan) for e in epochs}
        for n,(fit_ids,held_ids) in enumerate(GroupKFold(3).split(xt,y,groups)):
            assert not set(groups[fit_ids])&set(groups[held_ids]);status('fitting',fold=n+1)
            predictions,_,_,_,losses=fit(xt[fit_ids],y[fit_ids],groups[fit_ids],xt[held_ids],epochs)
            for e in epochs:oof[e][held_ids]=predictions[e]
            pd.DataFrame(losses).to_csv(public/f'fold_{n+1}_history.csv',index=False)
            print(f'CLIP residual fold {n+1}/3 complete',flush=True)
        scores=[{'epochs':e,'training_oof_mse':float(np.mean((np.maximum(oof[e],0)-y)**2)*1e6)} for e in epochs]
        pd.DataFrame(scores).to_csv(public/'training_cv.csv',index=False)
        chosen=min(scores,key=lambda s:s['training_oof_mse']);status('refitting',epochs=chosen['epochs'])
        pred,model,scaler,base,losses=fit(xt,y,groups,x[valid],[chosen['epochs']])
        metrics=save_predictions(out,frame.loc[valid],pred[chosen['epochs']]*1000,1300)
        torch.save(model.state_dict(),out/'residual.pt');joblib.dump({'scaler':scaler,'ridge':base},out/'base.joblib')
        pd.DataFrame(losses).to_csv(public/'refit_history.csv',index=False)
        pd.DataFrame({'imageid':frame.loc[train,'imageid'],'oof_prediction':np.maximum(oof[chosen['epochs']],0)*1000}).to_csv(out/'selected_training_oof.csv',index=False)
        write_json(public/'results.json',{**chosen,**metrics,'seconds':time.perf_counter()-start})
        print(json.dumps({**chosen,**metrics}),flush=True);status('completed')
    except Exception as exc:status('failed',error=repr(exc));raise

if __name__=='__main__':main()
