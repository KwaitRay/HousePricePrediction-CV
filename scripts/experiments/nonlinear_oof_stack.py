"""Fixed conservative nonlinear fusion on base training OOF predictions only."""
import argparse,time,uuid,json
import numpy as np,pandas as pd,joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from common.local_paths import local_path,REPOSITORY
from common.config import digest,write_json
from common.metrics import save_predictions

def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',required=True);p.add_argument('--name',required=True)
    p.add_argument('--method',choices=['hgb','ridge'],default='hgb');args=p.parse_args()
    root=local_path('records_root');split=local_path('split_path');frame=pd.read_csv(split,dtype={'imageid':str,'group_id':str})
    train=frame[frame.partition=='train'];val=frame[frame.partition=='validation'];assert len(train)==6399 and len(val)==1601
    tx=[];vx=[];provenance=[]
    for name in args.models:
        model=root/name;oo=pd.read_csv(model/'selected_training_oof.csv',dtype={'imageid':str});pv=pd.read_csv(model/'predictions.csv',dtype={'imageid':str})
        assert oo.imageid.is_unique and pv.imageid.is_unique and set(oo.imageid)==set(train.imageid) and set(pv.imageid)==set(val.imageid)
        tx.append(oo.set_index('imageid').loc[train.imageid,'oof_prediction'].to_numpy()/1000);vx.append(pv.set_index('imageid').loc[val.imageid,'prediction'].to_numpy()/1000)
        provenance.append({'model':name,'head_sha256':digest(model/'head.npz'),'oof_sha256':digest(model/'selected_training_oof.csv'),'prediction_sha256':digest(model/'predictions.csv')})
    x=np.stack(tx,axis=1);v=np.stack(vx,axis=1);y=train.price.to_numpy()/1000;assert np.isfinite(x).all() and np.isfinite(v).all()
    params={'loss':'squared_error','learning_rate':.05,'max_iter':100,'max_depth':2,'max_leaf_nodes':4,'min_samples_leaf':100,'l2_regularization':1.,'early_stopping':False,'random_state':2026} if args.method=='hgb' else {'alpha':1.,'fit_intercept':True,'solver':'cholesky'}
    run=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
    out=root/'15_fixed_ensembles'/args.name/run;out.mkdir(parents=True)
    public=REPOSITORY/'output/experiment_record/15_fixed_ensembles'/args.name/run;public.mkdir(parents=True)
    write_json(public/'registration.json',{'components':provenance,'params':params,'method':args.method,'fit_scope':'6399 training base OOF predictions and training labels only; no validation meta fitting',
        'features':'Only numeric image-model price predictions; no filename, price-tier label, scene labels or metadata input',
        'source_sha256':digest(__file__),'split_sha256':digest(split),'test_images_used':False,'course_scope':'Exploratory CLIP/BLIP visual components, eligibility unresolved',
        'caveat':'Meta training-fit error is not independent OOF performance; development and base hyperparameter selection bias remains'})
    model=HistGradientBoostingRegressor(**params) if args.method=='hgb' else Ridge(**params)
    with threadpool_limits(limits=2):model.fit(x,y);raw=model.predict(v)*1000;fit_pred=model.predict(x)*1000
    metrics=save_predictions(out,val,raw,1300);joblib.dump(model,out/'meta_head.joblib')
    write_json(out/'ensemble_components.json',provenance)
    result={'head':'nonlinear_oof_stack' if args.method=='hgb' else 'affine_oof_stack',**metrics,'meta_training_fit_mse':float(np.mean((np.maximum(fit_pred,0)-train.price.to_numpy())**2))}
    if args.method=='ridge':result.update(coefficients=model.coef_.tolist(),intercept=float(model.intercept_))
    write_json(public/'results.json',result);write_json(public/'status.json',{'status':'completed'})
    pred=np.maximum(raw,0);price=val.price.to_numpy();slices=[]
    for name,mask in [('low',price<=300),('middle',(price>300)&(price<=1300)),('high',price>1300)]:
        error=pred[mask]-price[mask];slices.append({'tier':name,'n':int(mask.sum()),'mse':float(np.mean(error**2)),'mae':float(np.mean(abs(error))),'bias':float(np.mean(error))})
    write_json(public/'price_slices.json',slices);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
