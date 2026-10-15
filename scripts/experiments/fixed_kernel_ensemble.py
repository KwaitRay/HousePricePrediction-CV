"""Explicit equal-weight kernel-model composition, without validation weight fitting."""
import argparse,json,time,uuid
import numpy as np,pandas as pd
from common.local_paths import REPOSITORY,local_path
from common.config import digest,write_json
from common.metrics import save_predictions

def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',required=True);p.add_argument('--name',required=True);p.add_argument('--learn-oof-weights',action='store_true');args=p.parse_args()
    assert len(args.models)>=2 and len(set(args.models))==len(args.models)
    root=local_path('records_root');models=[root/name for name in args.models]
    split=local_path('split_path');frame=pd.read_csv(split,dtype={'imageid':str,'group_id':str})
    val=frame[frame.partition=='validation'].copy();train=frame[frame.partition=='train'].copy()
    assert len(val)==1601 and len(train)==6399
    predictions=[];oofs=[]
    for model in models:
        assert (model/'head.npz').exists()
        pv=pd.read_csv(model/'predictions.csv',dtype={'imageid':str});pt=pd.read_csv(model/'selected_training_oof.csv',dtype={'imageid':str})
        assert pv.imageid.is_unique and pt.imageid.is_unique and set(pv.imageid)==set(val.imageid) and set(pt.imageid)==set(train.imageid)
        values=pv.set_index('imageid').loc[val.imageid];np.testing.assert_allclose(values.price.to_numpy(),val.price.to_numpy(),rtol=0,atol=0)
        predictions.append(values.prediction.to_numpy());oofs.append(pt.set_index('imageid').loc[train.imageid,'oof_prediction'].to_numpy())
    weights=np.full(len(models),1/len(models));fit_info=None
    if args.learn_oof_weights:
        from scipy.optimize import minimize
        z=np.stack(oofs,axis=1)/1000;y=train.price.to_numpy()/1000
        optimization=minimize(lambda w:float(np.mean((z@w-y)**2)),weights,jac=lambda w:2*z.T@(z@w-y)/len(y),
            bounds=[(0,1)]*len(models),constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones_like(w)}],
            method='SLSQP',options={'ftol':1e-12,'maxiter':1000})
        assert optimization.success and np.isfinite(optimization.x).all() and abs(optimization.x.sum()-1)<1e-8
        weights=optimization.x;fit_info={'algorithm':'SLSQP convex MSE minimization','scope':'Only training-component OOF predictions and training labels',
            'caveat':'This meta fitting criterion is not an independent OOF estimate; base hyperparameters already selected using training folds'}
    pred=np.stack(predictions,axis=1)@weights;oof=np.stack(oofs,axis=1)@weights
    assert np.isfinite(pred).all() and np.isfinite(oof).all()
    run=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
    out=root/'15_fixed_ensembles'/args.name/run;out.mkdir(parents=True)
    public=REPOSITORY/'output/experiment_record/15_fixed_ensembles'/args.name/run;public.mkdir(parents=True)
    registration={'models':args.models,'weights':weights.tolist(),'weights_learned':args.learn_oof_weights,'weight_fit':fit_info,
        'rule':'Convex average of constituent nonnegative predictions; no validation weight fitting',
        'model_sha256':[digest(m/'head.npz') for m in models],'prediction_sha256':[digest(m/'predictions.csv') for m in models],
        'source_sha256':digest(__file__),'split_sha256':digest(split),'test_images_used':False,
        'course_scope':'Exploratory visual CLIP/BLIP components; eligibility unresolved',
        'caveat':'Constituents selected after development comparisons; OOF from previously selected hyperparameters, not nested unbiased evaluation'}
    write_json(out/'ensemble.json',registration);write_json(public/'registration.json',registration)
    metrics=save_predictions(out,val,pred,1300)
    pd.DataFrame({'imageid':train.imageid,'prediction' if args.learn_oof_weights else 'oof_prediction':oof}).to_csv(out/('meta_training_predictions.csv' if args.learn_oof_weights else 'selected_training_oof.csv'),index=False)
    result={'head':'convex_oof_stack' if args.learn_oof_weights else 'fixed_equal_ensemble',**metrics,
        'meta_training_fit_mse' if args.learn_oof_weights else 'training_oof_mse':float(np.mean((oof-train.price.to_numpy())**2)),'weights':weights.tolist()}
    write_json(public/'results.json',result);write_json(public/'status.json',{'status':'completed'})
    error_matrix=np.stack(predictions)-val.price.to_numpy()[None,:]
    write_json(public/'error_correlations.json',np.corrcoef(error_matrix).tolist())
    slices=[]
    for name,mask in [('low',val.price<=300),('middle',(val.price>300)&(val.price<=1300)),('high',val.price>1300)]:
        error=pred[mask]-val.price.to_numpy()[mask];slices.append({'tier':name,'n':int(mask.sum()),'mse':float(np.mean(error**2)),'mae':float(np.mean(abs(error))),'bias':float(np.mean(error))})
    write_json(public/'price_slices.json',slices);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
