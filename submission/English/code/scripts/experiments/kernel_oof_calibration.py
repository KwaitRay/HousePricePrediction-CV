"""Fit a global affine price calibration using only base training OOF predictions."""
import argparse,time,uuid,json
import numpy as np,pandas as pd
from common.local_paths import local_path,REPOSITORY
from common.config import digest,write_json
from common.metrics import save_predictions

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--name',required=True);args=p.parse_args()
    source=local_path('records_root')/args.model;split=local_path('split_path');frame=pd.read_csv(split,dtype={'imageid':str,'group_id':str})
    train=frame[frame.partition=='train'];val=frame[frame.partition=='validation']
    oo=pd.read_csv(source/'selected_training_oof.csv',dtype={'imageid':str});pv=pd.read_csv(source/'predictions.csv',dtype={'imageid':str})
    assert set(oo.imageid)==set(train.imageid) and set(pv.imageid)==set(val.imageid) and oo.imageid.is_unique and pv.imageid.is_unique
    training_prediction=oo.set_index('imageid').loc[train.imageid,'oof_prediction'].to_numpy()/1000
    coef=np.linalg.lstsq(np.column_stack([training_prediction,np.ones(len(train))]),train.price.to_numpy()/1000,rcond=None)[0]
    assert np.isfinite(coef).all()
    validation_prediction=pv.set_index('imageid').loc[val.imageid,'prediction'].to_numpy()/1000
    raw=(validation_prediction*coef[0]+coef[1])*1000
    run=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
    out=local_path('records_root')/'15_fixed_ensembles'/args.name/run;out.mkdir(parents=True)
    public=REPOSITORY/'output/experiment_record/15_fixed_ensembles'/args.name/run;public.mkdir(parents=True)
    registration={'source_model':args.model,'slope':float(coef[0]),'intercept_thousand_usd':float(coef[1]*1000),
        'fit_scope':'Only base training OOF predictions and training labels; no validation calibration fitting','transform':'max(0,slope*base_nonnegative_prediction+intercept)',
        'source_head_sha256':digest(source/'head.npz'),'source_oof_sha256':digest(source/'selected_training_oof.csv'),'split_sha256':digest(split),'source_sha256':digest(__file__),
        'caveat':'Calibration training-fit error is not independent OOF performance; base hyperparameters already selected on these folds; development model selection bias remains',
        'course_scope':'Exploratory CLIP visual components, eligibility unresolved','test_images_used':False}
    write_json(out/'calibration.json',registration);write_json(public/'registration.json',registration)
    metrics=save_predictions(out,val,raw,1300)
    result={'head':'train_oof_affine_calibration',**metrics,'slope':float(coef[0]),'intercept_thousand_usd':float(coef[1]*1000)}
    write_json(public/'results.json',result);write_json(public/'status.json',{'status':'completed'});print(json.dumps(result),flush=True)

if __name__=='__main__':main()
