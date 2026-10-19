"""Fixed 50/50 average of two selected CLIP kernel models; no learned blend weights."""
import json,time,uuid
import numpy as np,pandas as pd
from common.local_paths import REPOSITORY,local_path
from common.config import digest,write_json
from common.metrics import save_predictions

def main():
    a=local_path('records_root')/'13_clip_optimization/clip_b16_kernel/20261013T111716Z_5ecc630f'
    b=local_path('records_root')/'13_clip_optimization/clip_multiview_kernel/20261013T111418Z_5ad0f81d'
    frame=pd.read_csv(local_path('split_path'),dtype={'imageid':str,'group_id':str});validation=frame[frame.partition=='validation']
    first=pd.read_csv(a/'predictions.csv',dtype={'imageid':str,'group_id':str});second=pd.read_csv(b/'predictions.csv',dtype={'imageid':str,'group_id':str})
    assert len(first)==len(second)==len(validation)==1601 and set(first.imageid)==set(validation.imageid)==set(second.imageid)
    joined=validation.merge(first[['imageid','prediction']],on='imageid',validate='one_to_one').merge(second[['imageid','prediction']],on='imageid',suffixes=('_b16','_multi'),validate='one_to_one')
    pred=(joined.prediction_b16.to_numpy()+joined.prediction_multi.to_numpy())/2
    run=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
    out=local_path('records_root')/'13_clip_optimization/clip_equal_ensemble'/run;out.mkdir(parents=True)
    public=REPOSITORY/'output/experiment_record/13_clip_optimization/clip_equal_ensemble'/run;public.mkdir(parents=True)
    registration={'weights':[0.5,0.5],'weights_learned':False,'models':[str(a.relative_to(local_path('records_root'))),str(b.relative_to(local_path('records_root')))],
        'model_sha256':[digest(a/'head.npz'),digest(b/'head.npz')],'prediction_sha256':[digest(a/'predictions.csv'),digest(b/'predictions.csv')],
        'rule':'Average each constituent nonnegative prediction with fixed equal weight','split_sha256':digest(local_path('split_path')),'source_sha256':digest(__file__),
        'course_scope':'Exploratory CLIP ensemble; eligibility unresolved','selection_caveat':'Constituents selected after development validation comparisons; no fitting of blend weights to validation labels'}
    write_json(out/'ensemble.json',registration);write_json(public/'registration.json',registration)
    metrics=save_predictions(out,joined,pred,1300)
    oo1=pd.read_csv(a/'selected_training_oof.csv',dtype={'imageid':str});oo2=pd.read_csv(b/'selected_training_oof.csv',dtype={'imageid':str})
    train=frame[frame.partition=='train'].merge(oo1[['imageid','oof_prediction']],on='imageid',validate='one_to_one').merge(oo2[['imageid','oof_prediction']],on='imageid',suffixes=('_b16','_multi'),validate='one_to_one')
    assert len(train)==6399
    train_pred=(train.oof_prediction_b16.to_numpy()+train.oof_prediction_multi.to_numpy())/2
    pd.DataFrame({'imageid':train.imageid,'oof_prediction':train_pred}).to_csv(out/'selected_training_oof.csv',index=False)
    result={'head':'fixed_equal_ensemble',**metrics,'training_oof_mse':float(np.mean((train_pred-train.price.to_numpy())**2)),
        'validation_error_correlation':float(np.corrcoef(joined.prediction_b16-joined.price,joined.prediction_multi-joined.price)[0,1])}
    write_json(public/'results.json',result);write_json(public/'status.json',{'status':'completed'})
    slices=[]
    for name,mask in [('low',joined.price<=300),('middle',(joined.price>300)&(joined.price<=1300)),('high',joined.price>1300)]:
        error=pred[mask]-joined.price.to_numpy()[mask];slices.append({'tier':name,'n':int(mask.sum()),'mse':float(np.mean(error**2)),'mae':float(np.mean(abs(error))),'bias':float(np.mean(error))})
    write_json(public/'price_slices.json',slices);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
