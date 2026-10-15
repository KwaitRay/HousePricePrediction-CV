"""Independently reload the MLP and measure its single increment to the fixed baseline."""
from pathlib import Path
import sys,os,time,uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
import numpy as np
import pandas as pd
import torch,joblib
from threadpoolctl import threadpool_limits
from common.config import read_json,write_json,digest
from common.local_paths import local_path
from common.metrics import save_predictions
from visual_kernel_submission import features_for
from clip_price_contrastive_head import Head
from encoder_oof_fusion import fit_weights,meta_cv
RUN='16_price_contrastive/pe_giant_whole_head_regression/20261015T032715Z_69ac7909'
BASE='15_fixed_ensembles/pe_giant_whole_incremental_fusion/20261014T082727Z_c5152cef'
root=local_path('records_root');public=ROOT/'output/experiment_record'/RUN;out=root/RUN
assert read_json(public/'status.json')['status']=='completed'
reg=read_json(public/'registration.json');result=read_json(public/'results.json')
assert digest(ROOT/'scripts/experiments/clip_price_contrastive_head.py')==reg['source_sha256']
assert reg['mode']=='regression' and reg['lambda']==0 and reg['backbone_frozen'] and reg['test_images_used'] is False
frame=pd.read_csv(local_path('split_path'),dtype={'imageid':str,'group_id':str})
assert digest(local_path('split_path'))==reg['split_sha256']
train=frame.partition.eq('train').to_numpy();valid=frame.partition.eq('validation').to_numpy()
assert train.sum()==6399 and valid.sum()==1601 and not set(frame.loc[train,'group_id'])&set(frame.loc[valid,'group_id'])
source_reg=read_json(ROOT/'output/experiment_record'/reg['reference_kernel']/'registration.json')
x=features_for(source_reg,frame)
torch.set_num_threads(2)
with threadpool_limits(limits=2):
    fitted=joblib.load(out/'base.joblib')
    np.testing.assert_allclose(fitted['scaler'].mean_,x[train].mean(axis=0),rtol=1e-10,atol=1e-12)
    values=fitted['scaler'].transform(x[valid]).astype(np.float32)
    baseline=fitted['ridge'].predict(values.astype(np.float64)).astype(np.float32)
    model=Head(2560);model.load_state_dict(torch.load(out/'head.pt',map_location='cpu',weights_only=True));model.eval()
    with torch.inference_mode():raw=(baseline+model(torch.from_numpy(values))[0].numpy())*1000
saved=pd.read_csv(out/'predictions.csv',dtype={'imageid':str})
assert saved.imageid.tolist()==frame.loc[valid,'imageid'].tolist()
np.testing.assert_array_equal(saved.price,frame.loc[valid,'price'])
np.testing.assert_allclose(raw,saved.raw_prediction,rtol=1e-7,atol=1e-4)
np.testing.assert_allclose(np.mean((np.maximum(raw,0).astype(np.float64)-saved.price.to_numpy())**2),result['mse'],rtol=1e-10)
write_json(public/'independent_artifact_audit.json',{'passed':True,'validation_n':1601,
    'max_price_error':float(abs(raw-saved.raw_prediction.to_numpy()).max()),'source_sha256':digest(__file__),
    'head_sha256':digest(out/'head.pt'),'base_sha256':digest(out/'base.joblib'),
    'scope':'CPU reload all1601 MLP+Ridge prices and training scaler means, no refit of all OOF networks',
    'test_images_used':False,'export_eligible':False})
base=read_json(root/BASE/'ensemble.json');models=base['models']+[RUN]
relative=Path('15_fixed_ensembles/pe_whole_mlp_incremental_fusion')/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
local=root/relative;pub=ROOT/'output/experiment_record'/relative;local.mkdir(parents=True,exist_ok=False)
registration={'baseline':BASE,'added_model':RUN,'models':models,'source_sha256':digest(__file__),
    'optimizer_source_sha256':digest(ROOT/'scripts/experiments/encoder_oof_fusion.py'),
    'baseline_ensemble_sha256':digest(root/BASE/'ensemble.json'),'split_sha256':digest(local_path('split_path')),
    'method':'Add exactly one frozen-feature residual MLP to the fixed baseline; training OOF convex weights',
    'gate':{'minimum_meta_cv_relative_gain':.01,'minimum_folds_won':2,'validation_improvement_required':True},
    'caveat':'Single training seed screen; hyperparameters selected on same OOF, not nested unbiased CV; three-seed adoption and MLP export adapter not implemented',
    'automatic_test_export':False,'test_images_used':False}
write_json(pub/'registration.json',registration);(local/'source.py').write_bytes(Path(__file__).read_bytes())
write_json(pub/'status.json',{'status':'fitting','pid':os.getpid()})
z=[];v=[];heads=[];hashes=[];oof_hashes=[]
for i,m in enumerate(models):
    d=root/m;assert read_json(ROOT/'output/experiment_record'/m/'status.json')['status']=='completed'
    heads.append(digest(d/('head.pt' if m==RUN else 'head.npz')))
    hashes.append(digest(d/'predictions.csv'));oof_hashes.append(digest(d/'selected_training_oof.csv'))
    if i<len(base['models']):assert heads[-1]==base['model_sha256'][i] and hashes[-1]==base['prediction_sha256'][i]
    a=pd.read_csv(d/'selected_training_oof.csv',dtype={'imageid':str});b=pd.read_csv(d/'predictions.csv',dtype={'imageid':str})
    assert a.imageid.tolist()==frame.loc[train,'imageid'].tolist() and b.imageid.tolist()==frame.loc[valid,'imageid'].tolist()
    np.testing.assert_array_equal(b.price,frame.loc[valid,'price'])
    z.append(a.oof_prediction.to_numpy()/1000);v.append(b.prediction.to_numpy())
z=np.column_stack(z);v=np.column_stack(v);y=frame.loc[train,'price'].to_numpy()/1000
with threadpool_limits(limits=2):
    folds,bmeta,nmeta=meta_cv(z,y,frame.loc[train,'group_id'].to_numpy(),len(base['models']),1e-12)
    weights=fit_weights(z,y,1e-12)
fixed=v[:,:-1]@np.asarray(base['weights']);previous=pd.read_csv(root/BASE/'predictions.csv')
np.testing.assert_allclose(fixed,previous.prediction,rtol=1e-7,atol=1e-4)
metrics=save_predictions(local,frame.loc[valid],v@weights,1300)
bmse,nmse=[float(np.mean((p-y)**2)*1e6) for p in [bmeta,nmeta]]
bm=float(np.mean((fixed-frame.loc[valid,'price'].to_numpy())**2));wins=sum(f['augmented_mse']<f['baseline_mse'] for f in folds)
passed=nmse<bmse*.99 and wins>=2 and metrics['mse']<bm
result={**metrics,'baseline_validation_mse':bm,'baseline_meta_cv_mse':bmse,'meta_cv_mse':nmse,
    'meta_relative_gain':1-nmse/bmse,'folds_won':wins,'weights':weights.tolist(),'screen_passed':bool(passed),
    'export_eligible':False,'three_seed_adoption_completed':False,'full_refit_adapter_supported':False}
write_json(pub/'results.json',result);write_json(pub/'meta_folds.json',folds)
write_json(local/'ensemble.json',{'models':models,'weights':weights.tolist(),'model_sha256':heads,'prediction_sha256':hashes,
    'oof_sha256':oof_hashes,'mlp_base_sha256':digest(out/'base.joblib'),'test_images_used':False,'export_eligible':False})
write_json(pub/'status.json',{'status':'completed'});print(str(relative),flush=True);print(result,flush=True)
