"""Audit selected whole-PE PLS equations/prices, then run its fixed incremental fusion."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
import numpy as np
import pandas as pd
import joblib
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from common.config import read_json,write_json,digest
from common.local_paths import local_path
from visual_kernel_submission import features_for
from pe_pls_incremental_fusion import main as fuse
RUN='19_image_embedding_reference/pe_core_giant_whole_price_metric_kernel/20261015T031833Z_add4cd08'
BASE='15_fixed_ensembles/pe_giant_whole_incremental_fusion/20261014T082727Z_c5152cef'
public=ROOT/'output/experiment_record'/RUN;local=local_path('records_root')/RUN
binding=read_json(public/'audit_fusion_binding.json')
assert digest(__file__)==binding['audit_source_sha256']
for name,sha in binding['dependency_sha256'].items():assert digest(ROOT/'scripts/experiments'/name)==sha
assert read_json(public/'status.json')['status']=='completed'
reg=read_json(public/'registration.json');result=read_json(public/'results.json')
assert digest(ROOT/'scripts/experiments/visual_price_metric_kernel.py')==reg['source_sha256']
assert digest(local/'source.py')==reg['source_sha256']
assert digest(local_path('split_path'))==reg['split_sha256']
frame=pd.read_csv(local_path('split_path'),dtype={'imageid':str,'group_id':str})
train=frame.partition.eq('train').to_numpy();valid=frame.partition.eq('validation').to_numpy()
assert train.sum()==6399 and valid.sum()==1601 and not set(frame.loc[train,'group_id'])&set(frame.loc[valid,'group_id'])
source_reg=read_json(ROOT/'output/experiment_record'/reg['source']/'registration.json')
x=features_for(source_reg,frame)
with np.load(local/'head.npz',allow_pickle=False) as h:head={k:h[k].copy() for k in h.files}
np.testing.assert_array_equal(head['train_features'],x[train])
components=int(head['components']);mean=float(head['target_mean'])
assert components==result['components'] and float(head['alpha'])==result['alpha']
np.testing.assert_allclose(mean,frame.loc[train,'price'].mean()/1000,rtol=1e-12)
metric=joblib.load(local/'feature_metric.joblib') if components else None
with threadpool_limits(limits=2):
    if metric:
        np.testing.assert_allclose(metric['pls']._x_mean,x[train].mean(axis=0),rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(metric['pls']._y_mean,[mean],rtol=1e-10,atol=1e-12)
        transformed=metric['pls'].transform(x)
        np.testing.assert_allclose(metric['scaler'].mean_,transformed[train].mean(axis=0),rtol=1e-10,atol=1e-10)
        scores=metric['scaler'].transform(transformed)[:,:components]
        np.testing.assert_allclose(scores[train],head['train_pls_scores'],rtol=1e-10,atol=1e-10)
    def replay(indices):
        outputs=[]
        for start in range(0,len(indices),128):
            rows=indices[start:start+128]
            k=np.exp(-float(head['image_gamma'])*np.maximum(2-2*(x[rows]@head['train_features'].T),0))
            if components:k*=np.exp(-float(head['pls_gamma'])*cdist(scores[rows],head['train_pls_scores'],'sqeuclidean')/components)
            outputs.extend(((k@head['coefficients']+mean)*1000).tolist())
        return np.asarray(outputs)
    training_prices=replay(np.flatnonzero(train));raw=replay(np.flatnonzero(valid))
expected=frame.loc[train,'price'].to_numpy()-1000*float(head['alpha'])*head['coefficients']
np.testing.assert_allclose(training_prices,expected,rtol=1e-7,atol=1e-4)
table=pd.read_csv(local/'predictions.csv',dtype={'imageid':str,'group_id':str})
assert table.imageid.tolist()==frame.loc[valid,'imageid'].tolist() and table.group_id.tolist()==frame.loc[valid,'group_id'].tolist()
np.testing.assert_array_equal(table.price,frame.loc[valid,'price'])
np.testing.assert_allclose(raw,table.raw_prediction,rtol=1e-7,atol=1e-4)
p=np.maximum(raw,0);np.testing.assert_allclose(p,table.prediction,rtol=1e-7,atol=1e-4)
y=frame.loc[valid,'price'].to_numpy();mse=float(np.mean((p-y)**2))
np.testing.assert_allclose(mse,result['mse'],rtol=1e-10,atol=1e-6)
oof=pd.read_csv(local/'selected_training_oof.csv',dtype={'imageid':str})
assert oof.imageid.tolist()==frame.loc[train,'imageid'].tolist()
score=float(np.mean((oof.oof_prediction.to_numpy()-frame.loc[train,'price'].to_numpy())**2))
np.testing.assert_allclose(score,result['training_oof_mse'],rtol=1e-10)
rows=pd.read_csv(public/'training_cv.csv')
np.testing.assert_allclose(rows.training_oof_mse.min(),score,rtol=1e-10)
audit={'passed':True,'n_train_equation':6399,'n_validation_replay':1601,'mse':mse,
    'max_training_equation_error':float(abs(training_prices-expected).max()),
    'max_validation_price_error':float(abs(raw-table.raw_prediction.to_numpy()).max()),
    'head_sha256':digest(local/'head.npz'),'predictions_sha256':digest(local/'predictions.csv'),
    'source_sha256':digest(__file__),'scope':'CPU head bank, fitted transform means/scores, all training equations, all validation prices and selected OOF score. Does not refit all OOF models or reencode GPU images.',
    'export_eligible':False,'test_images_used':False}
write_json(public/'independent_artifact_audit.json',audit);print(audit,flush=True)
fuse(BASE,RUN,'pe_whole_pls_incremental_fusion')
