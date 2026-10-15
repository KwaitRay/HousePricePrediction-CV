"""Audit all PCA bank/equation/validation prices and the label-blind bandwidth rule."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from common.config import read_json,write_json,digest
from common.local_paths import local_path
from visual_kernel_submission import features_for
from pe_pca_euclidean_incremental_fusion import main as fuse
RUN='19_image_embedding_reference/pe_giant_whole_pca_euclidean_kernel/20261015T034629Z_226d870e'
BASE='15_fixed_ensembles/pe_whole_pca_incremental_fusion/20261015T033810Z_8aa6ec62'
root=local_path('records_root');out=root/RUN;pub=ROOT/'output/experiment_record'/RUN
assert read_json(pub/'status.json')['status']=='completed'
reg=read_json(pub/'registration.json');result=read_json(pub/'results.json')
assert digest(ROOT/'scripts/experiments/pe_whole_pca_euclidean_kernel.py')==reg['source_sha256']==digest(out/'source.py')
assert digest(local_path('split_path'))==reg['split_sha256']
assert digest(root/reg['source_kernel']/'head.npz')==reg['source_head_sha256']
frame=pd.read_csv(local_path('split_path'),dtype={'imageid':str,'group_id':str})
train=frame.partition.eq('train').to_numpy();valid=frame.partition.eq('validation').to_numpy()
assert train.sum()==6399 and valid.sum()==1601 and not set(frame.loc[train,'group_id'])&set(frame.loc[valid,'group_id'])
source_file=ROOT/'output/experiment_record'/reg['source_kernel']/'registration.json'
assert digest(source_file)==reg['source_registration_sha256']
x=features_for(read_json(source_file),frame)
with np.load(out/'head.npz',allow_pickle=False) as h:head={k:h[k].copy() for k in h.files}
with np.load(out/'feature_transform.npz',allow_pickle=False) as h:t={k:h[k].copy() for k in h.files}
dim=result['components'];power=float(t['whitening_power'])
assert t['vectors'].shape==(2560,dim) and power==result['whitening_power']
np.testing.assert_allclose(t['mean'],x[train].mean(axis=0),rtol=1e-10,atol=1e-12)
with threadpool_limits(limits=2):
    np.testing.assert_allclose(t['vectors'].T@t['vectors'],np.eye(dim),rtol=1e-8,atol=1e-10)
    centered=x[train]-t['mean']
    eigen_residual=centered.T@(centered@t['vectors'])/(6399-1)-t['vectors']*t['variance']
    assert np.max(abs(eigen_residual))<1e-10
    z=(x-t['mean'])@t['vectors']
    z/=np.maximum(t['variance'],t['variance'][0]*1e-8)**(power/2)
    assert np.isfinite(z).all()
np.testing.assert_allclose(head['train_features'],z[train],rtol=1e-10,atol=1e-12)
pairs=np.random.default_rng(2026).integers(0,6399,size=(8192,2));pairs=pairs[pairs[:,0]!=pairs[:,1]]
med=[]
for values in [x[train],z[train]]:
    med.append(float(np.median(np.sum((values[pairs[:,0]]-values[pairs[:,1]])**2,axis=1))))
gamma=4*med[0]/med[1]
np.testing.assert_allclose(gamma,float(head['gamma']),rtol=1e-12)
assert float(head['alpha'])==1
np.testing.assert_allclose(float(head['target_mean']),frame.loc[train,'price'].mean()/1000,rtol=1e-12)
def replay(query):
    prices=[]
    for i in range(0,len(query),128):
        k=np.exp(-gamma*np.maximum(np.sum(query[i:i+128]**2,axis=1)[:,None]+np.sum(head['train_features']**2,axis=1)[None,:]-2*(query[i:i+128]@head['train_features'].T),0))
        prices.extend(((k@head['coefficients']+float(head['target_mean']))*1000).tolist())
    return np.asarray(prices)
with threadpool_limits(limits=2):a=replay(z[train]);raw=replay(z[valid])
expected=frame.loc[train,'price'].to_numpy()-1000*head['coefficients']
np.testing.assert_allclose(a,expected,rtol=1e-7,atol=1e-4)
saved=pd.read_csv(out/'predictions.csv',dtype={'imageid':str,'group_id':str})
assert saved.imageid.tolist()==frame.loc[valid,'imageid'].tolist() and saved.group_id.tolist()==frame.loc[valid,'group_id'].tolist()
np.testing.assert_array_equal(saved.price,frame.loc[valid,'price'])
np.testing.assert_allclose(raw,saved.raw_prediction,rtol=1e-7,atol=1e-4)
np.testing.assert_allclose(np.maximum(raw,0),saved.prediction,rtol=1e-7,atol=1e-4)
np.testing.assert_allclose(np.mean((np.maximum(raw,0)-saved.price.to_numpy())**2),result['mse'],rtol=1e-10)
oof=pd.read_csv(out/'selected_training_oof.csv',dtype={'imageid':str})
assert oof.imageid.tolist()==frame.loc[train,'imageid'].tolist()
np.testing.assert_allclose(np.mean((oof.oof_prediction.to_numpy()-frame.loc[train,'price'].to_numpy())**2),result['training_oof_mse'],rtol=1e-10)
write_json(pub/'independent_artifact_audit.json',{'passed':True,'source_sha256':digest(__file__),
    'head_sha256':digest(out/'head.npz'),'predictions_sha256':digest(out/'predictions.csv'),
    'transform_sha256':digest(out/'feature_transform.npz'),'n_train_equations':6399,'n_validation':1601,
    'eigen_residual_max':float(abs(eigen_residual).max()),'max_training_price_error':float(abs(a-expected).max()),
    'max_validation_price_error':float(abs(raw-saved.raw_prediction.to_numpy()).max()),
    'median_calibrated_gamma':gamma,'scope':'CPU all fitted PCA eigenpairs/means, transformed training bank, bandwidth, all6399 equations/all1601 prices/selected OOF score. Does not refit every OOF model or reencode images.',
    'test_images_used':False,'export_eligible':False})
fuse(BASE,RUN,'pe_whole_pca_euclidean_incremental_fusion')
