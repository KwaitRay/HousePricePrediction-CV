"""Explicit replay/full-label refit for the frozen fold-selected PCA kernel recipe."""
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from common.config import digest,read_json
from pe_whole_pca_bandwidth_kernel import eigensystem,calibrate_gamma,fit_head,prices

METHOD='Fold-fitted PCA partial-whitening image RBF with training-median bandwidth calibration'

def apply_transform(values,transform):
    z=(values-transform['mean'])@transform['vectors']
    z/=np.maximum(transform['variance'],transform['variance'][0]*1e-8)**(float(transform['whitening_power'])/2)
    norms=np.linalg.norm(z,axis=1,keepdims=True)
    if not np.isfinite(z).all() or (norms<=1e-12).any():raise ValueError('Invalid PCA transform')
    return z/norms

def replay_and_refit(source,reg,values,train,valid,y,path,reference_raw):
    if reg['method']!=METHOD:raise ValueError('Not the registered median-calibrated PCA protocol')
    with threadpool_limits(limits=2):
        with np.load(source/'feature_transform.npz',allow_pickle=False) as f:t={k:f[k].copy() for k in f.files}
        dim=t['vectors'].shape[1];power=float(t['whitening_power'])
        if dim not in (64,256,1024) or power not in (0.,.5):raise ValueError('Unexpected PCA recipe')
        old=apply_transform(values,t)
        with np.load(source/'head.npz',allow_pickle=False) as h:
            np.testing.assert_allclose(old[train],h['train_features'],rtol=0,atol=1e-12)
            expected_gamma,_,_=calibrate_gamma(values[train],old[train])
            np.testing.assert_allclose(expected_gamma,float(h['gamma']),rtol=1e-12)
            assert float(h['alpha'])==1
            raw=prices(old[valid],h['train_features'],h['coefficients'],float(h['target_mean']),float(h['gamma']))
        np.testing.assert_allclose(raw,reference_raw,rtol=1e-7,atol=1e-4)
        mean,vectors,variance=eigensystem(values)
        full={'mean':mean,'vectors':vectors[:,:dim],'variance':variance[:dim],'whitening_power':power}
        bank=apply_transform(values,full);gamma,original_median,transformed_median=calibrate_gamma(values,bank)
        c,prior=fit_head(bank,y,gamma)
        np.savez_compressed(path,train_features=bank,coefficients=c,gamma=gamma,alpha=1.,target_mean=prior)
        transform_path=path.with_name(path.stem.replace('_head','_pca')+'.npz')
        np.savez_compressed(transform_path,**full)
        equation=prices(bank,bank,c,prior,gamma)
        np.testing.assert_allclose(equation,1000*(y-c),rtol=1e-7,atol=1e-4)
    return raw,{'kernel_type':'pca_image_rbf','feature_dim':values.shape[1],'head_feature_dim':dim,
        'full_head_sha256':digest(path),'full_pca_file':transform_path.name,'full_pca_sha256':digest(transform_path),
        'reference_pca_sha256':digest(source/'feature_transform.npz'),'pca_components':dim,'whitening_power':power,
        'gamma':gamma,'alpha':1.,'gamma_rule':reg['gamma_rule'],
        'full_training_original_median_distance':original_median,'full_training_transformed_median_distance':transformed_median,
        'all8000_training_equation_max_price_error':float(abs(equation-1000*(y-c)).max()),
        'pca_adapter_sha256':digest(__file__),'pca_recipe_source_sha256':digest(Path(__file__).with_name('pe_whole_pca_bandwidth_kernel.py'))}
