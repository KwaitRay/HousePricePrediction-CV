"""Fold-fitted PCA/partial whitening geometry for verified frozen PE-G features."""
import os,time,uuid
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh,solve
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from common.config import read_json,write_json,digest
from common.local_paths import local_path,REPOSITORY
from common.metrics import save_predictions
from visual_kernel_submission import features_for
SOURCE='19_image_embedding_reference/pe_core_g14_448_whole_kernel/20261014T090655Z_2d55c743'
BASE='15_fixed_ensembles/pe_giant_whole_incremental_fusion/20261014T082727Z_c5152cef'

def transform(x,mean,vectors,variance,dim,power):
    z=(x-mean)@vectors[:,:dim]
    z/=np.maximum(variance[:dim],variance[0]*1e-8)**(power/2)
    norms=np.linalg.norm(z,axis=1,keepdims=True)
    assert np.isfinite(z).all() and (norms>1e-12).all()
    return z

def eigensystem(x):
    mean=x.mean(axis=0);z=x-mean
    cov=z.T@z/(len(z)-1)
    values,vectors=eigh(cov,subset_by_index=[cov.shape[0]-1024,cov.shape[0]-1],check_finite=False,overwrite_a=True)
    return mean,vectors[:,::-1].copy(),np.maximum(values[::-1],0)

def prices(query,bank,c,mean,gamma=4.):
    result=[]
    for i in range(0,len(query),128):
        result.extend(((np.exp(-gamma*np.maximum(np.sum(query[i:i+128]**2,axis=1)[:,None]+np.sum(bank**2,axis=1)[None,:]-2*(query[i:i+128]@bank.T),0))@c+mean)*1000).tolist())
    return np.asarray(result)

def fit_head(bank,y,gamma=4.):
    k=np.exp(-gamma*np.maximum(np.sum(bank**2,axis=1)[:,None]+np.sum(bank**2,axis=1)[None,:]-2*(bank@bank.T),0));k.flat[::len(bank)+1]+=1
    return solve(k,y-y.mean(),assume_a='pos',check_finite=False,overwrite_a=True),float(y.mean())

def calibrate_gamma(original, transformed):
    rng=np.random.default_rng(2026)
    pairs=rng.integers(0,len(original),size=(8192,2))
    pairs=pairs[pairs[:,0]!=pairs[:,1]]
    def median_distance(z):
        return float(np.median(np.sum((z[pairs[:,0]]-z[pairs[:,1]])**2,axis=1)))
    old,new=median_distance(original),median_distance(transformed)
    assert old>1e-12 and new>1e-12
    return 4.*old/new,old,new

def main():
    root=local_path('records_root');source_pub=REPOSITORY/'output/experiment_record'/SOURCE
    source_reg=read_json(source_pub/'registration.json')
    relative=Path('19_image_embedding_reference/pe_giant_whole_pca_euclidean_kernel')/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
    out=root/relative;pub=REPOSITORY/'output/experiment_record'/relative;out.mkdir(parents=True,exist_ok=False)
    registration={'method':'Fold-fitted radial-preserving PCA Euclidean RBF with training-median bandwidth calibration','source_kernel':SOURCE,'baseline_fusion':BASE,
        'source_sha256':digest(__file__),'feature_reader_sha256':digest(Path(__file__).with_name('visual_kernel_submission.py')),
        'split_sha256':digest(local_path('split_path')),'source_registration_sha256':digest(source_pub/'registration.json'),
        'source_head_sha256':digest(root/SOURCE/'head.npz'),'dimensions':[64,256,1024],'whitening_powers':[0.,.5],
        'base_gamma':4.,'gamma_rule':'4*median(original fit-fold squared distances)/median(transformed fit-fold squared distances), fixed 8192 sampled distinct-row pairs seed2026, no prices or held-out rows','alpha':1.,'train_only_group_cv':3,'eigenvalue_floor_relative':1e-8,
        'hypothesis':'Post-PCA per-image unit normalization discards radial information. Keep projected feature radii and use true squared Euclidean distances; preserve the PCA grid and training-median bandwidth rule, with no new labels or encoder training.',
        'formula':'(x-fit_mean)@fit_top_eigenvectors / max(fit_variance,fit_max_variance*1e-8)^(power/2); no per-image unit normalization, squared Euclidean kernel distances',
        'null_control':'Original untransformed kernel; independently reproduce all6399 raw OOF and all1601 validation before PCA candidates',
        'selection':'Minimum original-price training OOF MSE across null and six fixed transforms; full validation once after selection',
        'fit_scope':'PCA mean/eigenvectors/variances only fitting-fold frozen features, no held-out features or prices in PCA fit',
        'caveat':'Source hyperparameters already selected, new OOF selection is not nested unbiased CV',
        'cache_family':'pe_giant_whole_pca_euclidean_features','feature_dim':2560,'no_transformed_cache_created':True,
        'full_refit_adapter_supported':False,'automatic_test_export':False,'test_images_used':False,'supervised_encoder':False}
    write_json(pub/'registration.json',registration);(out/'source.py').write_bytes(Path(__file__).read_bytes());print(str(relative),flush=True)
    def status(s,**kw):write_json(pub/'status.json',{'status':s,'pid':os.getpid(),**kw})
    began=time.perf_counter()
    try:
        status('checking_control')
        frame=pd.read_csv(local_path('split_path'),dtype={'imageid':str,'group_id':str})
        assert digest(local_path('split_path'))==registration['split_sha256']
        train=frame.partition.eq('train').to_numpy();valid=frame.partition.eq('validation').to_numpy()
        assert train.sum()==6399 and valid.sum()==1601 and not set(frame.loc[train,'group_id'])&set(frame.loc[valid,'group_id'])
        x=features_for(source_reg,frame);xt=x[train];xv=x[valid];y=frame.loc[train,'price'].to_numpy()/1000
        groups=frame.loc[train,'group_id'].to_numpy();folds=list(GroupKFold(3).split(xt,y,groups))
        reference=pd.read_csv(root/SOURCE/'selected_training_oof.csv',dtype={'imageid':str});vr=pd.read_csv(root/SOURCE/'predictions.csv',dtype={'imageid':str})
        assert reference.imageid.tolist()==frame.loc[train,'imageid'].tolist() and vr.imageid.tolist()==frame.loc[valid,'imageid'].tolist()
        null=np.empty(6399)
        settings=[(d,p) for d in registration['dimensions'] for p in registration['whitening_powers']]
        oofs={s:np.full(6399,np.nan) for s in settings};calibrations=[]
        with threadpool_limits(limits=2):
            for fit,held in folds:
                c,mean=fit_head(xt[fit],y[fit]);null[held]=prices(xt[held],xt[fit],c,mean)
            np.testing.assert_allclose(null,reference.raw_oof_prediction,rtol=1e-7,atol=1e-4)
            c,mean=fit_head(xt,y);control=prices(xv,xt,c,mean)
            np.testing.assert_allclose(control,vr.raw_prediction,rtol=1e-7,atol=1e-4)
            write_json(pub/'direct_control.json',{'passed':True,'oof_n':6399,'validation_n':1601,
                'maximum_oof_price_error':float(abs(null-reference.raw_oof_prediction.to_numpy()).max()),
                'maximum_validation_price_error':float(abs(control-vr.raw_prediction.to_numpy()).max())})
            for n,(fit,held) in enumerate(folds):
                assert not set(groups[fit])&set(groups[held])
                status('fitting_pca',fold=n+1)
                mean,vectors,variance=eigensystem(xt[fit])
                for dim,power in settings:
                    status('fitting_head',fold=n+1,components=dim,whitening_power=power)
                    a=transform(xt[fit],mean,vectors,variance,dim,power);b=transform(xt[held],mean,vectors,variance,dim,power)
                    gamma,old_median,new_median=calibrate_gamma(xt[fit],a)
                    calibrations.append({'fold':n+1,'components':dim,'whitening_power':power,'gamma':gamma,'original_median_distance':old_median,'transformed_median_distance':new_median})
                    write_json(pub/'bandwidth_calibration.json',calibrations)
                    c,prior=fit_head(a,y[fit],gamma);oofs[(dim,power)][held]=prices(b,a,c,prior,gamma)
                    print({'fold':n+1,'components':dim,'whitening_power':power},flush=True)
            best={'components':0,'whitening_power':0.,'training_oof_mse':float(np.mean((np.maximum(null,0)-y*1000)**2))}
            best_oof=null.copy();rows=[best.copy()]
            for (dim,power),oof in oofs.items():
                assert np.isfinite(oof).all()
                row={'components':dim,'whitening_power':power,'training_oof_mse':float(np.mean((np.maximum(oof,0)-y*1000)**2))}
                rows.append(row)
                if row['training_oof_mse']<best['training_oof_mse']:best=row.copy();best_oof=oof.copy()
            pd.DataFrame(rows).to_csv(pub/'training_cv.csv',index=False)
            dim,power=best['components'],best['whitening_power']
            if dim:
                status('refitting_pca',components=dim,whitening_power=power)
                mean,vectors,variance=eigensystem(xt)
                bank=transform(xt,mean,vectors,variance,dim,power);query=transform(xv,mean,vectors,variance,dim,power)
                np.savez_compressed(out/'feature_transform.npz',mean=mean,vectors=vectors[:,:dim],variance=variance[:dim],whitening_power=power)
            else:bank=xt;query=xv
            gamma,old_median,new_median=calibrate_gamma(xt,bank) if dim else (4.,None,None)
            c,prior=fit_head(bank,y,gamma);prediction=prices(query,bank,c,prior,gamma)
        metrics=save_predictions(out,frame.loc[valid],prediction,1300)
        np.savez_compressed(out/'head.npz',train_features=bank,coefficients=c,gamma=gamma,alpha=1.,target_mean=prior,distance_type='squared_euclidean')
        pd.DataFrame({'imageid':frame.loc[train,'imageid'],'raw_oof_prediction':best_oof,'oof_prediction':np.maximum(best_oof,0)}).to_csv(out/'selected_training_oof.csv',index=False)
        result={**best,**metrics,'gamma':gamma,'full_training_original_median_distance':old_median,'full_training_transformed_median_distance':new_median,'seconds':time.perf_counter()-began,'export_eligible':False,'full_refit_adapter_supported':False,'test_images_used':False}
        write_json(pub/'results.json',result);status('completed');print(result,flush=True)
    except BaseException as exc:status('failed',error=repr(exc));raise

if __name__=='__main__':main()
