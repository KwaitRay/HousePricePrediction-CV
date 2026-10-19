"""Train-only choice of crop/whole geometry weights on verified frozen PE features."""
import os,time,uuid
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from common.config import read_json,write_json,digest
from common.local_paths import local_path,REPOSITORY
from common.metrics import save_predictions
from visual_kernel_submission import features_for

SOURCE='19_image_embedding_reference/pe_core_g14_448_whole_kernel/20261014T090655Z_2d55c743'
BASELINE='15_fixed_ensembles/pe_giant_whole_incremental_fusion/20261014T082727Z_c5152cef'

def main():
    root=local_path('records_root')
    source_public=REPOSITORY/'output/experiment_record'/SOURCE
    source_reg=read_json(source_public/'registration.json')
    relative=Path('19_image_embedding_reference/pe_giant_whole_view_weight_kernel')/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
    out=root/relative;public=REPOSITORY/'output/experiment_record'/relative
    out.mkdir(parents=True,exist_ok=False)
    reg={'method':'mean-centered weighted-view RBF kernel ridge','source_kernel':SOURCE,'baseline_fusion':BASELINE,
        'cache_family':'pe_giant_whole_view_weight_features','source_feature_registration_sha256':digest(source_public/'registration.json'),
        'source_sha256':digest(__file__),'feature_reader_sha256':digest(Path(__file__).with_name('visual_kernel_submission.py')),
        'split_sha256':digest(local_path('split_path')),'whole_weights':[.5,.25,.75,1.,0.],
        'gamma':4.,'alpha':1.,'train_only_group_cv':3,'feature_dim':2560,
        'hypothesis':'Equal crop/whole weighting is an untested convention; tune only its geometry weight, preserving the verified encoder, readout gamma/alpha, labels and grouped folds.',
        'formula':'concat(sqrt(1-w)*unit(crop),sqrt(w)*unit(whole)), followed by unit normalization',
        'control':'w=0.5 replay all6399 source raw OOF and all1601 source validation prices before evaluating other weights',
        'selection':'Minimum clipped original-price training OOF MSE; fixed order breaks ties. Validation only for selected weight.',
        'fusion_gate':'Later separate incremental OOF fusion requires >=1% metaCV improvement, >=2/3 paired wins and improved full validation.',
        'test_images_used':False,'supervised_encoder':False,'full_refit_adapter_supported':False,'automatic_test_export':False}
    write_json(public/'registration.json',reg)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    print(str(relative),flush=True)
    def status(state,**kw):write_json(public/'status.json',{'status':state,'pid':os.getpid(),**kw})
    began=time.perf_counter()
    try:
        status('checking_features')
        frame=pd.read_csv(local_path('split_path'),dtype={'imageid':str,'group_id':str})
        assert digest(local_path('split_path'))==reg['split_sha256']
        x=features_for(source_reg,frame)
        assert x.shape==(8000,2560) and np.isfinite(x).all()
        train=frame.partition.eq('train').to_numpy();valid=frame.partition.eq('validation').to_numpy()
        assert train.sum()==6399 and valid.sum()==1601
        assert not set(frame.loc[train,'group_id'])&set(frame.loc[valid,'group_id'])
        crop,whole=[z/np.linalg.norm(z,axis=1,keepdims=True) for z in np.split(x,2,axis=1)]
        y=frame.loc[train,'price'].to_numpy()/1000;groups=frame.loc[train,'group_id'].to_numpy()
        folds=list(GroupKFold(3).split(x[train],y,groups))
        reference=pd.read_csv(root/SOURCE/'selected_training_oof.csv',dtype={'imageid':str})
        valref=pd.read_csv(root/SOURCE/'predictions.csv',dtype={'imageid':str})
        assert reference.imageid.tolist()==frame.loc[train,'imageid'].tolist()
        assert valref.imageid.tolist()==frame.loc[valid,'imageid'].tolist()
        results=[];best=None
        with threadpool_limits(limits=2):
            for weight in reg['whole_weights']:
                z=np.concatenate([np.sqrt(1-weight)*crop,np.sqrt(weight)*whole],axis=1)
                z/=np.linalg.norm(z,axis=1,keepdims=True)
                xt=z[train]
                k=np.exp(-4*np.maximum(2-2*xt@xt.T,0))
                oof=np.full(6399,np.nan)
                for n,(fit,held) in enumerate(folds):
                    status('fitting',whole_weight=weight,fold=n+1)
                    assert not set(groups[fit])&set(groups[held])
                    matrix=k[np.ix_(fit,fit)].copy();matrix.flat[::len(fit)+1]+=1
                    c=solve(matrix,y[fit]-y[fit].mean(),assume_a='pos',check_finite=False,overwrite_a=True)
                    oof[held]=k[np.ix_(held,fit)]@c+y[fit].mean()
                    print({'weight':weight,'fold':n+1},flush=True)
                assert np.isfinite(oof).all()
                if weight==.5:
                    np.testing.assert_allclose(oof*1000,reference.raw_oof_prediction,rtol=1e-7,atol=1e-4)
                    matrix=k.copy();matrix.flat[::6400]+=1
                    c=solve(matrix,y-y.mean(),assume_a='pos',check_finite=False,overwrite_a=True)
                    pred=(np.exp(-4*np.maximum(2-2*z[valid]@xt.T,0))@c+y.mean())*1000
                    np.testing.assert_allclose(np.maximum(pred,0),valref.prediction,rtol=1e-7,atol=1e-4)
                    write_json(public/'direct_control.json',{'passed':True,'all6399_raw_oof_max_error':float(abs(oof*1000-reference.raw_oof_prediction.to_numpy()).max()),
                        'all1601_validation_max_error':float(abs(np.maximum(pred,0)-valref.prediction.to_numpy()).max())})
                row={'whole_weight':weight,'training_oof_mse':float(np.mean((np.maximum(oof,0)-y)**2)*1e6)}
                results.append(row);write_json(public/'training_cv.json',results);print(row,flush=True)
                if best is None or row['training_oof_mse']<best['training_oof_mse']:
                    best=row.copy();best_oof=oof.copy()
            weight=best['whole_weight']
            z=np.concatenate([np.sqrt(1-weight)*crop,np.sqrt(weight)*whole],axis=1);z/=np.linalg.norm(z,axis=1,keepdims=True)
            xt=z[train];matrix=np.exp(-4*np.maximum(2-2*xt@xt.T,0));matrix.flat[::6400]+=1
            c=solve(matrix,y-y.mean(),assume_a='pos',check_finite=False,overwrite_a=True)
            pred=(np.exp(-4*np.maximum(2-2*z[valid]@xt.T,0))@c+y.mean())*1000
        metrics=save_predictions(out,frame.loc[valid],pred,1300)
        np.savez_compressed(out/'head.npz',train_features=xt,coefficients=c,gamma=4.,alpha=1.,target_mean=y.mean(),whole_weight=weight)
        pd.DataFrame({'imageid':frame.loc[train,'imageid'],'raw_oof_prediction':best_oof*1000,'oof_prediction':np.maximum(best_oof,0)*1000}).to_csv(out/'selected_training_oof.csv',index=False)
        write_json(out/'selected_oof_protocol.json',{'whole_weight':weight,'gamma':4.,'alpha':1.,'caveat':'Selected on these folds, not nested unbiased CV'})
        result={**best,**metrics,'seconds':time.perf_counter()-began,'baseline_single_validation_mse':float(np.mean((valref.prediction-valref.price)**2)),
            'export_eligible':False,'full_refit_adapter_supported':False,'test_images_used':False}
        write_json(public/'results.json',result);status('completed');print(result,flush=True)
    except BaseException as exc:status('failed',error=repr(exc));raise

if __name__=='__main__':main()
