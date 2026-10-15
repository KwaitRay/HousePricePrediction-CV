"""One preregistered local grid refinement of the existing whole-frame PE-G KRR."""
import argparse,json,time,uuid,os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh,solve
from threadpoolctl import threadpool_limits
from common.local_paths import REPOSITORY,local_path
from common.config import digest,write_json
from common.metrics import save_predictions

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache-key',required=True);parser.add_argument('--name',required=True)
    parser.add_argument('--cache-family',choices=['imagenet_features','clip_features','clip_multiview','blip_features','fused_features','streetclip_features','siglip_features','siglip_multiview','dinov2_features','siglip2_naflex_features','pe_core_features','siglip2_giant_features','pe_core_giant_features','pe_core_giant_whole_features','aimv2_features','eva_clip_features'],default='imagenet_features')
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--original-manifest',type=Path)
    args=parser.parse_args();assert args.cache_family=='pe_core_giant_whole_features' and args.name=='pe_giant_whole_refined_kernel'
    cache=local_path('cache_root')/args.cache_family/args.cache_key
    manifest_path=args.manifest or cache/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    split=local_path('split_path');assert digest(split)==manifest['split_sha256']
    frame=pd.read_csv(split,dtype={'imageid':str,'group_id':str})
    assert frame.imageid.tolist()==manifest['imageids']
    x=np.concatenate([np.load(p,allow_pickle=False) for p in sorted(cache.glob('*.npy'))]).astype(np.float64)
    if args.cache_family in {'clip_multiview','siglip_multiview'}:
        if args.original_manifest is None:raise ValueError('Original feature manifest required')
        assert digest(args.original_manifest)==manifest['original_manifest_sha256']
        original_manifest=json.loads(args.original_manifest.read_text())
        import hashlib
        original_key=hashlib.sha256(json.dumps(original_manifest,sort_keys=True).encode()).hexdigest()
        original_family='siglip_features' if args.cache_family=='siglip_multiview' else 'clip_features'
        original_cache=local_path('cache_root')/original_family/original_key
        original=np.concatenate([np.load(p,allow_pickle=False) for p in sorted(original_cache.glob('*.npy'))]).astype(np.float64)
        assert original_manifest['imageids']==manifest['imageids'] and original.ndim==2
        dimension=original.shape[1];assert x.shape==(8000,2,dimension) and len(original)==8000
        x=np.concatenate([original,x.reshape(8000,dimension*2)],axis=1)
    assert len(x)==8000 and np.isfinite(x).all()
    x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    train=frame.partition.eq('train').to_numpy();valid=frame.partition.eq('validation').to_numpy()
    assert train.sum()==6399 and valid.sum()==1601
    xt=x[train];xv=x[valid];y=frame.loc[train,'price'].to_numpy()/1000
    groups=frame.loc[train,'group_id'].to_numpy()
    from sklearn.model_selection import GroupKFold
    folds=list(GroupKFold(3).split(xt,y,groups))
    gammas=[2.,4.,8.];alphas=[0.1,0.3,1.,3.]
    control_source=local_path('records_root')/'19_image_embedding_reference/pe_core_g14_448_whole_kernel/20261014T090655Z_2d55c743'
    control_oof=pd.read_csv(control_source/'selected_training_oof.csv',dtype={'imageid':str})
    assert control_oof.imageid.tolist()==frame.loc[train,'imageid'].tolist()
    control_passed=False
    run=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
    stage='19_image_embedding_reference' if args.cache_family in {'siglip_features','siglip_multiview','dinov2_features','siglip2_naflex_features','pe_core_features','siglip2_giant_features','pe_core_giant_features','pe_core_giant_whole_features','aimv2_features','eva_clip_features'} else '17_geo_visual_reference' if args.cache_family=='streetclip_features' else '11_pretrained_cnn' if args.cache_family=='imagenet_features' else '14_blip_reference' if args.cache_family=='blip_features' else '15_fixed_ensembles' if args.cache_family=='fused_features' else '13_clip_optimization'
    out=local_path('records_root')/stage/args.name/run;out.mkdir(parents=True)
    public=REPOSITORY/'output/experiment_record'/stage/args.name/run;public.mkdir(parents=True)
    write_json(public/'registration.json',{'method':'mean-centered RBF kernel ridge','gammas':gammas,'alphas':alphas,'train_only_group_cv':3,
        'grid_refinement_only':True,'direct_control_gamma':4.,'direct_control_alpha':1.,
        'control_source':'19_image_embedding_reference/pe_core_g14_448_whole_kernel/20261014T090655Z_2d55c743',
        'control_oof_sha256':digest(control_source/'selected_training_oof.csv'),
        'reference_implementation_sha256':digest(Path(__file__).with_name('frozen_kernel_search.py')),
        'hypothesis':'Original training grid has nearly equal gamma4 alpha0.1/1 scores; a fixed local grid may reduce coarse-grid readout error without changing representations or targets',
        'automatic_test_export':False,
        'feature_cache_key':args.cache_key,'manifest_sha256':digest(manifest_path),'source_sha256':digest(__file__),
        'original_manifest_sha256':digest(args.original_manifest) if args.original_manifest else None,'feature_dim':int(x.shape[1]),
        'cache_family':args.cache_family,'course_scope':'General-purpose frozen image tower only; no MLLM or price-task model' if stage=='19_image_embedding_reference' else 'StreetCLIP vision-only reference, course eligibility unresolved' if stage=='17_geo_visual_reference' else 'CLIP/BLIP fusion exploratory; eligibility unresolved' if stage=='15_fixed_ensembles' else 'BLIP vision-only exploratory; eligibility unresolved' if stage=='14_blip_reference' else 'CLIP exploratory; eligibility unresolved' if stage=='13_clip_optimization' else 'ImageNet feature reference',
        'split_sha256':digest(split),'dtype':'float64','centering':'Only training fold target mean, no validation label use',
        'selection':'Original-price unweighted clipped MSE over all training OOF images; validation evaluated once after selection'})
    def status(state,**kw):write_json(public/'status.json',{'status':state,'pid':os.getpid(),**kw})
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    status('running');start=time.perf_counter()
    try:
        distances=np.maximum(2.-2.*(xt@xt.T),0)
        results=[];best=None
        with threadpool_limits(limits=2):
            for gamma in gammas:
                kernel=np.exp(-gamma*distances);oof={a:np.full(len(y),np.nan) for a in alphas}
                for n,(fit,held) in enumerate(folds):
                    assert not set(groups[fit])&set(groups[held])
                    status('fitting',gamma=gamma,fold=n+1)
                    eigen,vectors=eigh(kernel[np.ix_(fit,fit)],check_finite=False,overwrite_a=True)
                    projected=vectors.T@(y[fit]-y[fit].mean());held_vectors=kernel[np.ix_(held,fit)]@vectors
                    for alpha in alphas:oof[alpha][held]=held_vectors@(projected/(np.maximum(eigen,0)+alpha))+y[fit].mean()
                    print(f'{args.name}: gamma={gamma} fold={n+1}/3',flush=True)
                for alpha in alphas:
                    if gamma==4. and alpha==1.:
                        np.testing.assert_allclose(oof[alpha]*1000,control_oof.raw_oof_prediction.to_numpy(),rtol=1e-7,atol=1e-4)
                        control_passed=True
                        write_json(public/'direct_control.json',{'passed':True,'n_training_oof':6399,'maximum_raw_oof_error':float(np.max(np.abs(oof[alpha]*1000-control_oof.raw_oof_prediction.to_numpy()))),'scope':'Same grouped folds, representation and original gamma4 alpha1; complete training OOF replay'})
                    pred=np.maximum(oof[alpha],0);score=float(np.mean((pred-y)**2)*1e6)
                    row={'gamma':gamma,'alpha':alpha,'training_oof_mse':score}
                    results.append(row);print(json.dumps(row),flush=True)
                    pd.DataFrame(results).to_csv(public/'training_cv.csv',index=False)
                    if best is None or score<best['training_oof_mse']:
                        best=row.copy();best_oof=oof[alpha].copy()
            assert control_passed
            gamma=best['gamma'];alpha=best['alpha'];mean=float(y.mean())
            coefficients=solve(np.exp(-gamma*distances)+alpha*np.eye(len(y)),y-mean,assume_a='pos',check_finite=False)
            vd=np.maximum(2.-2.*(xv@xt.T),0);pred=(np.exp(-gamma*vd)@coefficients+mean)*1000
        metrics=save_predictions(out,frame.loc[valid],pred,1300)
        np.savez_compressed(out/'head.npz',train_features=xt,coefficients=coefficients,gamma=gamma,alpha=alpha,target_mean=mean)
        write_json(public/'results.json',{**best,**metrics,'seconds':time.perf_counter()-start})
        pd.DataFrame({'imageid':frame.loc[train,'imageid'],'raw_oof_prediction':best_oof*1000,'oof_prediction':np.maximum(best_oof,0)*1000}).to_csv(out/'selected_training_oof.csv',index=False)
        write_json(out/'selected_oof_protocol.json',{'gamma':gamma,'alpha':alpha,'caveat':'Selected using these training folds; not nested-CV performance'})
        print(json.dumps({**best,**metrics}),flush=True);status('completed')
    except Exception as exc:status('failed',error=repr(exc));raise

if __name__=='__main__':main()
