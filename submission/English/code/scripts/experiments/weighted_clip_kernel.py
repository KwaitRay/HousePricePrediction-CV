"""Mild training-price loss weights on fixed CLIP L/14 features; unweighted evaluation."""
import json,time,uuid,os,argparse
from pathlib import Path
import numpy as np,pandas as pd
from scipy.linalg import eigh,solve
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from common.local_paths import local_path,REPOSITORY
from common.config import digest,write_json
from common.metrics import save_predictions

def weights(prices):
    band=np.digitize(prices,[300,700,1300],right=True);counts=np.bincount(band,minlength=4)
    w=np.sqrt(len(prices)/(4*np.maximum(counts[band],1)));w=np.clip(w,0.8,1.25);return w/w.mean()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference-kernel',type=Path);parser.add_argument('--name',default='clip_l14_weighted_kernel');args=parser.parse_args()
    split=local_path('split_path');frame=pd.read_csv(split,dtype={'imageid':str,'group_id':str})
    train=frame.partition.eq('train').to_numpy();valid=frame.partition.eq('validation').to_numpy()
    gamma=4.;source=None;source_sha=None
    if args.reference_kernel:
        from visual_kernel_submission import features_for
        reference=args.reference_kernel.resolve();source=str(reference.relative_to(local_path('records_root')))
        public_reference=REPOSITORY/'output/experiment_record'/source
        assert json.loads((public_reference/'status.json').read_text(encoding='utf-8'))['status']=='completed'
        registration=json.loads((public_reference/'registration.json').read_text(encoding='utf-8'))
        x=features_for(registration,frame)
        source_sha=digest(reference/'head.npz')
        with np.load(reference/'head.npz',allow_pickle=False) as previous:
            assert 'training_weights' not in previous.files
            np.testing.assert_allclose(x[train],previous['train_features'],rtol=0,atol=1e-12)
            gamma=float(previous['gamma'])
        feature_manifest_sha=registration['manifest_sha256']
    else:
        cache=local_path('cache_root')/'clip_features/6cc6ea162d96836a66ec8b9b53757f40f2853e0029fdfd1dcf48ef23f98c3529'
        manifest=json.loads((cache/'manifest.json').read_text());assert digest(split)==manifest['split_sha256'] and frame.imageid.tolist()==manifest['imageids']
        x=np.concatenate([np.load(p) for p in sorted(cache.glob('*.npy'))]).astype(np.float64);assert x.shape==(8000,768)
        x/=np.linalg.norm(x,axis=1,keepdims=True)
        feature_manifest_sha=digest(cache/'manifest.json')
    xt=x[train];xv=x[valid];prices=frame.loc[train,'price'].to_numpy();y=prices/1000;groups=frame.loc[train,'group_id'].to_numpy()
    assert train.sum()==6399 and valid.sum()==1601
    alphas=[0.01,0.1,1.,10.]
    run=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
    stage='19_image_embedding_reference' if args.reference_kernel and registration['cache_family'].startswith('siglip') else '13_clip_optimization'
    out=local_path('records_root')/stage/args.name/run;out.mkdir(parents=True)
    public=REPOSITORY/'output/experiment_record'/stage/args.name/run;public.mkdir(parents=True)
    write_json(public/'registration.json',{'gamma':gamma,'gamma_source':'previous train-only CV-selected reference kernel','alphas':alphas,
        'source_kernel':source,'source_kernel_sha256':source_sha,'feature_dim':int(x.shape[1]),
        'weights':'Four price bands; sqrt(N/(4*n_band)), clip 0.8..1.25, normalize training mean1; recompute independently per fit fold',
        'center':'Fit-fold weighted target mean','evaluation':'Unweighted original-price clipped MSE, train-only GroupKFold3 selects alpha; validation once',
        'source_sha256':digest(__file__),'split_sha256':digest(split),'feature_manifest_sha256':feature_manifest_sha,
        'test_images_used':False,'course_scope':registration['course_scope'] if args.reference_kernel else 'Exploratory CLIP, eligibility unresolved'})
    def status(s,**kw):write_json(public/'status.json',{'status':s,'pid':os.getpid(),**kw})
    status('fitting');start=time.perf_counter();fold_stats=[]
    try:
        with threadpool_limits(limits=4):
            k=np.exp(-gamma*np.maximum(2-2*xt@xt.T,0));oof={a:np.full(len(y),np.nan) for a in alphas}
            for n,(fit,held) in enumerate(GroupKFold(3).split(xt,y,groups)):
                assert not set(groups[fit])&set(groups[held]);w=weights(prices[fit]);s=np.sqrt(w);mean=float(np.average(y[fit],weights=w))
                fold_stats.append({'fold':n,'mean_weight':float(w.mean()),'min_weight':float(w.min()),'max_weight':float(w.max())})
                eigen,v=eigh(k[np.ix_(fit,fit)]*s[:,None]*s[None,:],check_finite=False)
                projection=v.T@(s*(y[fit]-mean));hv=k[np.ix_(held,fit)]@(s[:,None]*v)
                for alpha in alphas:oof[alpha][held]=hv@(projection/(np.maximum(eigen,0)+alpha))+mean
                print(f'{args.name}: weighted kernel fold {n+1}/3',flush=True);write_json(public/'fold_weights.json',fold_stats)
            rows=[{'alpha':a,'training_oof_mse':float(np.mean((np.maximum(oof[a],0)-y)**2)*1e6)} for a in alphas]
            pd.DataFrame(rows).to_csv(public/'training_cv.csv',index=False);best=min(rows,key=lambda r:r['training_oof_mse']);alpha=best['alpha']
            w=weights(prices);mean=float(np.average(y,weights=w));coef=solve(k+alpha*np.diag(1/w),y-mean,assume_a='pos',check_finite=False)
            raw=(np.exp(-gamma*np.maximum(2-2*xv@xt.T,0))@coef+mean)*1000
        metrics=save_predictions(out,frame.loc[valid],raw,1300)
        np.savez_compressed(out/'head.npz',train_features=xt,coefficients=coef,gamma=gamma,alpha=alpha,target_mean=mean,training_weights=w)
        pd.DataFrame({'imageid':frame.loc[train,'imageid'],'oof_prediction':np.maximum(oof[alpha],0)*1000}).to_csv(out/'selected_training_oof.csv',index=False)
        write_json(public/'results.json',{'head':'weighted_kernel',**best,**metrics,'seconds':time.perf_counter()-start})
        price=frame.loc[valid,'price'].to_numpy();pred=np.maximum(raw,0);slices=[]
        for name,mask in [('low',price<=300),('middle',(price>300)&(price<=1300)),('high',price>1300)]:
            error=pred[mask]-price[mask];slices.append({'tier':name,'n':int(mask.sum()),'mse':float(np.mean(error**2)),'mae':float(np.mean(abs(error))),'bias':float(np.mean(error)),'mean_positive_overestimate':float(np.maximum(error,0).mean())})
        write_json(public/'price_slices.json',slices);print(json.dumps({**best,**metrics}),flush=True);status('completed')
    except Exception as exc:status('failed',error=repr(exc));raise

if __name__=='__main__':main()
