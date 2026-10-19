"""Read-only price-band, split, curve and difficult-image audit; no model selection."""
from continue_training import *
import json
from scipy.stats import ks_2samp, wasserstein_distance
from preparation.split import group_split
from PIL import Image, ImageOps, ImageDraw

OUT = REPOSITORY/'output/experiment_record/05_augmentation/analyses/error_review_20261005'
PRIVATE = local_path('records_root')/'05_augmentation/error_review_20261005'
BINS = [0,300,700,1300,float('inf')]
LABELS = ['≤300','300–700','700–1300','>1300']


def band_metrics(frames):
    rows=[]
    for lo,hi,label in zip(BINS[:-1],BINS[1:],LABELS):
        values=[]
        for f in frames:
            z=f[(f.price>lo)&(f.price<=hi)]
            values.append({'n':len(z),'mse':float(z.squared_error.mean()),'mae':float(z.error.abs().mean()),
                           'bias':float(z.error.mean()),'relative_absolute_error':float((z.error.abs()/z.price).mean()),
                           'contribution':float(z.squared_error.sum()/len(f))})
        rows.append({'band':label,**{k:float(np.mean([v[k] for v in values])) for k in values[0]}})
    return rows


def sheet(frame,name):
    canvas=Image.new('RGB',(4*320,((len(frame)+3)//4)*235),'white'); draw=ImageDraw.Draw(canvas)
    for i,(_,r) in enumerate(frame.iterrows()):
        with Image.open(local_path('data_root')/'train'/r.imageid) as im:
            im=ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'),(316,190))
            x=(i%4)*320;y=(i//4)*235;canvas.paste(im,(x,y))
            text=f"{i+1}: {r.imageid} true={r.price:.0f}"
            if 'mean_prediction' in r:text+=f" pred={r.mean_prediction:.0f}"
            draw.text((x,y+192),text,fill='black')
            if 'mean_squared_error' in r:draw.text((x,y+210),f"mean SE={r.mean_squared_error:.0f}",fill='black')
    canvas.save(PRIVATE/f'{name}.png')
    frame.to_csv(PRIVATE/f'{name}.csv',index=False,encoding='utf-8-sig')


def main():
    OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    split=pd.read_csv(local_path('split_path'));meta=read_json(local_path('split_path').with_suffix('.json'))
    train=split[split.partition=='train']; val=split[split.partition=='validation']
    cfg=resolve(ROOT/'configs/selected/training.json')
    source=pd.read_csv(local_path('data_root')/'train.csv')
    rebuilt=group_split(source,split.set_index('imageid').pixel_hash.reindex(source.imageid).tolist(),cfg['data']['near_pairs'],cfg['data']['split_seed'],cfg['data']['validation_fraction'])
    assert rebuilt.set_index('imageid')[['partition','group_id']].sort_index().equals(split.set_index('imageid')[['partition','group_id']].sort_index())
    distribution=[]
    for lo,hi in zip([0,300,700,1300,2000,3000],[300,700,1300,2000,3000,float('inf')]):
        a=train[(train.price>lo)&(train.price<=hi)];b=val[(val.price>lo)&(val.price<=hi)]
        distribution.append({'band':f'{lo}–{hi}','train_n':len(a),'val_n':len(b),'train_pct':100*len(a)/len(train),'val_pct':100*len(b)/len(val)})
    audit={'distribution':distribution,'split_seed':meta['seed'],'split_hash_verified':digest(local_path('split_path'))==meta['split_sha256'],
           'reconstructed_exactly':True,'groups':split.group_id.nunique(),
           'overlap':{k:len(set(train[k])&set(val[k])) for k in ('imageid','group_id','pixel_hash')},
           'ks_statistic':float(ks_2samp(train.price,val.price).statistic),'wasserstein_price':float(wasserstein_distance(train.price,val.price)),
           'train_summary':train.price.describe(percentiles=[.1,.5,.9,.95,.99]).to_dict(),
           'val_summary':val.price.describe(percentiles=[.1,.5,.9,.95,.99]).to_dict()}
    # Reconstruct group-price rank strata: groups, not individual images, are balanced.
    g=split.groupby('group_id').agg(price=('price','median'),partition=('partition','first')).reset_index().sort_values(['price','group_id'])
    g['stratum']=np.minimum(np.arange(len(g))*10//len(g),9)
    audit['strata']=g.groupby(['stratum','partition']).size().unstack(fill_value=0).reset_index().to_dict('records')
    legacy=REPOSITORY.parent
    old=read_json(legacy/'output/experiment_record/01_preprocessing/pipeline_state_v2.json')['runs']
    controls={s:legacy/'output/experiment_record'/old[f'preprocess_01_stretch:{s}'] for s in SEEDS}
    states=[read_json(local_path('records_root')/f'05_augmentation/{name}/state.json') for name in ('geometry_20261004','robustness_20261004')]
    paths={k:Path(v) for s in states for k,v in s['runs'].items()}
    def predictions(path):
        f=pd.read_csv(path/'predictions.csv').sort_values('imageid').reset_index(drop=True)
        assert set(f.imageid)==set(val.imageid)
        return f
    base={s:predictions(p) for s,p in controls.items()}
    rows=[];curves=[]
    for key,path in paths.items():
        c=read_json(path/'config.json');f=predictions(path);seed=c['seed'];bands=band_metrics([f]);ref=band_metrics([base[seed]])
        for a,b in zip(bands,ref):a['delta_mse']=a['mse']-b['mse'];a['delta_contribution']=a['contribution']-b['contribution']
        rows.append({'experiment':c['experiment']['id'],'seed':seed,'mse':float(f.squared_error.mean()),'bands':bands})
        h=pd.read_csv(path/'history.csv');best=h.loc[h.val_mse.idxmin()];last=h.iloc[-1]
        curves.append({'experiment':c['experiment']['id'],'seed':seed,'best_epoch':int(best.epoch),'epochs':len(h),
                       'best_val_mse':float(best.val_mse),'last_val_mse':float(last.val_mse),
                       'train_loss_best':float(best.data_loss),'train_loss_last':float(last.data_loss),
                       'probe_best':float(best.probe_mse),'probe_last':float(last.probe_mse),
                       'val_last10_min':float(h.tail(10).val_mse.min()),'val_last10_max':float(h.tail(10).val_mse.max())})
    flip=[predictions(paths[f'augment_01_flip_{s}']) for s in SEEDS]
    trainflip=[pd.read_csv(paths[f'augment_01_flip_{s}']/'train_evaluation/predictions.csv') for s in SEEDS]
    hard=flip[0][['imageid','group_id','price']].copy()
    hard['mean_prediction']=np.mean([f.prediction.to_numpy() for f in flip],axis=0)
    hard['mean_squared_error']=np.mean([f.squared_error.to_numpy() for f in flip],axis=0)
    hard['mean_absolute_error']=np.mean([f.error.abs().to_numpy() for f in flip],axis=0)
    hard['prediction_seed_std']=np.std([f.prediction.to_numpy() for f in flip],axis=0,ddof=1)
    hard['relative_absolute_error']=hard.mean_absolute_error/hard.price
    concentration=[]
    for n in (16,24,80,161):
        top=hard.nlargest(n,'mean_squared_error')
        concentration.append({'n':n,'fraction_of_total_squared_error':float(top.mean_squared_error.sum()/hard.mean_squared_error.sum()),'median_price':float(top.price.median()),'over1300':int((top.price>1300).sum()),'over2000':int((top.price>2000).sum())})
    sheet(hard.nlargest(24,'mean_squared_error'),'top24_squared_error')
    sheet(hard.nlargest(24,'relative_absolute_error'),'top24_relative_error')
    sheet(hard[hard.price<=700].nlargest(24,'mean_squared_error'),'lower_price_hard24')
    sheet(train[train.price>1300].sample(24,random_state=2026),'train_high_price_random24')
    sheet(hard[hard.price>1300].nsmallest(24,'mean_squared_error'),'high_price_easy24')
    hard.to_csv(PRIVATE/'per_image_review.csv',index=False,encoding='utf-8-sig')
    result={'split':audit,'per_run_bands':rows,'control_bands_2026':band_metrics([base[2026]]),
            'control_bands_three_seeds':band_metrics(list(base.values())),'flip_bands_three_seeds':band_metrics(flip),
            'flip_train_bands_three_seeds':band_metrics(trainflip),'curves':curves,'error_concentration':concentration,
            'ranking_note':'Mean squared errors of three separately trained models, not ensemble prediction MSE.',
            'flip_prediction_distribution':{'target_std':float(hard.price.std()),'per_seed_prediction_std':[float(f.prediction.std()) for f in flip],
                                             'target_min':float(hard.price.min()),'target_max':float(hard.price.max()),
                                             'prediction_min':[float(f.prediction.min()) for f in flip],'prediction_max':[float(f.prediction.max()) for f in flip]}}
    write_json(OUT/'metrics.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,3,figsize=(16,4.8))
    edges=np.linspace(0,max(train.price.max(),val.price.max()),26)
    axes[0].hist(train.price,bins=edges,weights=np.ones(len(train))/len(train),alpha=.5,label='训练集')
    axes[0].hist(val.price,bins=edges,weights=np.ones(len(val))/len(val),alpha=.5,label='验证集');axes[0].set_title('价格分布（比例）');axes[0].legend()
    for seed,f in zip(SEEDS,flip):axes[1].scatter(f.price,f.prediction,s=5,alpha=.15,label=str(seed))
    limit=max(hard.price.max(),max(f.prediction.max() for f in flip));axes[1].plot([0,limit],[0,limit],'k--');axes[1].set_title('水平翻转：真实价格与预测');axes[1].set_xlabel('真实价格');axes[1].set_ylabel('预测价格')
    z=result['flip_bands_three_seeds'];axes[2].bar(LABELS,[x['contribution']/126810.00739756384*100 for x in z]);axes[2].set_title('各价格段对总平方误差的贡献（%）')
    fig.tight_layout();fig.savefig(OUT/'price_and_error.png',dpi=150);plt.close(fig)
    print(json.dumps({k:result[k] for k in ('split','flip_bands_three_seeds','flip_train_bands_three_seeds','error_concentration','flip_prediction_distribution')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
