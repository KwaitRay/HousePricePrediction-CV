"""Train-only two-level diagnostic index; unreviewed visual properties stay unknown."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from common.local_paths import local_path, REPOSITORY
from common.config import digest, write_json

TYPES={'F':'地面外观','A':'航拍或高位全景','D':'局部或环境特写','I':'室内','R':'建筑效果图','U':'未知'}
FLAGS={'P':'visible_pool','T':'twilight_night','O':'major_vegetation_occlusion','W':'woodland_cabin_appearance'}
FEATURES=[*FLAGS.values(),'aerial_view','partial_subject','interior_view','rendering','large_building','multistorey']


def tier(price):
    # Diagnostic cutoffs, thousand USD; independent of model errors.
    return np.where(price<=300,'低价',np.where(price<=1300,'中价','高价'))


def build():
    split=local_path('split_path')
    frame=pd.read_csv(split,dtype={'imageid':str,'group_id':str})
    train=frame.loc[frame.partition=='train',['imageid','price','group_id','partition']].copy().reset_index(drop=True)
    if len(train)!=6399 or train.imageid.duplicated().any():raise ValueError('Unexpected training partition')
    train['price_tier']=tier(train.price)
    train['scene_primary']='未知'
    for feature in FEATURES:train[feature]='未知'
    train['annotation_status']='未审阅';train['annotation_source']='';train['review_code']=''
    old=local_path('records_root')/'05_augmentation/visual_coverage_20261005'
    sample=pd.read_csv(old/'sample.csv',dtype={'imageid':str})
    labels=pd.read_csv(old/'labels_joined.csv',dtype=str,keep_default_na=False)
    joined=sample.merge(labels,on=['code','band'],validate='one_to_one')
    if len(joined)!=400 or not set(joined.imageid).issubset(set(train.imageid)):
        raise ValueError('Reviewed samples do not match training partition')
    indexed=train.set_index('imageid')
    for _,row in joined.iterrows():
        name=row.imageid
        if indexed.loc[name,'price']!=row.price:raise ValueError('Annotation sample labels changed')
        typ=row['type'];indexed.loc[name,'scene_primary']=TYPES[typ]
        indexed.loc[name,'annotation_status']='已审阅'
        indexed.loc[name,'annotation_source']='20261005助手视觉审阅，含原复核修正'
        indexed.loc[name,'review_code']=row.code
        for key,feature in FLAGS.items():
            value={'True':'是','False':'否','':'未知'}.get(row[key])
            if value is None:raise ValueError('Invalid reviewed flag')
            indexed.loc[name,feature]=value
        for feature,code in [('aerial_view','A'),('partial_subject','D'),('interior_view','I'),('rendering','R')]:
            indexed.loc[name,feature]='未知' if typ=='U' else '是' if typ==code else '否'
    high_folder=REPOSITORY/'.local/artifacts/high_price_annotation'
    if (high_folder/'annotations.json').exists():
        high=pd.read_csv(high_folder/'manifest.csv',dtype={'imageid':str})
        annotations=json.loads((high_folder/'annotations.json').read_text(encoding='utf-8'))
        if len(high)!=636 or len(annotations)!=636 or high.imageid.duplicated().any():
            raise ValueError('Incomplete high-price review')
        flags={**FLAGS,'M':'multistorey','L':'large_building'}
        for (_,row),encoded in zip(high.iterrows(),annotations):
            typ,positive,unknown=encoded.split(':')
            if typ not in TYPES or set(positive)&set(unknown) or not set(positive+unknown)<=set(flags):
                raise ValueError('Invalid annotation')
            name=row.imageid
            if name not in indexed.index or indexed.loc[name,'price']<=1300:
                raise ValueError('High-price scope mismatch')
            indexed.loc[name,'scene_primary']=TYPES[typ]
            indexed.loc[name,'annotation_status']='已审阅'
            indexed.loc[name,'annotation_source']='20261013助手逐图视觉审阅；未展示价格和预测；不确定项保留未知'
            indexed.loc[name,'review_code']=row.code
            for key,feature in flags.items():
                indexed.loc[name,feature]='是' if key in positive else '未知' if key in unknown else '否'
            for feature,code in [('aerial_view','A'),('partial_subject','D'),('interior_view','I'),('rendering','R')]:
                indexed.loc[name,feature]='是' if typ==code else '未知' if typ=='U' else '否'
    result=indexed.reset_index()
    # A multi-label description is a display index, not a model input.
    names={'visible_pool':'画面可见泳池','twilight_night':'黄昏或夜景','major_vegetation_occlusion':'明显植被遮挡',
           'woodland_cabin_appearance':'木质山林小屋外观','aerial_view':'航拍高位','partial_subject':'主体局部',
           'interior_view':'室内','rendering':'效果图','large_building':'较大建筑','multistorey':'多层建筑'}
    result['visible_feature_tags']=result.apply(lambda row:'；'.join(names[k] for k in FEATURES if row[k]=='是') or ('未审阅' if row.annotation_status=='未审阅' else '无已确认的附加阳性标签'),axis=1)
    folder=REPOSITORY/'.local/artifacts/training_scene_index'
    folder.mkdir(parents=True,exist_ok=True)
    destination=folder/'training_scene_index.csv'
    result.to_csv(destination,index=False,encoding='utf-8-sig')
    counts=[]
    for level in ('低价','中价','高价'):
        subset=result[result.price_tier==level]
        for feature in FEATURES:
            counts.append({'price_tier':level,'feature':feature,'feature_name':names[feature],
                'population':len(subset),'known_yes':int(subset[feature].eq('是').sum()),
                'known_no':int(subset[feature].eq('否').sum()),'unknown':int(subset[feature].eq('未知').sum())})
    public=REPOSITORY/'output/experiment_record/09_data_quality/training_categories'
    public.mkdir(parents=True,exist_ok=True)
    write_json(public/'summary.json',{'tier_cutoffs_thousand_usd':{'low':'price <= 300','middle':'300 < price <= 1300','high':'price > 1300'},
        'tier_counts':result.price_tier.value_counts().to_dict(),'reviewed':int(result.annotation_status.eq('已审阅').sum()),'unreviewed':int(result.annotation_status.eq('未审阅').sum()),'cells':counts,
        'caveat':'Known counts are annotations, not population prevalence estimates. Original audit sampled 100 in each of four price strata; merged middle stratum cannot be estimated by raw pooled annotation rates.'})
    write_json(public/'provenance.json',{'split_sha256':digest(split),'prior_sample_sha256':digest(old/'sample.csv'),
        'prior_labels_sha256':digest(old/'labels_joined.csv'),'index_sha256':digest(destination),
        'scope':'Only 6399 development training images; validation and test excluded; no new pretrained labeler',
        'high_review_sha256':digest(high_folder/'annotations.json') if (high_folder/'annotations.json').exists() else None,
        'large_building_and_multistorey':'High-price images reviewed directly; ambiguous visible mass/levels remain unknown; no actual floor area inference'})
    print('Training index:',destination)
    print(result.groupby('price_tier').agg(n=('imageid','size'),reviewed=('annotation_status',lambda s:int(s.eq('已审阅').sum()))).to_string())
    print(result[result.visible_pool=='是'].groupby('price_tier').size().to_string())
    return destination


if __name__=='__main__':build()
