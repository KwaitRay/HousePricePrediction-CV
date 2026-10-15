"""Direct-review high-price labels and grouped out-of-fold diagnostic errors."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from common.local_paths import local_path, REPOSITORY
from common.config import digest, write_json
from training_scene_index import build, FEATURES

RANGES=['≤100','100–300','300–500','500–1000','>1000']
NAMES={'visible_pool':'可见泳池','twilight_night':'黄昏/夜景','major_vegetation_occlusion':'明显植被遮挡',
       'woodland_cabin_appearance':'木质山林小屋外观','aerial_view':'航拍/高位','partial_subject':'局部/环境',
       'interior_view':'室内','rendering':'效果图','large_building':'明显宽阔建筑体量','multistorey':'明确多层'}

def table(frame):
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in frame.itertuples(index=False,name=None))

def run():
    index=pd.read_csv(build(),dtype={'imageid':str,'group_id':str})
    split=local_path('split_path')
    frame=pd.read_csv(split,dtype={'imageid':str,'group_id':str})
    ref=local_path('records_root')/'03_clip_reference/runs/20261013T043141Z_c9f899f2'
    manifest=json.loads((ref/'data_manifest.json').read_text(encoding='utf-8'))
    if digest(split)!=manifest['split_sha256']:raise ValueError('Feature split changed')
    import hashlib
    cache_key=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    cache=local_path('cache_root')/'clip_features'/cache_key
    files=sorted(cache.glob('*.npy'))
    x=np.concatenate([np.load(p) for p in files])
    ordered=frame.set_index('imageid').loc[list(map(str,manifest['imageids']))].reset_index()
    if len(x)!=len(ordered) or x.shape[1]!=512 or not np.isfinite(x).all():raise ValueError('Invalid feature cache')
    mask=ordered.partition.eq('train').to_numpy()
    train=ordered.loc[mask].reset_index(drop=True); x=x[mask]
    if set(train.imageid)!=set(index.imageid):raise ValueError('Training index mismatch')
    local=REPOSITORY/'.local/artifacts/high_price_annotation'
    destination=local/'train_clip_grouped_oof.csv'
    protocol={'alpha':1000,'folds':3,'split_sha256':digest(split),'manifest_sha256':digest(ref/'data_manifest.json'),
              'model':'Frozen CLIP ViT-B/32 + fold-specific StandardScaler + Ridge(svd)',
              'selection_caveat':'Alpha was previously selected using training CV. This is a diagnostic OOF replay, not nested-CV unbiased performance estimation.'}
    if destination.exists() and (local/'oof_protocol.json').exists() and json.loads((local/'oof_protocol.json').read_text())==protocol:
        predictions=pd.read_csv(destination,dtype={'imageid':str})
    else:
        pred=np.full(len(train),np.nan);fold=np.full(len(train),-1)
        with threadpool_limits(limits=4):
            for n,(fit,held) in enumerate(GroupKFold(3).split(x,train.price,train.group_id)):
                assert not set(train.group_id.iloc[fit])&set(train.group_id.iloc[held])
                model=make_pipeline(StandardScaler(),Ridge(alpha=1000,solver='svd'))
                model.fit(x[fit],train.price.iloc[fit].to_numpy()/1000)
                pred[held]=model.predict(x[held])*1000;fold[held]=n
                print(f'OOF fold {n+1}/3 finished',flush=True)
        if not np.isfinite(pred).all():raise ValueError('Incomplete OOF')
        predictions=pd.DataFrame({'imageid':train.imageid,'oof_prediction':pred,'fold':fold})
        predictions.to_csv(destination,index=False,encoding='utf-8-sig');write_json(local/'oof_protocol.json',protocol)
    merged=index.merge(predictions,on='imageid',validate='one_to_one')
    high=merged.loc[merged.price>1300].copy()
    assert len(high)==636 and high.annotation_status.eq('已审阅').all()
    high['error']=high.oof_prediction-high.price
    high['absolute_error']=high.error.abs();high['squared_error']=high.error**2
    high['error_range']=pd.cut(high.absolute_error,[0,100,300,500,1000,np.inf],labels=RANGES,include_lowest=True)
    high.sort_values('absolute_error',ascending=False).to_csv(local/'high_price_errors.csv',index=False,encoding='utf-8-sig')
    public=REPOSITORY/'output/experiment_record/09_data_quality/training_categories/high_price'
    public.mkdir(parents=True,exist_ok=True)
    scenes=pd.crosstab(high.error_range,high.scene_primary,dropna=False).reindex(RANGES,fill_value=0)
    scenes['总数']=scenes.sum(axis=1)
    assert scenes['总数'].sum()==636
    scenes.to_csv(public/'error_scene_counts.csv',encoding='utf-8-sig')
    scenes.div(scenes['总数'].replace(0,np.nan),axis=0).to_csv(public/'error_scene_row_fractions.csv',encoding='utf-8-sig')
    features=[];metrics=[]
    def stats(subset,label,status):
        return {'类别':label,'状态':status,'数量':len(subset),'平均真实价格':round(float(subset.price.mean()),2) if len(subset) else None,
                'MAE':round(float(subset.absolute_error.mean()),2) if len(subset) else None,
                'MSE':round(float(subset.squared_error.mean()),2) if len(subset) else None,
                '平均偏差_预测减真实':round(float(subset.error.mean()),2) if len(subset) else None,
                '误差>500比例':round(float(subset.absolute_error.gt(500).mean()),4) if len(subset) else None,
                '误差>1000比例':round(float(subset.absolute_error.gt(1000).mean()),4) if len(subset) else None}
    metrics.append(stats(high,'全部','全部'))
    for scene,subset in high.groupby('scene_primary'):metrics.append(stats(subset,scene,'场景'))
    for feature in FEATURES:
        for status in ['是','否','未知']:
            subset=high.loc[high[feature]==status];metrics.append(stats(subset,NAMES[feature],status))
            for band in RANGES:
                count=int(subset.error_range.eq(band).sum());band_n=int(high.error_range.eq(band).sum())
                features.append({'误差范围_千美元':band,'特征':NAMES[feature],'状态':status,'数量':count,
                    '该误差档比例':count/band_n if band_n else None,'该特征状态内比例':count/len(subset) if len(subset) else None})
    feature_frame=pd.DataFrame(features);metric_frame=pd.DataFrame(metrics)
    feature_frame.to_csv(public/'error_feature_counts.csv',index=False,encoding='utf-8-sig')
    metric_frame.to_csv(public/'category_metrics.csv',index=False,encoding='utf-8-sig')
    write_json(public/'summary.json',{'high_price_count':636,'annotations_sha256':digest(local/'annotations.json'),
        'manifest_sha256':digest(local/'manifest.csv'),'protocol':protocol,'absolute_error_units':'thousand USD',
        'counts':{str(k):int(v) for k,v in high.error_range.value_counts(sort=False).items()},
        'overall':metrics[0],'label_counts':{f:high[f].value_counts().to_dict() for f in FEATURES}})
    yes=metric_frame.loc[metric_frame['状态'].eq('是') & metric_frame['数量'].gt(0)]
    feature_counts=feature_frame.loc[feature_frame['状态'].eq('是')].pivot(index='误差范围_千美元',columns='特征',values='数量').reindex(RANGES)
    feature_counts=feature_counts[[NAMES[f] for f in FEATURES if high[f].eq('是').any()]]
    report='''# 高价组视觉标注与误差分布

已对固定训练分区中价格 >1300 千美元的全部 636 张图片逐图完成一轮助手视觉审阅。图片随机排序，只展示图片和 H 编号，未展示具体价格、预测或误差。低价、中价仍只保留原抽样标注。此轮并非独立专家标注；未知字段保留，不代表所有属性均已确定。

主场景互斥：地面外观、航拍或高位全景、局部或环境特写、室内、建筑效果图、未知。泳池、多层、黄昏等是可重叠的可见特征。否表示该照片未显示该特征，不表示房产没有泳池或不是多层。航拍顶视和局部照片无法判断层数时记未知。明显宽阔建筑体量仅标记画面明确展示的宽阔/多翼主体；无法推断真实面积，大量未知不能用于体量类别的可靠比较。遮挡约超过主体立面三分之一为阳性，无法确定遮挡比例保留未知。泳池与自然湖泊分开，木质小屋不等同于附近有树。

误差来自训练侧 3 折 GroupKFold 的折外预测，每折独立拟合 StandardScaler 和 Ridge，group_id 不跨折。使用已选定的 CLIP 特征及 alpha=1000，不重新调参，不使用验证或测试标签。此前 alpha 在训练 CV 上选择，因此结果用于定位训练样本问题，不能当成无选择偏差的泛化成绩，也不能直接与原验证 MSE 或 Kaggle 公榜归因等同。每张图片的预测来自不含其所属组的训练折。

以下误差范围按绝对误差分档，单位千美元；100 即 10 万美元，MSE 单位是千美元的平方。

## 各误差档的主场景数量

'''+table(scenes.reset_index())+'\n## 各误差档的可见特征阳性数量（标签可重叠）\n\n'+table(feature_counts.reset_index())+'\n## 可见特征阳性组的误差\n\n'+table(yes)+'\n## 解读与后续使用\n\n'
    baseline=float(high.absolute_error.gt(500).mean())
    report+=f'全部高价图的 MAE 为 {high.absolute_error.mean():.2f} 千美元，平均偏差为 {high.error.mean():.2f} 千美元，误差超过 50 万美元的比例为 {baseline:.1%}。\n\n'
    for f in ['visible_pool','aerial_view','partial_subject','twilight_night','multistorey']:
        s=high[high[f]=='是'];no=high[high[f]=='否']
        report+=f'- {NAMES[f]}：阳性 {len(s)} 张，误差 >500 的比例 {s.absolute_error.gt(500).mean():.1%}；画面未见该特征的 {len(no)} 张为 {no.absolute_error.gt(500).mean():.1%}。两组真实价格分布可能不同，属于描述性关联。\n'
    report+='''
当前最大的错误来源是地面外观：359 张中有261张绝对误差超过50万美元，其中51张超过100万美元；但样本占比本身也最大。地面外观大误差比例为72.7%，航拍为47.8%。可见泳池94张中有46张超过50万美元（48.9%），只有1张超过100万美元；本训练高价分区不支持“泳池是主要困难类别”的猜测。多层301张中有191张超过50万美元（63.5%），同样不能仅凭频数认定多层特征导致错误。明显植被遮挡阳性11张中10张超过50万美元，值得逐图检查，但样本少且遮挡不确定项较多，不能推广为可靠因果结论。效果图仅5张，暂作个案诊断。

高价组整体表现以低估为主。由于此分组用真实价格定义，这种条件偏差本身不构成可以在测试时识别并统一上调“高价组”的依据。

先定位样本充足且大误差比例较高的类别，再结合逐图误差表检查主体大小、图像细节和视觉信息不足。场景频数高不等于该场景风险高，应结合类别内误差比例、平均真实价格和样本数比较；标签重叠，不能相加得出总图片数。小样本类别不做确定性结论，当前统计也不能证明注意力机制或数据配比是误差原因。新增增强或加权实验仍用原始未加权验证 MSE，并检查低价高估。

本地逐图文件：`.local/artifacts/high_price_annotation/high_price_errors.csv`（按绝对误差降序，含 H 编号、imageid、真实价格、折外预测、误差与标签）。审阅证据为同目录 `manifest.csv`、`annotations.json`、32 张 contact sheets。原高价抽样标签保留在原审阅目录，新主索引覆盖为本轮标签。标签仅用于诊断，不作为模型输入或自动删数据依据。

`error_scene_counts.csv` 是互斥主场景计数；`error_scene_row_fractions.csv` 是误差档内占比；`error_feature_counts.csv` 同时提供各特征是/否/未知计数、误差档内占比和特征状态内部的误差分布；`category_metrics.csv` 提供对照指标。运行：`.venv\\Scripts\\python.exe scripts/experiments/high_price_scene_errors.py`。
'''
    (public/'高价标注与误差分布.md').write_text(report,encoding='utf-8')
    print(scenes.to_string(),flush=True);print(yes.to_string(index=False),flush=True)
    print(public/'高价标注与误差分布.md',flush=True)

if __name__=='__main__':run()
