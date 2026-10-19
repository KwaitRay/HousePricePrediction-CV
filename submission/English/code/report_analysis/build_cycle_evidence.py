"""Build focused route/branch comparisons and deterministic success/failure cases."""
from pathlib import Path
import os
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
LOCAL=REPO/'evidence/components'
FIG=ROOT/'figures'
SRC=FIG/'analysis_sources'
old=pd.read_csv(SRC/'report_compact_results_table.csv')
if not (SRC/'representative_table_before_cycles.csv').exists():old.to_csv(SRC/'representative_table_before_cycles.csv',index=False)
old=pd.read_csv(SRC/'representative_table_before_cycles.csv')
rows=[]
for method,name in [('AlexNet (no augmentation)','AlexNet（无增强）'),('AlexNet + horizontal flip','AlexNet + 水平翻转'),('Flip AlexNet + mean-pooling head','翻转 AlexNet + 均值池化头'),('Frozen CLIP B32 + Ridge','冻结 CLIP B32 + 岭回归')]:
    row=old[old.method.eq(method)].iloc[0].to_dict();row.update(name_zh=name,group='A');rows.append(row)
branches=[
 ('NaFlex + RBF','NaFlex + RBF','siglip2_naflex_576_kernel/20261014T020905Z_e457778a'),
 ('PE-L, three views + RBF','PE-L 三视图 + RBF','pe_core_l14_336_multiview_kernel/20261014T043810Z_6840dfd9'),
 ('PE-L, three views + PLS/RBF','PE-L 三视图 + PLS/RBF','pe_core_multiview_price_metric_kernel/20261014T051116Z_13d3ea16'),
 ('PE-G, official view + RBF','PE-G 官方视图 + RBF','pe_core_g14_448_kernel/20261014T073812Z_e0339201'),
 ('PE-G, two views + RBF','PE-G 双视图 + RBF','pe_core_g14_448_whole_kernel/20261014T090655Z_2d55c743'),
 ('PE-G, two views + PCA/RBF','PE-G 双视图 + PCA/RBF','pe_giant_whole_pca_bandwidth_kernel/20261015T033416Z_4b848f75')]
sources=[]
for name,zh,rel in branches:
    path=LOCAL/'19_image_embedding_reference'/rel/'predictions.csv';sources.append(path)
    df=pd.read_csv(path);assert len(df)==1601
    e=df.prediction-df.price
    rows.append(dict(method=name,name_zh=zh,seeds='—',validation_mse=float(np.mean(e**2)),mse_seed_sd=np.nan,validation_mae=float(np.mean(abs(e))),source=rel,group='B'))
row=old.iloc[-1].to_dict();row.update(name_zh='三编码器融合',group='B');rows.append(row)
data=pd.DataFrame(rows);data.to_csv(SRC/'report_compact_results_table.csv',index=False)

body=[]
for i,r in data.iterrows():
    if i in [0,4]:
        body.append(r'\multicolumn{6}{l}{\emph{'+('A. Representation route' if i==0 else 'B. Frozen branches and fusion')+r'}} \\')
    sd='' if pd.isna(r.mse_seed_sd) else f'{r.mse_seed_sd:,.0f}'
    pm='' if not sd else r'$\pm$'
    cells=[r.method,'--' if r.seeds=='—' else str(r.seeds),f'{r.validation_mse:,.0f}',pm,sd,f'{r.validation_mae:.2f}']
    if i==len(data)-1:cells=[r'\textbf{'+x+'}' for x in cells]
    body.append(' & '.join(cells)+r' \\')
tex=r'''\begin{table*}[t]
\centering
\caption{From learned representations to the selected frozen branches.}
\label{tab:representative-results}
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.0}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcr@{\,}c@{\,}rr@{}}
\hline
Method & Seeds & \multicolumn{3}{c}{Validation MSE} & Validation MAE \\
\hline
'''+ '\n'.join(body)+r'''
\hline
\end{tabular*}
\par\smallskip
\begin{minipage}{0.98\textwidth}\scriptsize
Validation $n=1{,}601$; MSE: (thousand USD)$^2$; MAE: thousand USD.
Three seeds: mean $\pm$ sample SD; --: frozen-feature evaluation.
Panel A compares complete routes; matched PE-L and PE-G rows in B isolate the listed changes.
\end{minipage}
\end{table*}
'''
(FIG/'report_compact_results_table.tex').write_text(tex,encoding='utf-8')

final=REPO/'evidence/final_validation_predictions.csv'
df=pd.read_csv(final);df['absolute_error']=abs(df.prediction-df.price)
ranked=df.sort_values(['absolute_error','imageid'],kind='stable')
success=ranked.iloc[round(.25*(len(ranked)-1))]
cases=pd.concat([success.to_frame().T,df[df.imageid.eq('5352.jpg')],df[df.imageid.eq('2138.jpg')]],ignore_index=True)
cases['selection']=['25th percentile of absolute error','Fixed overestimation case','Fixed underestimation case']
cases.to_csv(SRC/'report_success_failure_cases.csv',index=False)
settings=json.loads((REPO/('local_paths.json' if (REPO/'local_paths.json').exists() else 'local_paths.example.json')).read_text(encoding='utf-8'))
data_root=Path(os.environ.get('CV_DATA_ROOT',str(REPO/settings['data_root'])))/'train'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,3,figsize=(7,1.28))
fig.subplots_adjust(left=.005,right=.995,bottom=.19,top=.98,wspace=.08)
images=[]
for ax,row in zip(axes,cases.itertuples()):
    path=data_root/row.imageid;images.append(path)
    with Image.open(path) as im:ax.imshow(ImageOps.exif_transpose(im).convert('RGB'))
    ax.axis('off')
    ax.text(.02,.03,f'{row.imageid[:-4]}: y={float(row.price):,.1f}; ŷ={float(row.prediction):,.1f}',transform=ax.transAxes,
            fontsize=6,bbox=dict(facecolor='white',alpha=.86,edgecolor='none',pad=1))
for ax,label in zip(axes,['(a) Low error (25th percentile)','(b) Overestimation','(c) Underestimation']):
    ax.text(.5,-.035,label,transform=ax.transAxes,ha='center',va='top',fontsize=7)
for suffix in ['pdf','png','svg']:fig.savefig(FIG/f'report_success_failure_cases.{suffix}',dpi=360,bbox_inches='tight',pad_inches=.025)
plt.close(fig)
# Fit regional maps and independent outcome cases into one clearly divided panel.
from matplotlib.colors import TwoSlopeNorm
table=pd.read_csv(SRC/'occlusion/per_region_predictions.csv')
norm=TwoSlopeNorm(vmin=-float(table.delta_price.abs().max()),vcenter=0,vmax=float(table.delta_price.abs().max()))
fig,axes=plt.subplots(2,5,figsize=(7,2.25),gridspec_kw={'width_ratios':[1,1,1,1,1.18]})
fig.subplots_adjust(left=.005,right=.995,bottom=.15,top=.87,wspace=.075,hspace=.14)
for j,title in enumerate(['Original','Single view','Two views','+ PCA','Fusion outcomes']):axes[0,j].set_title(title,fontsize=7,pad=3)
models=['Single-view KRR','Two-view KRR','Two-view PCA KRR']
for i,name in enumerate(['4920.jpg','5712.jpg']):
    path=data_root/name;images.append(path)
    with Image.open(path) as im:arr=np.array(ImageOps.exif_transpose(im).convert('RGB'))
    h,w=arr.shape[:2];sub=table[table.imageid.eq(name)]
    for j in range(4):
        ax=axes[i,j];ax.imshow(arr);ax.axis('off')
        if j:
            selected=sub[sub['mode'].eq('blur')&sub.model.eq(models[j-1])].sort_values('cell')
            artist=ax.imshow(selected.delta_price.to_numpy().reshape(4,4),cmap='RdBu_r',norm=norm,alpha=.58,
                interpolation='nearest',extent=(-.5,w-.5,h-.5,-.5))
            for a in range(1,4):
                ax.axhline(a*h/4-.5,color='white',alpha=.4,lw=.3);ax.axvline(a*w/4-.5,color='white',alpha=.4,lw=.3)
            value=float(sub[sub['mode'].eq('original')&sub.model.eq(models[j-1])].prediction.iloc[0])
            label=f'ŷ={value:,.0f}'
        else:label=f'{name[:-4]}: y={float(sub.true_price.iloc[0]):,.1f}'
        ax.text(.02,.025,label,transform=ax.transAxes,fontsize=5.5,bbox=dict(facecolor='white',alpha=.85,pad=1,edgecolor='none'))
    row=cases.iloc[0 if i==0 else 2]
    ax=axes[i,4]
    with Image.open(data_root/row.imageid) as im:ax.imshow(ImageOps.exif_transpose(im).convert('RGB'))
    ax.axis('off')
    ax.text(.02,.025,('Low error\n' if i==0 else '')+f'{row.imageid[:-4]}: y={float(row.price):,.1f}\nŷ={float(row.prediction):,.1f}',transform=ax.transAxes,
        fontsize=5.5,bbox=dict(facecolor='white',alpha=.88,pad=1,edgecolor='none'))
    if i==1:ax.text(.5,1.015,'High error',transform=ax.transAxes,ha='center',fontsize=6,color='#a04e68')
fig.add_artist(plt.Line2D([.776,.776],[.14,.94],transform=fig.transFigure,color='.7',lw=.6))
cax=fig.add_axes([.11,.055,.53,.022]);cb=fig.colorbar(artist,cax=cax,orientation='horizontal',ticks=[-400,-200,0,200,400])
cb.ax.tick_params(labelsize=5.5,pad=1);cb.set_label('Prediction drop under blur (thousand USD)',fontsize=6,labelpad=1)
for suffix in ['pdf','png','svg']:fig.savefig(FIG/f'report_regional_and_outcome_cases.{suffix}',dpi=360,bbox_inches='tight',pad_inches=.025)
plt.close(fig)
meta={'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources+[final]+images},
      'selection':'Success: nearest order statistic to 25th absolute-error percentile, stable imageid tie-break; failures retained from prior report.',
      'low_error_case':cases.iloc[0].to_dict()}
(SRC/'report_cycle_evidence_provenance.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print(data[['method','validation_mse','validation_mae']].to_string(index=False))
print(cases[['imageid','price','prediction','absolute_error']].to_string(index=False))
