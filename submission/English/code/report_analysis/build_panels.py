"""Compact publication panels from existing validation/occlusion records; CPU only."""
import os
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
ASSIGNMENT = ROOT.parent
REPO = ROOT.parent
FIG = ROOT / 'figures'
SRC = FIG / 'analysis_sources'
PRED = REPO / 'evidence/final_validation_predictions.csv'
DATA = Path(os.environ.get('CV_DATA_ROOT', str(REPO/'dataset')))/'train'
SOURCES = [PRED, SRC / 'occlusion/per_region_predictions.csv', SRC / 'pca_sampled_pair_geometry.csv', SRC / 'pca_bandwidth_oof_comparison.csv']
OUTPUTS = []
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7,
    'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5, 'legend.fontsize': 6,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
COLORS = ['#287d78', '#426b9c', '#b68a53', '#a04e68']


def save(fig, stem):
    for ext in ['png', 'svg', 'pdf']:
        path = FIG / f'{stem}.{ext}'
        fig.savefig(path, dpi=360, facecolor='white', bbox_inches='tight', pad_inches=.025)
        OUTPUTS.append(path)
    plt.close(fig)


def prediction_panels():
    df = pd.read_csv(PRED)
    assert len(df) == 1601 and df.imageid.is_unique
    y, p = df.price.to_numpy(), df.prediction.to_numpy()
    error = p-y
    np.testing.assert_allclose(error, df.error)
    assert abs(np.mean(error**2)-52201.21126766699) < 1e-7
    groups = pd.cut(y, [0, 300, 700, 1300, np.inf], right=True, labels=False)
    assert not np.isnan(groups).any()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.92), gridspec_kw={'width_ratios': [1.1, 1, 1]})
    fig.subplots_adjust(left=.07, right=.99, bottom=.25, top=.86, wspace=.48)
    ax = axes[0]
    for i, color in enumerate(COLORS):
        use = groups == i
        ax.scatter(y[use], p[use], s=3, alpha=.5, c=color, edgecolors='none', rasterized=True)
    ax.plot([0,2050],[0,2050], '--', color='.3', lw=.8)
    ax.set(xlim=(0,2050), ylim=(0,2050), xticks=[0,1000,2000], yticks=[0,1000,2000],
           xlabel='True price (thousand USD)', ylabel='Predicted price', title='(a) Prediction vs. true price')
    # Equal scales prevent distortion of the agreement line.
    ax.set_aspect('equal', adjustable='box')
    stats=[]
    for i in range(4):
        mask=groups==i
        stats.append(dict(band=i,n=int(mask.sum()),mean_residual=float(error[mask].mean()),
                          image_share=float(mask.mean()*100),sse_share=float((error[mask]**2).sum()/(error**2).sum()*100)))
    stats=pd.DataFrame(stats)
    x=np.arange(4)
    labels=['0–300','300–700','700–1300','>1300']
    ax=axes[1]
    ax.bar(x,stats.mean_residual,color=COLORS,width=.65)
    ax.axhline(0,color='.35',lw=.6)
    ax.set(ylim=(-485,170),xticks=x,xticklabels=labels,ylabel='Mean residual (thousand USD)',title='(b) Price-dependent bias')
    for i,v in enumerate(stats.mean_residual):
        ax.text(i,v+(12 if v>=0 else -14),f'{v:+.0f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=6.5)
    ax=axes[2]
    ax.bar(x-.17,stats.image_share,width=.34,color='#9aaec4',label='Images')
    ax.bar(x+.17,stats.sse_share,width=.34,color='#266e83',label='Squared error')
    ax.set(ylim=(0,65),xticks=x,xticklabels=labels,ylabel='Share (%)',title='(c) Concentration of error')
    ax.legend(frameon=False,loc='upper center',ncol=1,handlelength=1)
    for i,v in enumerate(stats.sse_share):
        ax.text(i+.17,v+1.2,f'{v:.1f}',ha='center',fontsize=6)
    for ax in axes[1:]:
        ax.tick_params(axis='x',labelrotation=25,pad=1)
        ax.set_xlabel('True price band (thousand USD)',labelpad=2)
    for ax in axes:
        ax.grid(axis='y',alpha=.15,zorder=0)
        ax.set_axisbelow(True)
    stats.to_csv(SRC/'report_prediction_price_band_summary.csv',index=False)
    save(fig,'report_prediction_scatter_price_band_errors')


def pca_panels():
    pairs=pd.read_csv(SOURCES[2]); scores=pd.read_csv(SOURCES[3])
    assert len(pairs)==8192
    fig,axes=plt.subplots(1,3,figsize=(7.0,1.75))
    fig.subplots_adjust(left=.06,right=.99,bottom=.26,top=.86,wspace=.45)
    # Empirical CDFs use the original pairs directly, without smoothing choices.
    for col,label,color in [('original_squared_distance','Original',COLORS[1]),('pca_squared_distance','PCA256',COLORS[0])]:
        a=np.sort(pairs[col]); axes[0].plot(a,np.arange(1,len(a)+1)/len(a),label=label,color=color,lw=1)
    axes[0].set(xlabel='Squared distance',ylabel='Cumulative fraction',title='(a) Training-pair distances',ylim=(0,1))
    axes[0].legend(frameon=False,loc='lower right')
    for col,label,color in [('original_kernel','Original',COLORS[1]),('pca_fixed_gamma_kernel','PCA, fixed γ',COLORS[3]),('pca_calibrated_kernel','PCA, calibrated γ',COLORS[0])]:
        a=np.sort(pairs[col]); axes[1].plot(a,np.arange(1,len(a)+1)/len(a),label=label,color=color,lw=1)
    axes[1].set(xlabel='RBF similarity',ylabel='Cumulative fraction',title='(b) Kernel similarities',ylim=(0,1),xlim=(0,.4))
    axes[1].legend(frameon=False,loc='lower right')
    for bw,label,color in [('Fixed gamma=4','Fixed γ = 4',COLORS[3]),('Training-median calibration','Calibrated γ',COLORS[0])]:
        s=scores[(scores.bandwidth==bw)&(scores.whitening_power==0)&(scores.components>0)].sort_values('components')
        axes[2].plot(range(3),s.training_oof_mse/1000,'o-',markersize=3,lw=1,label=label,color=color)
    axes[2].axhline(53646.45535782965/1000,ls=':',color='.4',lw=.8,label='No PCA')
    axes[2].set(xticks=range(3),xticklabels=['64','256','1024'],xlabel='PCA dimensions',ylabel='OOF MSE (×1000)',title='(c) Bandwidth comparison',ylim=(49,99),yticks=[50,70,90])
    axes[2].legend(frameon=False,loc='upper left')
    for ax in axes:
        ax.grid(alpha=.15);ax.set_axisbelow(True)
    save(fig,'report_pca_distance_bandwidth_oof')


def occlusion_panels():
    table=pd.read_csv(SOURCES[1])
    assert len(table)==198
    vmax=float(table.delta_price.abs().max())
    norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax)
    # Show complete blur maps for all three heads; retain gray/blur consistency in text.
    fig,axes=plt.subplots(2,4,figsize=(7.0,2.35))
    fig.subplots_adjust(left=.02,right=.91,bottom=.03,top=.90,wspace=.06,hspace=.25)
    models=['Single-view KRR','Two-view KRR','Two-view PCA KRR']
    titles=['Original image','Single view','Two views','Two views + PCA']
    for j,title in enumerate(titles):axes[0,j].set_title(title,pad=3)
    for i,name in enumerate(['4920.jpg','5712.jpg']):
        path=DATA/name;SOURCES.append(path)
        with Image.open(path) as im:arr=np.array(ImageOps.exif_transpose(im).convert('RGB'))
        height,width=arr.shape[:2]
        sub=table[table.imageid.eq(name)]
        truth=float(sub.true_price.iloc[0])
        for j in range(4):
            ax=axes[i,j];ax.imshow(arr);ax.set_axis_off()
            if j:
                rows=sub[sub['mode'].eq('blur')&sub.model.eq(models[j-1])].sort_values('cell')
                assert len(rows)==16
                heat=rows.delta_price.to_numpy().reshape(4,4)
                artist=ax.imshow(heat,cmap='RdBu_r',norm=norm,alpha=.58,interpolation='nearest',extent=(-.5,width-.5,height-.5,-.5))
                for a in range(1,4):
                    ax.axhline(a*height/4-.5,color='white',alpha=.45,lw=.3)
                    ax.axvline(a*width/4-.5,color='white',alpha=.45,lw=.3)
                pred=float(sub[sub['mode'].eq('original')&sub.model.eq(models[j-1])].prediction.iloc[0])
                label=f'ŷ = {pred:,.0f}'
            else:label=f'{name[:-4]}: y = {truth:,.1f}'
            ax.text(.02,.025,label,transform=ax.transAxes,ha='left',va='bottom',fontsize=6,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.86,'pad':1.2})
    cax=fig.add_axes([.93,.13,.014,.66])
    cb=fig.colorbar(artist,cax=cax,ticks=[-400,-200,0,200,400])
    cb.set_label('Prediction drop (thousand USD)',fontsize=6.5,labelpad=3)
    cb.ax.tick_params(labelsize=6,pad=2)
    save(fig,'report_pe_g_occlusion_model_comparison')


if __name__=='__main__':
    prediction_panels();pca_panels();occlusion_panels()
    payload={'description':'Compact panels from archived records; no inference, training or new selection.',
             'validation_n':1601,'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES},
             'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUTPUTS}}
    (SRC/'report_analysis_panels_provenance.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'outputs':[p.name for p in OUTPUTS]},indent=2))
