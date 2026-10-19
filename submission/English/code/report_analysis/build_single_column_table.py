"""Uniform point estimates in a native single-column IEEE table."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures'
DATA=pd.read_csv(FIG/'analysis_sources/report_compact_results_table.csv')
EN=['AlexNet','AlexNet + flip','Flip AlexNet + mean head','CLIP B/32 + Ridge',
    'NaFlex + RBF','PE-L (3 views) + RBF','PE-L (3 views) + PLS/RBF',
    'PE-G (official view) + RBF','PE-G (2 views) + RBF','PE-G (2 views) + PCA/RBF','Three-encoder fusion']
ZH=['AlexNet','AlexNet + 翻转','翻转 AlexNet + 均值头','CLIP B/32 + 岭回归',
    'NaFlex + RBF','PE-L（三视图）+ RBF','PE-L（三视图）+ PLS/RBF',
    'PE-G（官方视图）+ RBF','PE-G（双视图）+ RBF','PE-G（双视图）+ PCA/RBF','三编码器融合']
assert len(DATA)==len(EN)==len(ZH)


def build():
    for zh,names in [(False,EN),(True,ZH)]:
        body=[r'\multicolumn{3}{l}{\emph{'+('参照基线' if zh else 'Reference baselines')+r'}} \\',
              ('训练均价' if zh else 'Training mean')+r' & 151,840 & 294.47 \\',
              r'SIFT--BoVW--Ridge & 145,337 & 288.66 \\']
        for i,(r,name) in enumerate(zip(DATA.itertuples(),names)):
            if i in [0,4]:
                group=(['A. 表示路线','B. 冻结分支与融合'] if zh else ['A. Representation route','B. Frozen branches and fusion'])[i==4]
                body.append(r'\multicolumn{3}{l}{\emph{'+group+r'}} \\')
            cells=[name,f'{r.validation_mse:,.0f}',f'{r.validation_mae:.2f}']
            if i==len(DATA)-1:cells=[r'\textbf{'+x+'}' for x in cells]
            body.append(' & '.join(cells)+r' \\')
        label='tab:representative-results'+('-zh' if zh else '')
        caption='固定验证集上的代表性结果。' if zh else 'Representative validation results.'
        note=(r'$n=1{,}601$；MSE 单位为（千美元）$^2$，MAE 为千美元。SIFT 与三个 AlexNet 结果为三种子均值；冻结特征行使用选定拟合。' if zh else
              r'$n=1{,}601$; MSE: (thousand USD)$^2$; MAE: thousand USD. SIFT and the three AlexNet rows are three-seed means; frozen-feature rows use selected fits.')
        tex=r'''\begin{table}[t]
\centering
\caption{'''+caption+r'''}
\label{'''+label+r'''}
\small
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.0}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrr@{}}
\hline
'''+('方法' if zh else 'Method')+r''' & MSE & MAE \\
\hline
'''+ '\n'.join(body)+r'''
\hline
\end{tabular*}
\par\smallskip
\begin{minipage}{\columnwidth}\scriptsize
'''+note+r'''
\end{minipage}
\end{table}
'''
        (FIG/('report_compact_results_table'+('_zh' if zh else '')+'.tex')).write_text(tex,encoding='utf-8')


if __name__=='__main__':build()
