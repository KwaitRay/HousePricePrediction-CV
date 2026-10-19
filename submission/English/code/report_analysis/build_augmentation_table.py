"""Build bilingual single-column augmentation tables from archived metrics."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'


def build():
    early = pd.read_csv(FIG / 'analysis_sources/early_experiment_paired_deltas.csv')
    early = early[(early.panel == 'Augmentation') & (early.seed == 2026)]
    photo = pd.read_csv(FIG / 'analysis_sources/photometric_screen_20261019/summary.csv')
    labels = [('Horizontal flip', '水平翻转'), ('Rotation', '旋转'),
              ('Translation', '平移'), ('Random crop', '随机裁剪'),
              ('Gaussian blur', '高斯模糊'), ('Gaussian noise', '高斯噪声'),
              ('Random erasing', '随机擦除')]
    for zh in (False, True):
        suffix = '_zh' if zh else ''
        rows = []
        md = ['| ' + ('设置' if zh else 'Setting') + ' | MSE | MAE |', '|---|---:|---:|']
        def group(en, cn):
            title = cn if zh else en
            rows.append(r'\multicolumn{3}{l}{\emph{' + title + r'}} \\')
            md.append(f'| **{title}** | | |')
        def row(name, mse, mae):
            rows.append(f'{name} & {mse:,.2f} & {mae:.2f}' + r' \\')
            md.append(f'| {name} | {mse:,.2f} | {mae:.2f} |')
        group('A. Geometric and degradation screen', 'A. 几何与退化增强初筛')
        control = early.iloc[0]
        row('无增强' if zh else 'No augmentation', control.control_mse, control.control_mae)
        for en, cn in labels:
            selected = early[early.candidate == en]
            assert len(selected) == 1
            item = selected.iloc[0]
            row(cn if zh else en, item.candidate_mse, item.candidate_mae)
        group('B. Photometric screen', 'B. 光度增强初筛')
        for key, en, cn in [('control', 'Fresh control', '新训练对照'),
                            ('brightness', 'Brightness', '亮度'), ('contrast', 'Contrast', '对比度')]:
            item = photo[photo.group == key].iloc[0]
            row(cn if zh else en, item.mse, item.mae)
        note = ('种子 2026，验证集 n=1,601；每次只增加一种增强。A、B 分别使用各自的无增强对照。'
                if zh else 'Seed 2026; validation n=1,601. One augmentation per run; A and B use their respective unaugmented controls.')
        caption = 'AlexNet 数据增强对照。' if zh else 'AlexNet augmentation comparisons.'
        tex = r'''\begin{table}[t]
\centering
\caption{''' + caption + r'''}
\label{tab:augmentation''' + ('-zh' if zh else '') + r'''}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrr@{}}
\hline
''' + ('设置' if zh else 'Setting') + r''' & MSE & MAE \\
\hline
''' + '\n'.join(rows) + r'''
\hline
\end{tabular*}
\par\smallskip
\begin{minipage}{\columnwidth}\scriptsize
''' + note + r'''
\end{minipage}
\end{table}
'''
        stem = 'report_augmentation_comparison' + suffix
        (FIG / (stem + '.tex')).write_text(tex, encoding='utf-8')
        (FIG / (stem + '.md')).write_text('\n'.join(md) + '\n\n' + note + '\n', encoding='utf-8')


if __name__ == '__main__':
    build()
