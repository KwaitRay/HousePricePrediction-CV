"""Render the audited aggregate evidence without touching training or raw data."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'output/experiment_record/04_training/analyses/completed_20261004'


def main():
    rows = json.loads((EVIDENCE / 'aggregates.json').read_text(encoding='utf-8'))
    names = ['现有对照', 'Adam 0.001', '余弦衰减', '有效批量16', '有效批量64', '随机失活0', '随机失活0.2']
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False})
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    y = np.arange(len(rows))
    axes[0].errorbar([r['mean'] for r in rows], y, xerr=[r['std'] for r in rows], fmt='o', capsize=4)
    axes[0].set_yticks(y, names); axes[0].invert_yaxis()
    axes[0].axvline(rows[0]['mean'], color='gray', linestyle='--')
    axes[0].set(title='三种子平均验证均方误差（越低越好）', xlabel='误差棒为种子间样本标准差，不是置信区间')
    colors = ['#4776b4', '#73ab73', '#f3aa48', '#c95c5c']
    pos = np.zeros(6); neg = np.zeros(6)
    for i, name in enumerate(['低价≤300', '中价300–700', '高价700–1300', '尾部>1300']):
        v = np.array([r['groups'][i]['contribution'] for r in rows[1:]])
        axes[1].barh(np.arange(6), v, left=np.where(v >= 0, pos, neg), color=colors[i], label=name)
        pos += np.maximum(v, 0); neg += np.minimum(v, 0)
    axes[1].set_yticks(np.arange(6), names[1:]); axes[1].invert_yaxis()
    axes[1].axvline(0, color='gray'); axes[1].set(title='各价格组对总误差变化的贡献', xlabel='正值使结果变差，负值使结果改善')
    axes[1].legend(fontsize=8, loc='lower right')
    fig.tight_layout(); fig.savefig(EVIDENCE / 'comparison.png', dpi=150); plt.close(fig)


if __name__ == '__main__': main()
