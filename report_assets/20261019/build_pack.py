"""Build a read-only report evidence pack. Never trains or changes submissions.

Rebuild on the originating workstation (private .local evidence is required).
The committed output is self-contained for collaborators; no rebuild is needed.
Run: uv run --with matplotlib python report_assets/20261019/build_pack.py
"""
from pathlib import Path
import hashlib
import html
import json
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
records = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))


def sanitize(value):
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items() if k != 'case_selection'}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        return value.replace(str(ROOT), '$REPO').replace(str(ROOT.parent), '$GROUP')
    return value


def collect(source, target, category):
    src, dst = ROOT / source, HERE / target
    assert src.is_file(), src
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == '.json':
        dst.write_text(json.dumps(sanitize(read(source)), ensure_ascii=False, indent=2), encoding='utf-8')
    else:
        shutil.copy2(src, dst)
    records.append(dict(file=target, source=source, category=category,
                        source_sha256=sha(src), packed_sha256=sha(dst), bytes=dst.stat().st_size))


def main():
    # Curated aggregate evidence; no dataset images, weights, features or row labels.
    sources = [
        ('output/experiment_record/03_traditional/baseline_summary/summary.json', 'data/traditional.json', 'Traditional baseline'),
        ('output/experiment_record/04_training/analyses/completed_20261004/summary.json', 'data/training_runs.json', 'Training'),
        ('output/experiment_record/04_training/analyses/completed_20261004/aggregates.json', 'data/training_aggregates.json', 'Training'),
        ('output/experiment_record/05_augmentation/analyses/geometry_completed.json', 'data/geometric_augmentation.json', 'Augmentation'),
        ('output/experiment_record/05_augmentation/augmentation_status.json', 'data/robustness_augmentation.json', 'Augmentation'),
        ('output/experiment_record/06_architecture/architecture_status.json', 'data/architecture.json', 'Architecture'),
        ('output/experiment_record/01_preprocessing/final_comparison_v2/decision.json', 'data/preprocessing_decision.json', 'Preprocessing'),
        ('.local/final_ablation_20261018/ablation_results.json', 'data/final_ablation.json', 'Final fusion'),
        ('.local/final_error_analysis_20261018/analysis_results.json', 'data/final_error_analysis.json', 'Error analysis'),
        ('.local/continuation_screen_20261019/results.json', 'data/continuation_screen.json', 'Rejected extensions'),
        ('.local/continuation_screen_20261019/mid40/fusion_results.json', 'data/mid40_fusion.json', 'Rejected extensions'),
        ('.local/best_model_bundle_20261018/replay_verification.json', 'data/cpu_replay_verification.json', 'Verification'),
    ]
    for name in ('results.json', 'protocol.json', 'audit.json', 'summary.csv', 'price_slices.csv', 'learning_curves.png'):
        sources.append((f'output/experiment_record/05_augmentation/photometric_screen_20261019/{name}', f'photometric/{name}', 'Photometric screen'))
    for name in ('summary.csv', 'seed_results.csv'):
        sources.append((f'output/experiment_record/01_preprocessing/final_comparison_v2/{name}', f'tables/preprocessing_{name}', 'Preprocessing'))
    for name in ('ablation_summary.csv', 'table_01_ablation_results.png', 'table_01_ablation_results.svg', 'figure_01_ablation_comparison.png', 'figure_01_ablation_comparison.svg', 'verification.json'):
        sources.append((f'.local/final_ablation_20261018/{name}', f'ablation/{name}', 'Final fusion'))
    for prefix in ('figure_01_prediction_and_residuals', 'figure_02_price_band_bias_and_error_share', 'figure_03_previous_vs_final'):
        for ext in ('png', 'svg'):
            sources.append((f'.local/final_error_analysis_20261018/{prefix}.{ext}', f'error_analysis/{prefix}.{ext}', 'Error analysis'))
    sources.extend([
        ('output/experiment_record/04_training/analyses/completed_20261004/comparison.png', 'figures/training_comparison.png', 'Training'),
        ('output/experiment_record/03_traditional/baseline_summary/validation_diagnostics.png', 'figures/traditional_diagnostics.png', 'Traditional baseline'),
        ('output/experiment_record/03_traditional/baseline_summary/keypoint_distribution.png', 'figures/sift_keypoint_distribution.png', 'Traditional baseline'),
        ('output/experiment_record/05_augmentation/analyses/error_review_20261005/price_and_error.png', 'figures/alexnet_price_and_error.png', 'Historical AlexNet analysis'),
        ('output/experiment_record/05_augmentation/analyses/visual_coverage_20261005/coverage.png', 'figures/training_visual_coverage.png', 'Exploratory coverage'),
        ('output/experiment_record/05_augmentation/analyses/visual_coverage_20261005/coverage.json', 'data/training_visual_coverage.json', 'Exploratory coverage'),
    ])
    for stage in ('01_preprocessing', '04_training', '05_augmentation', '06_architecture'):
        for path in sorted((ROOT / 'output/experiment_record' / stage / 'runs').rglob('learning_curves.png')):
            rel = path.relative_to(ROOT).as_posix()
            sources.append((rel, 'archive_curves/' + path.relative_to(ROOT / 'output/experiment_record').as_posix(), 'Per-run learning curves'))
        for path in sorted((ROOT / 'output/experiment_record' / stage).glob('*status.json')):
            sources.append((path.relative_to(ROOT).as_posix(), f'data/stages/{stage}/{path.name}', 'Stage decisions'))
    final_sources = {
        'naflex': 'siglip2_naflex_576_kernel/20261014T020905Z_e457778a',
        'pe_l_pls': 'pe_core_multiview_price_metric_kernel/20261014T051116Z_13d3ea16',
        'pe_g_pca': 'pe_giant_whole_pca_bandwidth_kernel/20261015T033416Z_4b848f75',
    }
    for label, run in final_sources.items():
        for name in ('registration.json', 'results.json'):
            sources.append((f'.local/best_model_bundle_20261018/output/experiment_record/19_image_embedding_reference/{run}/{name}', f'data/final_branches/{label}_{name}', 'Frozen branch definition'))
    for s, t, c in sources:
        collect(s, t, c)

    tables = json.loads((HERE / 'tables/report_tables.json').read_text(encoding='utf-8'))['tables']
    names = dict(overall='表I：方法总览', OVERVIEW='表II：实验总览', AUGMENT='表III：历史增强单项', PHOTO='表IV：补充亮度与对比度', architecture='表V：结构筛选', ABLATION='表VI：最终融合消融', ERROR_BANDS='表VII：最终模型分价位误差')
    notes = {
        'overall': '相同1,601张验证图片。±为三种子样本标准差；其余为已选单次配置。MAE/尾部MSE为独立模型指标均值，不是种子集成。方法间差异不是单模块消融。',
        'OVERVIEW': '只列已完成阶段。所有数值来自对应阶段，不将候选视为最终提交。',
        'AUGMENT': '种子2026历史单项；全部独立对照无增强128,268.67，不叠加翻转。翻转三种子均值126,810.01，2/3改善；擦除128,300.21，1/3改善。',
        'PHOTO': '2026-10-19，新对照131,389.94；相同初始化，单种子2026，均40轮。U(0.9,1.1)，概率1。对比度总体−1.93%但尾部+17.14%。不可与旧三种子均值配对；未更改最终融合。',
        'architecture': '单种子2026，独立改变AlexNet结构。不是完整VGG/Inception/ResNet的模型比较。',
        'ABLATION': '移除分支后只用训练OOF重拟合剩余权重。基础模型已选参，meta不是完全嵌套无偏CV。删除整条分支不等于PCA/PLS单操作消融；PE-L边际结果不一致。',
        'ERROR_BANDS': '价格/MAE/偏差单位为千美元；偏差=预测−真实。尾部来自训练90%分位1,300。该模型不是AlexNet。',
    }
    table_sections = []
    for key in ('overall', 'OVERVIEW', 'AUGMENT', 'PHOTO', 'architecture', 'ABLATION', 'ERROR_BANDS'):
        rows = tables[key]
        markup = '<table>' + ''.join('<tr>' + ''.join(f'<{"th" if i == 0 else "td"}>{html.escape(str(v))}</{"th" if i == 0 else "td"}>' for v in row) + '</tr>' for i, row in enumerate(rows)) + '</table>'
        section = f'<section id="{key}"><h2>{names[key]}</h2><p>{notes[key]}</p>{markup}</section>'
        table_sections.append(section)
        (HERE / f'tables/{key}.html').write_text(page(names[key], section), encoding='utf-8')

    # Recompute all derived chart values from full-precision evidence, not rounded tables.
    photo = read('output/experiment_record/05_augmentation/photometric_screen_20261019/results.json')['rows']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, field, title in zip(axes, ('mse', 'tail_mse'), ('Overall validation MSE', 'High-price tail MSE (>1,300)')):
        values = [r[field] for r in photo]
        ax.bar(['Control', 'Brightness', 'Contrast'], values, color=['#718096', '#4e79a7', '#e79a3b'])
        for i, val in enumerate(values):
            ax.text(i, val, f'{val:,.0f}', ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, max(values) * 1.16)
        ax.set_title(title)
        ax.set_ylabel('MSE ((thousand USD)^2)')
    fig.suptitle('Paired photometric screen | seed 2026 | fixed validation n=1,601')
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(HERE / f'figures/photometric_tradeoff.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    ablation = read('.local/final_ablation_20261018/ablation_results.json')
    assert abs(ablation['results'][-1]['validation']['mse'] - 52201.21126766699) < 1e-7
    assert abs(photo[2]['delta_mse_percent'] + 1.9251143798408172) < 1e-9

    gallery = []
    captions = {
        'figures/photometric_tradeoff.png': '补充光度增强：总体和尾部结论不同；单种子筛选，不代表最终编码器增强有效。',
        'photometric/learning_curves.png': '同批三组AlexNet学习曲线。训练模式包含Dropout/增强；验证关闭。不要直接混用缩放训练loss与原价MSE。',
        'ablation/figure_01_ablation_comparison.png': '最终分支消融：固定基预测，移除后训练OOF重拟合权重。无独立PCA/PLS因果归因。',
        'error_analysis/figure_01_prediction_and_residuals.png': '最终融合预测与残差，固定开发验证集，不是全8,000拟合后的训练误差。',
        'error_analysis/figure_02_price_band_bias_and_error_share.png': '高价159张仅占9.93%，贡献47.13%的平方误差；偏差=预测−真实。',
        'error_analysis/figure_03_previous_vs_final.png': '最终模型与历史前序方案的分价位比较；前序分组值来自归档，不是本轮重跑。',
        'figures/training_comparison.png': '历史训练策略比较；同一固定划分，注意单种子与复核组区别。',
        'figures/alexnet_price_and_error.png': '旧水平翻转AlexNet的诊断，不要误标为最终融合结果。',
        'figures/training_visual_coverage.png': '探索性训练侧覆盖抽样，每价位100张，共400张；非全量普查或独立专家标注。',
        'figures/traditional_diagnostics.png': 'SIFT–BoVW–Ridge传统路线诊断，非最终模型。',
        'figures/sift_keypoint_distribution.png': 'SIFT特征覆盖诊断；关键点数量不是预测准确率。',
    }
    for file, caption in captions.items():
        assert (HERE / file).is_file(), file
        gallery.append(f'<figure><a href="{file}"><img loading="lazy" src="{file}" alt="{html.escape(caption)}"></a><figcaption>{caption}</figcaption></figure>')
    links = ''.join(f'<li><a href="{r["file"]}">{r["file"]}</a> — {r["category"]}</li>' for r in records)
    intro = '''<h1>房价预测报告素材包</h1><p>截至2026-10-19。用途：两人协作写report，不是最终代码提交目录。</p>
    <p><a href="https://docs.google.com/document/d/1osyQDBjpkzbW7s0LzdxijcCU7Aq3U85L345mT6FeMSk/edit">中文草稿（云端为正文主版本）</a> · <a href="draft_zh.html">本次中文正文快照</a> · <a href="tables/report_tables.json">表格显示值JSON</a> · <a href="manifest.json">来源和哈希清单</a></p>
    <p><strong>分清三类分数：</strong>开发验证52,201.21；训练侧meta-crossfit 52,210.22；历史公榜47,291.34608。固定验证n=1,601，训练n=6,399；最终提交用全部8,000标签重拟合。</p>
    <p>新光度实验须使用同批新对照；不覆盖旧实验。未重新训练最终融合。表中MSE单位为(千美元)²，MAE为千美元。所有表格为已完成实验快照，非实时联动表。</p>'''
    body = intro + ''.join(table_sections) + '<h2>推荐报告图与图注</h2>' + ''.join(gallery) + '<h2>完整聚合证据与历史曲线</h2><ul>' + links + '</ul>'
    (HERE / 'index.html').write_text(page('报告素材包', body), encoding='utf-8')
    (HERE / 'CAPTIONS_ZH.txt').write_text('\n\n'.join(f'{k}\n{v}' for k, v in captions.items()), encoding='utf-8')
    tracked = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest = dict(as_of='2026-10-19', source_commit=tracked,
        best_model_source_commit='cceac4082c543d55af51673fd87662fa767be14c',
        split_sha256='c9f91f12b49978b77bc434d8a7ab878f1ee8b8e73bf96f6cf4474767a0c94213',
        frozen_model_run='20261015T034010Z_0c94a543', files=records,
        exclusions=['raw train/test images', 'sample-image collages and occlusion images', 'per-image labels/predictions', 'model weights and feature caches', 'credentials', 'design folder'],
        sanitization='JSON absolute local roots replaced with $REPO/$GROUP; case_selection removed. Source and packed hashes both retained.',
        caveats=['Supplementary photometric results use a fresh paired control and one seed.', 'No nested unbiased CV claim; no significance test.', 'Report assets are not the final code submission.', 'Curves are archival runs; use stage decisions, not every curve as a selected result.'])
    # Stable checkout hashes on Windows and Linux. SVG path whitespace is cosmetic.
    for p in HERE.rglob('*'):
        if p.is_file() and p.suffix in ('.json', '.html', '.md', '.txt', '.csv', '.svg', '.py'):
            value = p.read_text(encoding='utf-8-sig').replace('\r\n', '\n')
            if p.suffix == '.svg':
                value = '\n'.join(line.rstrip() for line in value.splitlines()) + '\n'
            p.write_text(value, encoding='utf-8', newline='\n')
    for r in records:
        r['packed_sha256'] = sha(HERE / r['file'])
        r['bytes'] = (HERE / r['file']).stat().st_size
    manifest['sanitization'] += ' Text stored as UTF-8 LF; SVG trailing whitespace removed.'
    copied = {r['file'] for r in records}
    manifest['derived_files'] = [dict(file=p.relative_to(HERE).as_posix(), sha256=sha(p), bytes=p.stat().st_size)
        for p in sorted(HERE.rglob('*')) if p.is_file() and p.name != 'manifest.json'
        and p.relative_to(HERE).as_posix() not in copied and '__pycache__' not in p.parts]
    (HERE / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    for r in records:
        assert sha(HERE / r['file']) == r['packed_sha256']
    print(json.dumps(dict(copied_files=len(records), tables=len(tables), total_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())), indent=2))


def page(title, body):
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + html.escape(title) + '''</title><style>
    body{font:16px/1.7 "Microsoft YaHei",Arial,sans-serif;max-width:1150px;margin:32px auto;padding:0 24px;color:#243449;background:#fff}h1,h2{color:#16365c}section{margin:42px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #cbd5e1;padding:9px 12px;text-align:left}th{background:#e4edf7}tr:nth-child(even){background:#f8fafc}figure{margin:30px 0;padding:18px;border:1px solid #dce4ed}img{max-width:100%;height:auto}figcaption{margin-top:10px;color:#46576e}a{color:#175ca8}li{overflow-wrap:anywhere}@media print{section,figure{break-inside:avoid}body{max-width:none;margin:0}th{background:#e4edf7}}
    </style>''' + body + '</html>'


if __name__ == '__main__':
    main()
