"""Describe current base OOF and meta-crossfit errors with existing train labels."""
from pathlib import Path
import time
import uuid
import numpy as np
import pandas as pd
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from training_scene_index import FEATURES

CURRENT = '15_fixed_ensembles/pe_pls_incremental_fusion/20261014T063915Z_573c1b3a'
PREVIOUS = '15_fixed_ensembles/pe_multiview_giant_oof_fusion/20261014T044041Z_ee76e803'


def stats(frame, prediction, model, scope, label):
    y = frame.price.to_numpy()
    e = np.asarray(prediction) - y
    n = len(y)
    if not n:
        return None
    return {'model': model, 'scope': scope, 'category': label, 'n': n,
        'price_mean': float(y.mean()), 'price_median': float(np.median(y)),
        'mse': float(np.mean(e ** 2)), 'mae': float(np.mean(abs(e))), 'bias': float(e.mean()),
        'abs_error_le100': int((abs(e) <= 100).sum()),
        'abs_error_100_300': int(((abs(e) > 100) & (abs(e) <= 300)).sum()),
        'abs_error_300_500': int(((abs(e) > 300) & (abs(e) <= 500)).sum()),
        'abs_error_500_1000': int(((abs(e) > 500) & (abs(e) <= 1000)).sum()),
        'abs_error_gt1000': int((abs(e) > 1000).sum())}


def main():
    root = local_path('records_root')
    split = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    train = split[split.partition.eq('train')].reset_index(drop=True)
    indexpath = REPOSITORY / '.local/artifacts/training_scene_index/training_scene_index.csv'
    index = pd.read_csv(indexpath, dtype={'imageid': str, 'group_id': str})
    assert index.imageid.is_unique and set(index.imageid) == set(train.imageid)
    index = index.set_index('imageid').loc[train.imageid].reset_index()
    np.testing.assert_array_equal(index.price, train.price)
    assert index.group_id.tolist() == train.group_id.tolist()
    ensemble = read_json(root / CURRENT / 'ensemble.json')
    predictions, provenance = {}, []
    for i, (model, weight) in enumerate(zip(ensemble['models'], ensemble['weights'])):
        if weight == 0:
            continue
        source = root / model
        pt = pd.read_csv(source / 'selected_training_oof.csv', dtype={'imageid': str})
        assert pt.imageid.tolist() == train.imageid.tolist()
        name = Path(model).parts[-2]
        predictions[name] = pt.oof_prediction.to_numpy()
        provenance.append({'model': model, 'oof_sha256': digest(source / 'selected_training_oof.csv')})
    for name, reference in [('previous_meta_crossfit', PREVIOUS), ('current_meta_crossfit', CURRENT)]:
        path = root / reference / 'meta_crossfit_predictions.csv'
        table = pd.read_csv(path, dtype={'imageid': str})
        assert table.imageid.tolist() == train.imageid.tolist()
        predictions[name] = table.augmented_meta_prediction.to_numpy()
        provenance.append({'model': reference, 'oof_sha256': digest(path)})
    relative = Path('09_data_quality/current_fusion_scene_audit') / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    out, public = root / relative, REPOSITORY / 'output/experiment_record' / relative
    out.mkdir(parents=True, exist_ok=False)
    write_json(public / 'registration.json', {'source_sha256': digest(__file__), 'split_sha256': digest(local_path('split_path')),
        'annotation_index_sha256': digest(indexpath), 'models': provenance,
        'scope': 'Existing training labels and annotated scenes, base OOF or meta crossfit; no refitting or model selection',
        'caveat': 'Base hyperparameters already selected; meta crossfit is not fully nested. Low/middle scene annotations are incomplete; unknown is not negative.',
        'test_images_used': False})
    rows = []
    reviewed = index.annotation_status.eq('已审阅').to_numpy()
    price = train.price.to_numpy()
    tiers = [('all', np.ones(len(train), dtype=bool)), ('low_le300', price <= 300),
        ('middle_300_1300', (price > 300) & (price <= 1300)), ('high_gt1300', price > 1300)]
    for name, prediction in predictions.items():
        assert np.isfinite(prediction).all()
        for tier, mask in tiers:
            row = stats(train.loc[mask], prediction[mask], name, tier, 'all')
            if row:
                rows.append(row)
            for scene in sorted(index.scene_primary.unique()):
                scene_mask = mask & reviewed & index.scene_primary.eq(scene).to_numpy()
                row = stats(train.loc[scene_mask], prediction[scene_mask], name, tier, 'scene:' + scene)
                if row:
                    rows.append(row)
            for feature in FEATURES:
                for state in ('是', '否', '未知'):
                    feature_mask = mask & reviewed & index[feature].eq(state).to_numpy()
                    row = stats(train.loc[feature_mask], prediction[feature_mask], name, tier, feature + ':' + state)
                    if row:
                        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(public / 'category_metrics.csv', index=False, encoding='utf-8-sig')
    errors = np.column_stack([p - price for p in predictions.values()])
    write_json(public / 'residual_correlations.json', {'models': list(predictions), 'correlations': np.corrcoef(errors.T).tolist()})
    detail = index.copy()
    for name, prediction in predictions.items():
        detail[name + '_prediction'] = prediction
        detail[name + '_error'] = prediction - price
    detail.to_csv(out / 'training_predictions_and_annotations.csv', index=False, encoding='utf-8-sig')
    ground_high = detail[(detail.price > 1300) & detail.scene_primary.eq('地面外观')]
    ground_high.sort_values('current_meta_crossfit_error').to_csv(out / 'high_ground_examples.csv', index=False, encoding='utf-8-sig')
    summary = table[table.model.isin(['previous_meta_crossfit', 'current_meta_crossfit']) & table.category.isin(['all', 'scene:地面外观', 'visible_pool:是'])]
    summary.to_csv(public / 'meta_comparison.csv', index=False, encoding='utf-8-sig')
    write_json(public / 'status.json', {'status': 'completed'})
    (public / '实验报告.md').write_text('# 当前融合价位与场景诊断\n\n训练侧 6399 条预测按 imageid 与原始标签、既有注释一一核对。基础模型使用训练 OOF；新旧融合使用元模型交叉拟合预测，没有用元模型训练拟合误差替代。基础选参偏差仍存在。\n\ncategory_metrics.csv 包含每个价位与场景的数量、真实价格分布、MSE、MAE、有符号偏差及五个绝对误差档；meta_comparison.csv 提供新旧融合对照。低中价视觉审阅未覆盖全量，未知单列；标签重叠，不能相加或作因果解释。逐图预测与高价地面外观案例仅保存在本地。没有验证场景标签或测试数据参与诊断。\n', encoding='utf-8')
    print(public, flush=True)
    print(summary[['model', 'scope', 'category', 'n', 'mse', 'bias']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
