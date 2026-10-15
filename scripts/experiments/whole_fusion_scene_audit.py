"""Paired train-only meta-crossfit diagnosis for original versus whole-frame PE-G."""
from pathlib import Path
import time
import uuid
import numpy as np
import pandas as pd
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from current_fusion_scene_audit import stats
from training_scene_index import FEATURES

CURRENT = '15_fixed_ensembles/pe_giant_whole_incremental_fusion/20261014T082727Z_c5152cef'
PREVIOUS = '15_fixed_ensembles/pe_giant_incremental_fusion/20261014T065752Z_88efe38c'
GEOMETRY = '09_data_quality/pe_giant_frame_geometry/20261014T081601Z_cdd4168d/geometry_and_training_errors.csv'


def main():
    root = local_path('records_root')
    split = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    train = split.loc[split.partition.eq('train')].reset_index(drop=True)
    assert len(train) == 6399 and train.imageid.is_unique
    annotation_path = REPOSITORY / '.local/artifacts/training_scene_index/training_scene_index.csv'
    annotations = pd.read_csv(annotation_path, dtype={'imageid': str, 'group_id': str})
    assert annotations.imageid.is_unique and set(annotations.imageid) == set(train.imageid)
    annotations = annotations.set_index('imageid').loc[train.imageid].reset_index()
    np.testing.assert_array_equal(annotations.price, train.price)
    assert annotations.group_id.tolist() == train.group_id.tolist()
    geometry = pd.read_csv(root / GEOMETRY, dtype={'imageid': str, 'group_id': str})
    assert geometry.imageid.tolist() == train.imageid.tolist()
    np.testing.assert_array_equal(geometry.price, train.price)
    tables = {}
    for label, reference in [('original', PREVIOUS), ('whole', CURRENT)]:
        public = REPOSITORY / 'output/experiment_record' / reference
        assert read_json(public / 'status.json')['status'] == 'completed'
        table = pd.read_csv(root / reference / 'meta_crossfit_predictions.csv', dtype={'imageid': str})
        assert table.imageid.tolist() == train.imageid.tolist()
        assert np.isfinite(table.augmented_meta_prediction).all()
        tables[label] = table
    np.testing.assert_allclose(tables['whole'].baseline_meta_prediction,
        tables['original'].augmented_meta_prediction, rtol=1e-8, atol=1e-6)
    relative = Path('09_data_quality/pe_giant_whole_scene_audit') / (
        time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    out, public = root / relative, REPOSITORY / 'output/experiment_record' / relative
    out.mkdir(parents=True, exist_ok=False)
    write_json(public / 'registration.json', {'source_sha256': digest(__file__),
        'statistics_helper_sha256': digest(Path(__file__).with_name('current_fusion_scene_audit.py')),
        'split_sha256': digest(local_path('split_path')), 'annotation_index_sha256': digest(annotation_path),
        'geometry_sha256': digest(root / GEOMETRY),
        'prediction_sha256': {label: digest(root / reference / 'meta_crossfit_predictions.csv')
            for label, reference in [('original', PREVIOUS), ('whole', CURRENT)]},
        'scope': '6399 training meta-crossfit rows; paired same fold baseline verified; no refitting or test access',
        'caveat': 'Base selection bias remains; geometry ideal crop proxy; incomplete low/middle annotations; descriptive slices, no causal claims',
        'test_images_used': False, 'validation_images_used': False})
    price = train.price.to_numpy()
    reviewed = annotations.annotation_status.eq('已审阅').to_numpy()
    retained = geometry.center_square_retained_area_fraction.to_numpy()
    scopes = [('all', np.ones(len(train), dtype=bool)), ('low_le300', price <= 300),
        ('middle_300_1300', (price > 300) & (price <= 1300)), ('high_gt1300', price > 1300)]
    rows, paired = [], []
    for scope, tier in scopes:
        slices = [('all', tier)]
        slices += [('scene:' + scene, tier & reviewed & annotations.scene_primary.eq(scene).to_numpy())
            for scene in sorted(annotations.scene_primary.unique())]
        slices += [(feature + ':' + value, tier & reviewed & annotations[feature].eq(value).to_numpy())
            for feature in FEATURES for value in ('是', '否', '未知')]
        slices += [('geometry:' + label, tier & mask) for label, mask in [
            ('retained_le_half', retained <= .5),
            ('retained_half_to_two_thirds', (retained > .5) & (retained <= 2/3)),
            ('retained_two_thirds_to_90pct', (retained > 2/3) & (retained <= .9)),
            ('retained_over_90pct', retained > .9)]]
        for category, mask in slices:
            if not mask.any():
                continue
            old = tables['original'].augmented_meta_prediction.to_numpy()[mask]
            new = tables['whole'].augmented_meta_prediction.to_numpy()[mask]
            target = price[mask]
            for name, prediction in [('original', old), ('whole', new)]:
                rows.append(stats(train.loc[mask], prediction, name, scope, category))
            old_sq, new_sq = (old-target)**2, (new-target)**2
            paired.append({'scope': scope, 'category': category, 'n': int(mask.sum()),
                'previous_mse': float(old_sq.mean()), 'current_mse': float(new_sq.mean()),
                'mse_delta': float((new_sq-old_sq).mean()),
                'relative_gain': float(1-new_sq.mean()/old_sq.mean()),
                'previous_bias': float((old-target).mean()), 'current_bias': float((new-target).mean()),
                'n_squared_error_improved': int((new_sq < old_sq).sum()),
                'previous_contribution_to_total_mse': float(old_sq.sum()/len(train)),
                'current_contribution_to_total_mse': float(new_sq.sum()/len(train))})
    pd.DataFrame(rows).to_csv(public / 'category_metrics.csv', index=False, encoding='utf-8-sig')
    comparison = pd.DataFrame(paired)
    comparison.to_csv(public / 'paired_comparison.csv', index=False, encoding='utf-8-sig')
    detail = annotations.copy()
    detail['retained_fraction'] = retained
    for label in tables:
        detail[label+'_prediction'] = tables[label].augmented_meta_prediction.to_numpy()
        detail[label+'_error'] = detail[label+'_prediction'] - price
    detail.to_csv(out / 'training_predictions_and_annotations.csv', index=False, encoding='utf-8-sig')
    highlights = comparison[comparison.category.isin(['all', 'scene:地面外观', 'visible_pool:是'])]
    write_json(public / 'summary.json', {'paired_baseline_verified': True,
        'descriptive_highlights': highlights.to_dict(orient='records'), 'test_images_used': False})
    (public / '实验报告.md').write_text('# PE-G 整图增量：训练价位与场景诊断\n\n'
        '比较原 PE-G 和原视图+整图的训练元模型交叉拟合预测，逐 ID/价格/分组对齐，并复现本轮基线。'
        '全部 6399 图分价位等权诊断；场景只对已审阅子集统计，未知单列，不外推为全量频数。'
        'paired_comparison.csv 包含配对 MSE 变化、偏差与总 MSE 贡献；场景/属性可重叠，贡献不能相加。'
        '几何仅理想裁剪代理，价位与场景混杂，不能证明边缘像素因果价值。'
        '基础模型已选参，非完全嵌套无偏 CV；没有测试或验证图参与诊断，也未改变现有权重。\n', encoding='utf-8')
    write_json(public / 'status.json', {'status': 'completed'})
    print(public, flush=True)
    print(highlights[['scope', 'category', 'n', 'previous_mse', 'current_mse', 'current_bias']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
