"""One preregistered verified PCA increment; unchanged existing convex fusion mathematics."""
from pathlib import Path
import argparse
import time
import uuid
import numpy as np
import pandas as pd
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from common.metrics import save_predictions
from encoder_oof_fusion import fit_weights, meta_cv

BASE = '15_fixed_ensembles/pe_multiview_giant_oof_fusion/20261014T044041Z_ee76e803'
PLS = '19_image_embedding_reference/pe_core_multiview_price_metric_kernel/20261014T051116Z_13d3ea16'


def main(baseline_reference=BASE, added_reference=PLS, experiment_name='pe_pls_incremental_fusion'):
    if not experiment_name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in experiment_name):
        raise ValueError('Invalid experiment name')
    root = local_path('records_root')
    base = read_json(root / baseline_reference / 'ensemble.json')
    added_registration = read_json(REPOSITORY / 'output/experiment_record' / added_reference / 'registration.json')
    if added_registration.get('method') != 'Fold-fitted PCA partial-whitening image RBF with training-median bandwidth calibration' or added_registration.get('supervised_encoder') is not False or added_registration.get('test_images_used') is not False:
        raise ValueError('Expected a registered train-only frozen visual PCA kernel')
    audit=read_json(REPOSITORY/'output/experiment_record'/added_reference/'independent_artifact_audit.json')
    assert audit['passed'] and audit['head_sha256']==digest(root/added_reference/'head.npz') and audit['predictions_sha256']==digest(root/added_reference/'predictions.csv')
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    train = frame[frame.partition.eq('train')]
    valid = frame[frame.partition.eq('validation')]
    assert len(train) == 6399 and len(valid) == 1601
    assert not set(train.group_id) & set(valid.group_id)
    models = base['models'] + [added_reference]
    z, v, hashes, heads = [], [], [], []
    for i, model in enumerate(models):
        source = root / model
        public_source = REPOSITORY / 'output/experiment_record' / model
        assert read_json(public_source / 'status.json')['status'] == 'completed'
        hashes.append(digest(source / 'predictions.csv'))
        heads.append(digest(source / 'head.npz'))
        if i < len(base['models']):
            assert hashes[-1] == base['prediction_sha256'][i]
            assert heads[-1] == base['model_sha256'][i]
        pt = pd.read_csv(source / 'selected_training_oof.csv', dtype={'imageid': str})
        pv = pd.read_csv(source / 'predictions.csv', dtype={'imageid': str})
        assert pt.imageid.tolist() == train.imageid.tolist()
        assert pv.imageid.tolist() == valid.imageid.tolist()
        np.testing.assert_array_equal(pv.price, valid.price)
        z.append(pt.oof_prediction.to_numpy() / 1000)
        v.append(pv.prediction.to_numpy())
    z, v = np.stack(z, axis=1), np.stack(v, axis=1)
    assert np.isfinite(z).all() and np.isfinite(v).all()
    relative = Path('15_fixed_ensembles') / experiment_name / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    out, public = root / relative, REPOSITORY / 'output/experiment_record' / relative
    out.mkdir(parents=True, exist_ok=False)
    registration = {'baseline': baseline_reference, 'baseline_ensemble_sha256': digest(root / baseline_reference / 'ensemble.json'),
        'added_model': added_reference, 'models': models, 'model_sha256': heads, 'prediction_sha256': hashes,
        'oof_sha256': [digest(root / m / 'selected_training_oof.csv') for m in models],
        'source_sha256': digest(__file__), 'optimizer_source_sha256': digest(Path(__file__).with_name('encoder_oof_fusion.py')),
        'split_sha256': digest(local_path('split_path')), 'weight_zero_tolerance': 1e-12,
        'gate': {'minimum_meta_cv_relative_gain': 0.01, 'minimum_folds_won': 2, 'require_fixed_validation_improvement': True},
        'method': 'Add exactly one preselected fold-fitted PCA kernel to the frozen baseline; convex training OOF weights',
        'selection_caveat': 'Base hyperparameters already selected; meta crossfit is not fully nested unbiased CV',
        'test_images_used': False, 'mllm_components': False, 'automatic_export': False}
    write_json(public / 'registration.json', registration)
    write_json(public / 'status.json', {'status': 'fitting'})
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    y = train.price.to_numpy() / 1000
    fixed_train = z[:, :-1] @ np.asarray(base['weights'])
    residual = np.stack([fixed_train - y, z[:, -2] - y, z[:, -1] - y], axis=1)
    slices = []
    for lo, hi in [(0, 300), (300, 700), (700, 1300), (1300, float('inf'))]:
        mask = (train.price.to_numpy() > lo) & (train.price.to_numpy() <= hi)
        slices.append({'lower_exclusive': lo, 'upper_inclusive': hi if np.isfinite(hi) else None,
            'n': int(mask.sum()), 'mse': np.mean(residual[mask] ** 2, axis=0).tolist(),
            'bias': (np.mean(residual[mask], axis=0) * 1000).tolist()})
    for row in slices:
        row['mse'] = [x * 1e6 for x in row['mse']]
    write_json(public / 'training_diagnostics.json', {'columns': ['fixed_fusion', 'last_baseline_component', 'added_pls'],
        'residual_correlations': np.corrcoef(residual.T).tolist(), 'price_slices': slices})
    folds, bmeta, nmeta = meta_cv(z, y, train.group_id.to_numpy(), len(base['models']), 1e-12)
    weights = fit_weights(z, y, 1e-12)
    fixed_valid = v[:, :-1] @ np.asarray(base['weights'])
    previous = pd.read_csv(root / baseline_reference / 'predictions.csv', dtype={'imageid': str})
    np.testing.assert_allclose(fixed_valid, previous.prediction, rtol=1e-7, atol=1e-4)
    metrics = save_predictions(out, valid, v @ weights, 1300)
    bmse, nmse = [float(np.mean((p - y) ** 2) * 1e6) for p in (bmeta, nmeta)]
    baseline_valid = float(np.mean((fixed_valid - valid.price.to_numpy()) ** 2))
    won = sum(f['augmented_mse'] < f['baseline_mse'] for f in folds)
    eligible = nmse < bmse * .99 and won >= 2 and metrics['mse'] < baseline_valid
    results = {**metrics, 'baseline_validation_mse': baseline_valid, 'baseline_meta_cv_mse': bmse,
        'meta_cv_mse': nmse, 'meta_relative_gain': 1 - nmse / bmse, 'folds_won': won,
        'weights': weights.tolist(), 'export_eligible': bool(eligible)}
    write_json(public / 'meta_folds.json', folds)
    write_json(public / 'results.json', results)
    pd.DataFrame({'imageid': train.imageid, 'baseline_meta_prediction': bmeta * 1000,
        'augmented_meta_prediction': nmeta * 1000}).to_csv(out / 'meta_crossfit_predictions.csv', index=False)
    write_json(out / 'ensemble.json', {'models': models, 'weights': weights.tolist(),
        'model_sha256': heads, 'prediction_sha256': hashes, 'split_sha256': registration['split_sha256'],
        'source_sha256': registration['source_sha256'], 'test_images_used': False, 'mllm_components': False})
    write_json(public / 'status.json', {'status': 'completed', 'export_eligible': bool(eligible)})
    (public / '实验报告.md').write_text(f'# PE PLS 增量融合\n\n固定基线验证 MSE {baseline_valid:.5f}；新融合 {metrics["mse"]:.5f}。\n\n元模型三折交叉拟合 {bmse:.5f} → {nmse:.5f}，改善 {1-nmse/bmse:.3%}，获胜 {won}/3 折；导出门槛通过：{eligible}。\n\n仅添加已经训练侧选参的 PE 三视图 PLS 核，权重由训练 OOF 学习。基础模型选参偏差仍存在；未使用测试图，未导出或提交 Kaggle。\n', encoding='utf-8')
    print(public, flush=True)
    print(results, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', default=BASE)
    parser.add_argument('--added-model', default=PLS)
    parser.add_argument('--experiment-name', default='pe_pls_incremental_fusion')
    args = parser.parse_args()
    main(args.baseline, args.added_model, args.experiment_name)
