"""Preregister PE-Core-G against the latest PLS fusion, then evaluate OOF increment."""
from pathlib import Path
import argparse
import os
import time
import uuid
import numpy as np
import pandas as pd
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from common.metrics import save_predictions
from encoder_oof_fusion import fit_weights, meta_cv, resolve_encoder

BASE = '15_fixed_ensembles/pe_pls_incremental_fusion/20261014T063915Z_573c1b3a'
SPEC = {'name': 'pe_core_g14_448_kernel', 'family': 'pe_core_giant_features',
    'model': 'timm/vit_pe_core_gigantic_patch14_448.fb', 'revision': 'fb096c09d13bd2e3827aed4dbdffb8a6b8e0c10b'}


def preregister(baseline_reference=BASE, new_encoder=None, experiment_name='pe_giant_incremental_fusion'):
    if not experiment_name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in experiment_name):
        raise ValueError('Invalid experiment name')
    spec = SPEC if new_encoder is None else new_encoder
    if not isinstance(spec, dict) or not {'name', 'family', 'model', 'revision'} <= set(spec):
        raise ValueError('A pinned frozen visual kernel specification is required')
    relative = Path('15_fixed_ensembles') / experiment_name / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    out = local_path('records_root') / relative
    public = REPOSITORY / 'output/experiment_record' / relative
    out.mkdir(parents=True, exist_ok=False)
    write_json(public / 'registration.json', {'baseline': baseline_reference,
        'baseline_ensemble_sha256': digest(local_path('records_root') / baseline_reference / 'ensemble.json'),
        'new_encoder': spec, 'source_sha256': digest(__file__),
        'optimizer_source_sha256': digest(Path(__file__).with_name('encoder_oof_fusion.py')),
        'split_sha256': digest(local_path('split_path')), 'weight_zero_tolerance': 1e-12,
        'gate': {'minimum_meta_cv_relative_gain': .01, 'minimum_folds_won': 2, 'validation_improvement_required': True},
        'selection': 'Add exactly one preregistered frozen visual kernel to the explicitly named fixed baseline; convex weights from training OOF only',
        'caveat': 'Base hyperparameters previously selected; meta crossfit is not fully nested unbiased CV',
        'test_images_used': False, 'mllm_components': False, 'automatic_test_export': False})
    write_json(public / 'status.json', {'status': 'preregistered'})
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    print(out, flush=True)
    return out


def run(out):
    out = out.resolve()
    root = local_path('records_root')
    public = REPOSITORY / 'output/experiment_record' / out.relative_to(root)
    registration = read_json(public / 'registration.json')
    assert read_json(public / 'status.json')['status'] == 'preregistered'
    assert digest(__file__) == registration['source_sha256']
    assert digest(Path(__file__).with_name('encoder_oof_fusion.py')) == registration['optimizer_source_sha256']
    assert digest(local_path('split_path')) == registration['split_sha256']
    def status(value, **extra):
        write_json(public / 'status.json', {'status': value, 'pid': os.getpid(), **extra})
    try:
        status('fitting_meta')
        frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
        train, valid = frame[frame.partition.eq('train')], frame[frame.partition.eq('validation')]
        assert len(train) == 6399 and len(valid) == 1601 and not set(train.group_id) & set(valid.group_id)
        reference = root / registration['baseline']
        assert digest(reference / 'ensemble.json') == registration['baseline_ensemble_sha256']
        baseline = read_json(reference / 'ensemble.json')
        models = baseline['models'] + [resolve_encoder(registration['new_encoder'])]
        z, v, heads, hashes = [], [], [], []
        for i, model in enumerate(models):
            source = root / model
            assert read_json(REPOSITORY / 'output/experiment_record' / model / 'status.json')['status'] == 'completed'
            heads.append(digest(source / 'head.npz'))
            hashes.append(digest(source / 'predictions.csv'))
            if i < len(baseline['models']):
                assert heads[-1] == baseline['model_sha256'][i] and hashes[-1] == baseline['prediction_sha256'][i]
            pt = pd.read_csv(source / 'selected_training_oof.csv', dtype={'imageid': str})
            pv = pd.read_csv(source / 'predictions.csv', dtype={'imageid': str})
            assert pt.imageid.tolist() == train.imageid.tolist() and pv.imageid.tolist() == valid.imageid.tolist()
            np.testing.assert_array_equal(pv.price, valid.price)
            z.append(pt.oof_prediction.to_numpy() / 1000)
            v.append(pv.prediction.to_numpy())
        z, v = np.column_stack(z), np.column_stack(v)
        assert np.isfinite(z).all() and np.isfinite(v).all()
        y = train.price.to_numpy() / 1000
        folds, bmeta, nmeta = meta_cv(z, y, train.group_id.to_numpy(), len(baseline['models']), 1e-12)
        weights = fit_weights(z, y, 1e-12)
        fixed = v[:, :-1] @ np.asarray(baseline['weights'])
        previous = pd.read_csv(reference / 'predictions.csv', dtype={'imageid': str})
        assert previous.imageid.tolist() == valid.imageid.tolist()
        np.testing.assert_allclose(fixed, previous.prediction, rtol=1e-7, atol=1e-4)
        metrics = save_predictions(out, valid, v @ weights, 1300)
        bmse, nmse = [float(np.mean((p - y) ** 2) * 1e6) for p in (bmeta, nmeta)]
        baseline_valid = float(np.mean((fixed - valid.price.to_numpy()) ** 2))
        won = sum(f['augmented_mse'] < f['baseline_mse'] for f in folds)
        eligible = nmse < bmse * .99 and won >= 2 and metrics['mse'] < baseline_valid
        results = {**metrics, 'baseline_validation_mse': baseline_valid, 'baseline_meta_cv_mse': bmse,
            'meta_cv_mse': nmse, 'meta_relative_gain': 1 - nmse / bmse, 'folds_won': won,
            'weights': weights.tolist(), 'export_eligible': bool(eligible)}
        write_json(public / 'meta_folds.json', folds)
        write_json(public / 'results.json', results)
        pd.DataFrame({'imageid': train.imageid, 'baseline_meta_prediction': bmeta * 1000,
            'augmented_meta_prediction': nmeta * 1000}).to_csv(out / 'meta_crossfit_predictions.csv', index=False)
        write_json(out / 'ensemble.json', {'models': models, 'weights': weights.tolist(), 'model_sha256': heads,
            'prediction_sha256': hashes, 'source_sha256': digest(__file__), 'split_sha256': registration['split_sha256'],
            'test_images_used': False, 'mllm_components': False})
        (public / '实验报告.md').write_text(f'# PE-Core-G 增量融合\n\n固定基线验证 {baseline_valid:.5f}，新融合 {metrics["mse"]:.5f}。\n\n元模型交叉拟合 {bmse:.5f} → {nmse:.5f}，改善 {1-nmse/bmse:.3%}，获胜 {won}/3；门槛通过：{eligible}。\n\n只新增一个固定 PE-Core-G 视觉编码器，训练 OOF 拟合凸权重。基础参数选择偏差仍存在，未使用测试图。公榜未验证，结果保存后再决定是否全量导出。\n', encoding='utf-8')
        status('completed', export_eligible=bool(eligible))
        print(results, flush=True)
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['preregister', 'run'], required=True)
    parser.add_argument('--protocol-dir', type=Path)
    parser.add_argument('--baseline', default=BASE)
    parser.add_argument('--new-encoder-spec', type=Path)
    parser.add_argument('--experiment-name', default='pe_giant_incremental_fusion')
    args = parser.parse_args()
    if args.phase == 'preregister':
        preregister(args.baseline, read_json(args.new_encoder_spec) if args.new_encoder_spec else None, args.experiment_name)
    elif args.protocol_dir:
        run(args.protocol_dir)
    else:
        parser.error('run requires protocol-dir')
