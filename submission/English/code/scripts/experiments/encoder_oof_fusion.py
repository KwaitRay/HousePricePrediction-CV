"""Preregister and evaluate one joint fusion of generic image encoders."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from common.metrics import save_predictions

BASELINE = '15_fixed_ensembles/siglip2_multiview_metric_clip336_oof_stack/20261013T170640Z_6d0238b8'
NEW_ENCODERS = [
    {'name': 'dinov2_large_cls_patchmean_kernel', 'family': 'dinov2_features', 'model': 'facebook/dinov2-large', 'revision': '47b73eefe95e8d44ec3623f8890bd894b6ea2d6c'},
    {'name': 'siglip2_naflex_576_kernel', 'family': 'siglip2_naflex_features', 'model': 'google/siglip2-so400m-patch16-naflex', 'revision': 'cc24074f717b612951c2dead130904ab9b65a81e', 'max_num_patches': 576},
    {'name': 'pe_core_l14_336_kernel', 'family': 'pe_core_features', 'model': 'timm/vit_pe_core_large_patch14_336.fb', 'revision': 'e63206c8e3a0e9b699e40f31080eebd78fd2258e'}]


def public_path(out):
    return REPOSITORY / 'output/experiment_record' / out.relative_to(local_path('records_root'))


def fit_weights(z, y, zero_tolerance=0.0):
    start = np.full(z.shape[1], 1 / z.shape[1])
    fit = minimize(lambda w: float(np.mean((z @ w - y) ** 2)), start,
        jac=lambda w: 2 * z.T @ (z @ w - y) / len(y), bounds=[(0, 1)] * len(start),
        constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1, 'jac': lambda w: np.ones_like(w)}],
        method='SLSQP', options={'ftol': 1e-12, 'maxiter': 1000})
    if not fit.success or not np.isfinite(fit.x).all() or (fit.x < 0).any() or abs(fit.x.sum() - 1) > 1e-8:
        raise ValueError('Invalid convex optimization result')
    weights = fit.x.copy()
    if zero_tolerance:
        if not 0 < zero_tolerance <= 1e-12:
            raise ValueError('Only preregistered numerical-noise removal is supported')
        weights[weights <= zero_tolerance] = 0
        weights /= weights.sum()
        # Verify the training price formula before any validation evaluation.
        np.testing.assert_allclose(z @ weights * 1000, z @ fit.x * 1000, rtol=0, atol=1e-6)
    return weights


def meta_cv(z, y, groups, baseline_n, zero_tolerance=0.0):
    baseline, augmented = np.full(len(y), np.nan), np.full(len(y), np.nan)
    rows = []
    with threadpool_limits(limits=2):
        for fold, (fit, held) in enumerate(GroupKFold(3).split(z, y, groups), 1):
            if set(groups[fit]) & set(groups[held]):
                raise ValueError('Meta-fold group leakage')
            base_weights = fit_weights(z[fit, :baseline_n], y[fit], zero_tolerance)
            new_weights = fit_weights(z[fit], y[fit], zero_tolerance)
            baseline[held] = z[held, :baseline_n] @ base_weights
            augmented[held] = z[held] @ new_weights
            rows.append({'fold': fold, 'n': len(held),
                'baseline_mse': float(np.mean((baseline[held] - y[held]) ** 2) * 1e6),
                'augmented_mse': float(np.mean((augmented[held] - y[held]) ** 2) * 1e6),
                'baseline_weights': base_weights.tolist(), 'augmented_weights': new_weights.tolist()})
    if not np.isfinite(baseline).all() or not np.isfinite(augmented).all():
        raise ValueError('Incomplete meta-fold predictions')
    return rows, baseline, augmented


def preregister(baseline_reference=BASELINE, new_encoders=None, experiment_name='encoder_family_oof_fusion', zero_tolerance=0.0):
    if not experiment_name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in experiment_name):
        raise ValueError('Invalid experiment name')
    if not 0 <= zero_tolerance <= 1e-12:
        raise ValueError('Invalid numerical zero threshold')
    specs = NEW_ENCODERS if new_encoders is None else new_encoders
    if not specs or any(not all(k in spec for k in ('name', 'family', 'model', 'revision')) for spec in specs):
        raise ValueError('Encoder specs must pin names, cache families, models and revisions')
    baseline_file = local_path('records_root') / baseline_reference / 'ensemble.json'
    baseline_n = len(read_json(baseline_file)['models'])
    out = local_path('records_root') / '15_fixed_ensembles' / experiment_name / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    registration = {'method': 'One joint convex OOF fusion of existing baseline and preregistered generic image kernels',
        'baseline': baseline_reference, 'baseline_ensemble_sha256': digest(baseline_file),
        'new_encoders': specs, 'source_sha256': digest(__file__), 'split_sha256': digest(local_path('split_path')),
        'selection': f'One joint {baseline_n + len(specs)}-component model; grouped three-fold meta check on training OOF inputs, then fixed validation evaluation',
        'weight_zero_tolerance': zero_tolerance,
        'numerical_weight_policy': 'After each training-only fit, set weights at or below the fixed threshold to zero and renormalize; verify training prices within 1e-6 thousand USD; never select this threshold on validation',
        'export_compilation': 'Remove exact zero-weight terms from the prepared full refit, with all 6399 OOF and 1601 validation compositions replayed',
        'export_gate': {'minimum_meta_cv_relative_gain': 0.01, 'minimum_meta_folds_won': 2, 'require_validation_mse_below_fixed_baseline': True},
        'caveat': 'Base OOF predictions and hyperparameters were previously selected; meta cross-fitting does not refit base models, so it is not an unbiased nested estimate',
        'mllm_components': False, 'test_images_used': False, 'test_labels_used': False,
        'automatic_test_inference': 'Only if both preregistered gates pass; wait for existing GPU export controller',
        'course_scope': 'General image-only encoders, provided labels and existing CLIP visual references; no MLLM or extra SoCal data'}
    write_json(public_path(out) / 'registration.json', registration)
    write_json(public_path(out) / 'status.json', {'status': 'preregistered'})
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    print(out, flush=True)
    return out


def resolve_encoder(spec):
    candidates = []
    public_root = REPOSITORY / 'output/experiment_record'
    for folder in (public_root / '19_image_embedding_reference' / spec['name']).glob('*'):
        if not (folder / 'status.json').exists() or read_json(folder / 'status.json')['status'] != 'completed':
            continue
        registration = read_json(folder / 'registration.json')
        if registration.get('cache_family') != spec['family']:
            continue
        manifest_path = local_path('cache_root') / spec['family'] / registration['feature_cache_key'] / 'manifest.json'
        manifest = read_json(manifest_path)
        if digest(manifest_path) != registration['manifest_sha256']:
            raise ValueError('New encoder feature manifest changed')
        if manifest.get('model') != spec['model'] or manifest.get('revision') != spec['revision'] or manifest.get('mllm_components') is not False or manifest.get('test_images_used') is not False:
            continue
        if 'max_num_patches' in spec and manifest['preprocess'].get('max_num_patches') != spec['max_num_patches']:
            continue
        candidates.append(str(folder.relative_to(public_root)))
    if len(candidates) != 1:
        raise ValueError(f'Expected one completed registered encoder for {spec["name"]}, got {len(candidates)}')
    return candidates[0]


def run(out, predecessor_status, predecessor_pid):
    out = out.resolve()
    public = public_path(out)
    registration = read_json(public / 'registration.json')
    if digest(__file__) != registration['source_sha256'] or digest(local_path('split_path')) != registration['split_sha256']:
        raise ValueError('Preregistered source or split changed')
    if read_json(public / 'status.json')['status'] != 'preregistered':
        raise ValueError('No automatic repeat of a protocol run')
    def status(state, **extra):
        write_json(public / 'status.json', {'status': state, 'pid': os.getpid(), **extra})
    try:
        status('fitting_meta')
        frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
        train, valid = frame[frame.partition.eq('train')], frame[frame.partition.eq('validation')]
        if len(train) != 6399 or len(valid) != 1601 or set(train.group_id) & set(valid.group_id):
            raise ValueError('Unexpected frozen split')
        baseline_dir = local_path('records_root') / registration['baseline']
        if digest(baseline_dir / 'ensemble.json') != registration['baseline_ensemble_sha256']:
            raise ValueError('Fixed baseline changed')
        baseline = read_json(baseline_dir / 'ensemble.json')
        models = baseline['models'] + [resolve_encoder(spec) for spec in registration['new_encoders']]
        oofs, validation = [], []
        heads, prediction_hashes = [], []
        for index, name in enumerate(models):
            source = local_path('records_root') / name
            if read_json(REPOSITORY / 'output/experiment_record' / name / 'status.json')['status'] != 'completed':
                raise ValueError('A component is not completed')
            heads.append(digest(source / 'head.npz'))
            prediction_hashes.append(digest(source / 'predictions.csv'))
            if index < len(baseline['models']) and (heads[-1] != baseline['model_sha256'][index] or prediction_hashes[-1] != baseline['prediction_sha256'][index]):
                raise ValueError('Frozen baseline component changed')
            pt = pd.read_csv(source / 'selected_training_oof.csv', dtype={'imageid': str})
            pv = pd.read_csv(source / 'predictions.csv', dtype={'imageid': str})
            if pt.imageid.tolist() != train.imageid.tolist() or pv.imageid.tolist() != valid.imageid.tolist():
                raise ValueError('Component OOF/validation image order differs')
            np.testing.assert_allclose(pv.price, valid.price, rtol=0, atol=0)
            oofs.append(pt.oof_prediction.to_numpy() / 1000)
            validation.append(pv.prediction.to_numpy())
        z, v = np.stack(oofs, axis=1), np.stack(validation, axis=1)
        if not np.isfinite(z).all() or not np.isfinite(v).all() or (z < 0).any() or (v < 0).any():
            raise ValueError('Invalid component predictions')
        y = train.price.to_numpy() / 1000
        zero_tolerance = registration.get('weight_zero_tolerance', 0.0)
        rows, base_meta, new_meta = meta_cv(z, y, train.group_id.to_numpy(), len(baseline['models']), zero_tolerance)
        write_json(public / 'meta_folds.json', rows)
        pd.DataFrame({'imageid': train.imageid, 'baseline_meta_prediction': base_meta * 1000,
            'augmented_meta_prediction': new_meta * 1000}).to_csv(out / 'meta_crossfit_predictions.csv', index=False)
        with threadpool_limits(limits=2):
            weights = fit_weights(z, y, zero_tolerance)
        # Validation is evaluated after the full-training weights have been fixed.
        fixed_baseline = v[:, :len(baseline['models'])] @ np.asarray(baseline['weights'])
        previous = pd.read_csv(baseline_dir / 'predictions.csv', dtype={'imageid': str})
        if previous.imageid.tolist() != valid.imageid.tolist():
            raise ValueError('Fixed baseline validation order changed')
        np.testing.assert_allclose(fixed_baseline, previous.prediction, rtol=1e-7, atol=1e-4)
        prediction = v @ weights
        metrics = save_predictions(out, valid, prediction, 1300)
        baseline_cv = float(np.mean((base_meta - y) ** 2) * 1e6)
        augmented_cv = float(np.mean((new_meta - y) ** 2) * 1e6)
        folds_won = sum(row['augmented_mse'] < row['baseline_mse'] for row in rows)
        baseline_mse = float(np.mean((fixed_baseline - valid.price.to_numpy()) ** 2))
        gate = registration['export_gate']
        eligible = augmented_cv < baseline_cv * (1 - gate['minimum_meta_cv_relative_gain']) and folds_won >= gate['minimum_meta_folds_won'] and metrics['mse'] < baseline_mse
        ensemble = {'models': models, 'weights': weights.tolist(), 'weights_learned': True,
            'model_sha256': heads, 'prediction_sha256': prediction_hashes, 'source_sha256': digest(__file__),
            'split_sha256': registration['split_sha256'], 'test_images_used': False, 'mllm_components': False,
            'weight_fit': {'algorithm': 'SLSQP convex MSE minimization', 'scope': 'Only training OOF inputs and labels',
                'weight_zero_tolerance': zero_tolerance, 'caveat': registration['caveat']}}
        write_json(out / 'ensemble.json', ensemble)
        registration.update(models=models, weights=weights.tolist(), model_sha256=heads, prediction_sha256=prediction_hashes)
        write_json(public / 'registration.json', registration)
        pd.DataFrame({'imageid': train.imageid, 'prediction': z @ weights * 1000}).to_csv(out / 'meta_training_predictions.csv', index=False)
        write_json(public / 'training_error_correlations.json', {'models': models, 'correlations': np.corrcoef((z - y[:, None]).T).tolist()})
        result = {'head': 'convex_oof_stack', **metrics, 'baseline_validation_mse': baseline_mse,
            'meta_cv_mse': augmented_cv, 'baseline_meta_cv_mse': baseline_cv, 'meta_folds_won': folds_won,
            'meta_training_fit_mse': float(np.mean((z @ weights - y) ** 2) * 1e6), 'weights': weights.tolist(),
            'export_eligible': bool(eligible), 'caveat': registration['caveat']}
        write_json(public / 'results.json', result)
        status('completed', export_eligible=bool(eligible))
        (public / '实验报告.md').write_text('# 新编码器联合融合实验\n\n'
            f'验证 MSE：{metrics["mse"]:.5f}；固定基线：{baseline_mse:.5f}。\n\n'
            f'训练侧元模型分组交叉拟合 MSE：{augmented_cv:.5f}；基线：{baseline_cv:.5f}；改善折数：{folds_won}/3。\n\n'
            f'是否通过预先登记的导出门槛：{bool(eligible)}。公榜未核验，不能声明 50000 目标达成。\n\n'
            f'{len(registration["new_encoders"])} 个新视觉编码器一起加入固定 {len(baseline["models"])} 组件基线，仅学习一次训练 OOF 凸权重；不按验证成绩选择编码器子集。数值零权重阈值预先固定为 {zero_tolerance:g}，在各训练拟合内执行。基础模型已调参，元模型交叉拟合不代表完全嵌套的无偏估计。详见 registration.json、meta_folds.json、results.json；最终导出与独立核验状态另见 export_queue.json。\n', encoding='utf-8')
        subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).with_name('improvement_summary.py'))], check=True)
        if eligible:
            from visual_kernel_submission import prepare
            from audit_visual_kernel_refit import main as audit_refit
            from compile_zero_weight_export import main as compile_export
            prepared_out = prepare(out)
            audit_refit(prepared_out)
            original_prepared_out = prepared_out
            if (weights == 0).any():
                prepared_out = compile_export(prepared_out)
            command_path = out / 'encoder_fusion_export_command.json'
            write_json(command_path, ['scripts/experiments/visual_kernel_submission.py', '--phase', 'infer', '--prepared-run', str(prepared_out)])
            command = [sys.executable, '-X', 'utf8', str(Path(__file__).with_name('after_process_command.py')),
                '--status', str(predecessor_status), '--pid', str(predecessor_pid), '--command', str(command_path)]
            write_json(public / 'export_queue.json', {'status': 'waiting_for_gpu_export', 'prepared_run': str(prepared_out),
                'original_prepared_run': str(original_prepared_out), 'command': command})
            child = subprocess.run(command, check=False)
            if child.returncode:
                write_json(public / 'export_queue.json', {'status': 'failed', 'prepared_run': str(prepared_out), 'exit_code': child.returncode})
                raise RuntimeError('Prepared encoder fusion export did not complete')
            write_json(public / 'export_queue.json', {'status': 'auditing', 'prepared_run': str(prepared_out), 'exit_code': child.returncode})
            audit = subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).with_name('audit_visual_submission.py')),
                '--prepared-run', str(prepared_out)], check=False)
            write_json(public / 'export_queue.json', {'status': 'completed' if audit.returncode == 0 else 'audit_failed',
                'prepared_run': str(prepared_out), 'original_prepared_run': str(original_prepared_out),
                'exit_code': child.returncode, 'audit_exit_code': audit.returncode})
            if audit.returncode:
                raise RuntimeError('Independent submission audit did not complete')
        else:
            write_json(public / 'export_queue.json', {'status': 'not_started', 'reason': 'Preregistered improvement gates not met; preserve evaluation record'})
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['preregister', 'run'], required=True)
    parser.add_argument('--protocol-dir', type=Path)
    parser.add_argument('--gpu-predecessor-status', type=Path)
    parser.add_argument('--gpu-predecessor-pid', type=int)
    parser.add_argument('--baseline', default=BASELINE)
    parser.add_argument('--new-encoders', type=Path)
    parser.add_argument('--experiment-name', default='encoder_family_oof_fusion')
    parser.add_argument('--weight-zero-tolerance', type=float, default=0.0)
    args = parser.parse_args()
    if args.phase == 'preregister':
        preregister(args.baseline, read_json(args.new_encoders) if args.new_encoders else None,
            args.experiment_name, args.weight_zero_tolerance)
    elif args.protocol_dir and args.gpu_predecessor_status and args.gpu_predecessor_pid:
        run(args.protocol_dir, args.gpu_predecessor_status, args.gpu_predecessor_pid)
    else:
        parser.error('run requires protocol and GPU predecessor status/pid')
