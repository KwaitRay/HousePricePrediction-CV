"""Train-only grouped RBF search for the explicitly registered PE multiview cache."""
import argparse
import os
from pathlib import Path
import time
import uuid
import numpy as np
import pandas as pd
from scipy.linalg import eigh, solve
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from common.metrics import save_predictions


def main(reference, expected_source):
    if digest(__file__) != expected_source:
        raise ValueError('Queued kernel source changed before launch')
    reference = reference.resolve()
    public_reference = REPOSITORY / 'output/experiment_record' / reference.relative_to(local_path('records_root'))
    state = read_json(public_reference / 'status.json')
    registration = read_json(public_reference / 'registration.json')
    if state['status'] != 'completed' or not state.get('feature_extraction_completed') or state.get('price_model_evaluated'):
        raise ValueError('Expected completed, unevaluated PE multiview features')
    if registration['cache_family'] != 'pe_core_multiview' or registration.get('mllm_components') is not False or registration.get('test_images_used') is not False:
        raise ValueError('Only the registered image-only multiview features are supported')
    protocol = registration['next_kernel_protocol']
    if protocol['gammas'] != [1., 4., 16.] or protocol['alphas'] != [.01, .1, 1., 10.]:
        raise ValueError('Fixed training-CV candidates changed')
    cost = read_json(public_reference / 'cost.json')
    cache = local_path('cache_root') / 'pe_core_multiview' / cost['feature_cache_key']
    manifest_path = cache / 'manifest.json'
    manifest = read_json(manifest_path)
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    if digest(local_path('split_path')) != registration['split_sha256'] or frame.imageid.tolist() != manifest['imageids']:
        raise ValueError('Frozen split or feature order changed')
    if digest(reference / 'source.py') != manifest['source_sha256'] or manifest['model'] != registration['model'] or manifest['revision'] != registration['revision']:
        raise ValueError('Feature extraction source or checkpoint changed')
    paths = sorted(cache.glob('*.npy'))
    expected = [f'{offset:05d}_4.npy' for offset in range(0, 8000, 4)]
    if [p.name for p in paths] != expected:
        raise ValueError('Incomplete or unordered multiview chunks')
    chunks = [np.load(path, allow_pickle=False) for path in paths]
    if any(value.shape != (4, 3072) for value in chunks):
        raise ValueError('Unexpected multiview chunk shape')
    x = np.concatenate(chunks).astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite frozen features')
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, rtol=0, atol=1e-5)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    train, valid = frame.partition.eq('train'), frame.partition.eq('validation')
    if len(frame) != 8000 or train.sum() != 6399 or valid.sum() != 1601 or set(frame.loc[train, 'group_id']) & set(frame.loc[valid, 'group_id']):
        raise ValueError('Unexpected train/validation split')
    xt, xv = x[train], x[valid]
    y = frame.loc[train, 'price'].to_numpy() / 1000
    groups = frame.loc[train, 'group_id'].to_numpy()
    folds = list(GroupKFold(3).split(xt, y, groups))
    relative = Path('19_image_embedding_reference/pe_core_l14_336_multiview_kernel') / (
        time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    out, public = local_path('records_root') / relative, REPOSITORY / 'output/experiment_record' / relative
    out.mkdir(parents=True, exist_ok=False)
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    write_json(public / 'registration.json', {'method': 'mean-centered RBF kernel ridge',
        'cache_family': 'pe_core_multiview', 'feature_cache_key': cost['feature_cache_key'],
        'manifest_sha256': digest(manifest_path), 'feature_dim': 3072,
        'source_sha256': digest(__file__), 'split_sha256': digest(local_path('split_path')),
        'reference': str(reference.relative_to(local_path('records_root'))),
        'reference_registration_sha256': digest(public_reference / 'registration.json'),
        'gammas': protocol['gammas'], 'alphas': protocol['alphas'], 'train_only_group_cv': 3,
        'selection': protocol['selection'], 'dtype': 'float64',
        'centering': 'Only fitting-fold target mean; validation labels used after selection',
        'mllm_components': False, 'test_images_used': False, 'test_labels_used': False,
        'course_scope': registration['course_scope']})
    def status(value, **extra):
        write_json(public / 'status.json', {'status': value, 'pid': os.getpid(), **extra})
    started = time.perf_counter()
    try:
        status('running')
        best, results = None, []
        with threadpool_limits(limits=4):
            distance = np.maximum(2 - 2 * (xt @ xt.T), 0)
            for gamma in protocol['gammas']:
                kernel = np.exp(-gamma * distance)
                oof = {alpha: np.full(len(y), np.nan) for alpha in protocol['alphas']}
                for fold, (fit, held) in enumerate(folds, 1):
                    if set(groups[fit]) & set(groups[held]):
                        raise ValueError('Training-fold group leakage')
                    status('fitting', gamma=gamma, fold=fold)
                    eigen, vectors = eigh(kernel[np.ix_(fit, fit)], check_finite=False, overwrite_a=True)
                    projected = vectors.T @ (y[fit] - y[fit].mean())
                    held_vectors = kernel[np.ix_(held, fit)] @ vectors
                    for alpha in protocol['alphas']:
                        oof[alpha][held] = held_vectors @ (projected / (np.maximum(eigen, 0) + alpha)) + y[fit].mean()
                    print(f'PE multiview gamma={gamma} fold={fold}/3', flush=True)
                for alpha in protocol['alphas']:
                    if not np.isfinite(oof[alpha]).all():
                        raise ValueError('Incomplete OOF predictions')
                    score = float(np.mean((np.maximum(oof[alpha], 0) - y) ** 2) * 1e6)
                    row = {'gamma': gamma, 'alpha': alpha, 'training_oof_mse': score}
                    results.append(row)
                    pd.DataFrame(results).to_csv(public / 'training_cv.csv', index=False)
                    if best is None or score < best['training_oof_mse']:
                        best, best_oof = row.copy(), oof[alpha].copy()
            gamma, alpha, mean = best['gamma'], best['alpha'], float(y.mean())
            kernel = np.exp(-gamma * distance)
            coefficients = solve(kernel + alpha * np.eye(len(y)), y - mean, assume_a='pos', check_finite=False)
            probes = np.arange(0, len(y), 511)
            np.testing.assert_allclose(kernel[probes] @ coefficients + mean,
                y[probes] - alpha * coefficients[probes], rtol=0, atol=1e-9)
            prediction = (np.exp(-gamma * np.maximum(2 - 2 * (xv @ xt.T), 0)) @ coefficients + mean) * 1000
        metrics = save_predictions(out, frame.loc[valid], prediction, 1300)
        np.savez_compressed(out / 'head.npz', train_features=xt, coefficients=coefficients,
            gamma=gamma, alpha=alpha, target_mean=mean)
        pd.DataFrame({'imageid': frame.loc[train, 'imageid'], 'raw_oof_prediction': best_oof * 1000,
            'oof_prediction': np.maximum(best_oof, 0) * 1000}).to_csv(out / 'selected_training_oof.csv', index=False)
        write_json(out / 'selected_oof_protocol.json', {**best,
            'caveat': 'Selected on these training folds, not nested-CV performance'})
        write_json(public / 'results.json', {**best, **metrics, 'seconds': time.perf_counter() - started})
        status('completed')
        (public / '实验报告.md').write_text('# PE-Core 全图与局部视图核回归\n\n'
            f'验证 MSE {metrics["mse"]:.5f}；MAE {metrics["mae"]:.5f}；高价组 MSE {metrics["tail_mse"]:.5f}。\n\n'
            f'训练侧三折 OOF 选择 gamma={gamma:g}、alpha={alpha:g}，OOF MSE {best["training_oof_mse"]:.5f}；全部 12 项候选见 training_cv.csv。\n\n'
            '保留全图，等权加入完整灰边视图和固定中心局部视图；验证标签不参与超参数选择。训练 OOF 已用于选择，因此不是嵌套 CV 无偏估计。尚无融合与公榜结果，不据单模型验证宣布 50000 目标达成。\n', encoding='utf-8')
        print(f'PE multiview completed: validation MSE {metrics["mse"]:.5f}', flush=True)
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    main(args.reference, args.source_sha256)
