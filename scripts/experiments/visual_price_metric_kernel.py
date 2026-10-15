"""Learn a price-relevant feature metric inside each training fold, with null control."""
import argparse
from pathlib import Path
import json
import os
import time
import uuid
import joblib
import numpy as np
import pandas as pd
from scipy.linalg import eigh, solve
from scipy.spatial.distance import cdist
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from common.metrics import save_predictions
from visual_kernel_submission import features_for


def main(source, name, expected_source, expected_head):
    if digest(__file__) != expected_source:
        raise ValueError('Queued supervised-metric source changed')
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in name):
        raise ValueError('Invalid experiment name')
    root = local_path('records_root')
    source_dir = (root / source).resolve()
    source_dir.relative_to(root.resolve())
    if digest(source_dir / 'head.npz') != expected_head:
        raise ValueError('Frozen source head changed')
    source_registration = read_json(REPOSITORY / 'output/experiment_record' / source / 'registration.json')
    source_public = REPOSITORY / 'output/experiment_record' / source
    if read_json(source_public / 'status.json')['status'] != 'completed' or source_registration.get('cache_family') not in {'siglip_features', 'siglip2_giant_features', 'pe_core_features', 'pe_core_multiview', 'pe_core_giant_features', 'pe_core_giant_whole_features'}:
        raise ValueError('Expected a completed generic image-only kernel source')
    manifest = read_json(local_path('cache_root') / source_registration['cache_family'] / source_registration['feature_cache_key'] / 'manifest.json')
    if manifest.get('mllm_components', False) is not False or manifest.get('test_images_used') is not False:
        raise ValueError('No MLLM or test-selected image features')
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    train, valid = frame.partition.eq('train').to_numpy(), frame.partition.eq('validation').to_numpy()
    if len(frame) != 8000 or train.sum() != 6399 or valid.sum() != 1601 or set(frame.loc[train, 'group_id']) & set(frame.loc[valid, 'group_id']):
        raise ValueError('Unexpected frozen group split')
    x = features_for(source_registration, frame)
    with np.load(root / source / 'head.npz', allow_pickle=False) as head:
        np.testing.assert_allclose(x[train], head['train_features'], rtol=0, atol=1e-12)
        image_gamma, null_alpha = float(head['gamma']), float(head['alpha'])
    xt, xv = x[train], x[valid]
    y = frame.loc[train, 'price'].to_numpy() / 1000
    groups = frame.loc[train, 'group_id'].to_numpy()
    prior = pd.read_csv(root / source / 'selected_training_oof.csv', dtype={'imageid': str})
    if prior.imageid.tolist() != frame.loc[train, 'imageid'].tolist():
        raise ValueError('Source OOF image order differs')
    null_oof = prior.raw_oof_prediction.to_numpy() / 1000
    null_score = float(np.mean((np.maximum(null_oof, 0) - y) ** 2) * 1e6)
    np.testing.assert_allclose(null_score, read_json(REPOSITORY / 'output/experiment_record' / source / 'results.json')['training_oof_mse'], rtol=0, atol=1e-7)
    settings = [(4, .01), (4, .1), (16, .01), (16, .1)]
    alphas = [.1, 1., 10.]
    run = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
    relative = '19_image_embedding_reference/' + name + '/' + run
    out, public = root / relative, REPOSITORY / 'output/experiment_record' / relative
    out.mkdir(parents=True, exist_ok=False)
    registration = {'method': 'Image RBF multiplied by fold-fitted supervised PLS-score RBF',
        'source': source, 'source_head_sha256': digest(root / source / 'head.npz'), 'source_sha256': digest(__file__),
        'cache_family': source_registration['cache_family'], 'feature_cache_key': source_registration['feature_cache_key'],
        'manifest_sha256': source_registration['manifest_sha256'], 'feature_dim': int(x.shape[1]),
        'implementation_lineage': 'Same fold-fitted PLS product-kernel recipe as the recorded SigLIP2 metric, applied to an explicitly pinned generic visual source',
        'split_sha256': digest(local_path('split_path')), 'source_feature_manifest_sha256': source_registration['manifest_sha256'],
        'image_gamma': image_gamma, 'settings': settings, 'alphas': alphas, 'null_control': {'pls_gamma': 0, 'alpha': null_alpha},
        'pls': '16 components fit on fitting-fold image embeddings and training prices only; use first 4 or 16 scores',
        'score_scaling': 'StandardScaler fit on fitting-fold scores only; squared distances divided by component count',
        'selection': 'Original unweighted training GroupKFold3 MSE; validation evaluated once after selection',
        'test_images_used': False, 'mllm_components': False,
        'course_scope': 'General-purpose frozen image features and provided training labels only',
        'caveat': 'Source image gamma previously selected on these folds; selected CV criterion is not an unbiased nested estimate'}
    write_json(public / 'registration.json', registration)
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    def status(state, **extra):
        write_json(public / 'status.json', {'status': state, 'pid': os.getpid(), **extra})
    started = time.perf_counter()
    try:
        status('fitting', fold=0)
        best = {'components': 0, 'pls_gamma': 0., 'alpha': null_alpha, 'training_oof_mse': null_score}
        best_oof = null_oof.copy()
        rows = [best.copy()]
        candidates = {(dim, gamma, alpha): np.full(len(y), np.nan) for dim, gamma in settings for alpha in alphas}
        with threadpool_limits(limits=2):
            image_kernel = np.exp(-image_gamma * np.maximum(2 - 2 * (xt @ xt.T), 0))
            for fold, (fit, held) in enumerate(GroupKFold(3).split(xt, y, groups)):
                if set(groups[fit]) & set(groups[held]):
                    raise ValueError('Group leakage')
                pls = PLSRegression(n_components=16, scale=True, max_iter=500)
                pls.fit(xt[fit], y[fit])
                scores = pls.transform(xt)
                scores = StandardScaler().fit(scores[fit]).transform(scores)
                for dim, gamma in settings:
                    status('fitting', fold=fold + 1, components=dim, pls_gamma=gamma)
                    a = image_kernel[np.ix_(fit, fit)] * np.exp(-gamma * cdist(scores[fit, :dim], scores[fit, :dim], 'sqeuclidean') / dim)
                    b = image_kernel[np.ix_(held, fit)] * np.exp(-gamma * cdist(scores[held, :dim], scores[fit, :dim], 'sqeuclidean') / dim)
                    eigen, vectors = eigh(a, check_finite=False, overwrite_a=True)
                    projected = vectors.T @ (y[fit] - y[fit].mean())
                    held_vectors = b @ vectors
                    for alpha in alphas:
                        candidates[(dim, gamma, alpha)][held] = held_vectors @ (projected / (np.maximum(eigen, 0) + alpha)) + y[fit].mean()
                    print(f'PLS metric fold {fold + 1}/3, dimensions={dim}, gamma={gamma}', flush=True)
                    del a, b, eigen, vectors, held_vectors
            for (dim, gamma, alpha), oof in candidates.items():
                if not np.isfinite(oof).all():
                    raise ValueError('Incomplete OOF predictions')
                score = float(np.mean((np.maximum(oof, 0) - y) ** 2) * 1e6)
                row = {'components': dim, 'pls_gamma': gamma, 'alpha': alpha, 'training_oof_mse': score}
                rows.append(row)
                if score < best['training_oof_mse']:
                    best, best_oof = row.copy(), oof.copy()
            pd.DataFrame(rows).to_csv(public / 'training_cv.csv', index=False)
            if best['components']:
                pls = PLSRegression(n_components=16, scale=True, max_iter=500).fit(xt, y)
                scaler = StandardScaler().fit(pls.transform(xt))
                st = scaler.transform(pls.transform(xt))[:, :best['components']]
                sv = scaler.transform(pls.transform(xv))[:, :best['components']]
                kernel = image_kernel * np.exp(-best['pls_gamma'] * cdist(st, st, 'sqeuclidean') / best['components'])
                validation_kernel = np.exp(-image_gamma * np.maximum(2 - 2 * (xv @ xt.T), 0)) * np.exp(-best['pls_gamma'] * cdist(sv, st, 'sqeuclidean') / best['components'])
                joblib.dump({'pls': pls, 'scaler': scaler}, out / 'feature_metric.joblib')
            else:
                kernel = image_kernel.copy()
                validation_kernel = np.exp(-image_gamma * np.maximum(2 - 2 * (xv @ xt.T), 0))
                st = np.empty((len(xt), 0))
            kernel.flat[::len(y) + 1] += best['alpha']
            mean = float(y.mean())
            coefficients = solve(kernel, y - mean, assume_a='pos', check_finite=False, overwrite_a=True)
            prediction = (validation_kernel @ coefficients + mean) * 1000
        metrics = save_predictions(out, frame.loc[valid], prediction, 1300)
        if best['components'] == 0:
            source_predictions = pd.read_csv(root / source / 'predictions.csv')
            np.testing.assert_allclose(prediction, source_predictions.raw_prediction, rtol=0, atol=1e-4)
        np.savez_compressed(out / 'head.npz', train_features=xt, train_pls_scores=st, coefficients=coefficients,
            image_gamma=image_gamma, pls_gamma=best['pls_gamma'], components=best['components'], alpha=best['alpha'], target_mean=mean)
        pd.DataFrame({'imageid': frame.loc[train, 'imageid'], 'raw_oof_prediction': best_oof * 1000,
            'oof_prediction': np.maximum(best_oof, 0) * 1000}).to_csv(out / 'selected_training_oof.csv', index=False)
        result = {'head': 'supervised_metric_kernel', **best, **metrics, 'seconds': time.perf_counter() - started}
        write_json(public / 'results.json', result)
        status('completed')
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--source-head-sha256', required=True)
    args = parser.parse_args()
    main(args.source, args.name, args.source_sha256, args.source_head_sha256)
