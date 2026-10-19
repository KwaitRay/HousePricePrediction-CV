"""Joint kernel regression on visual feature spaces; all selection is train-only."""
import argparse
import json
import os
import time
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh, solve
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from common.metrics import save_predictions
from visual_kernel_submission import features_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs=3, required=True)
    parser.add_argument('--name', default='additive_clip_blip_street_kernel')
    args = parser.parse_args()
    root = local_path('records_root')
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    train, valid = frame.partition.eq('train').to_numpy(), frame.partition.eq('validation').to_numpy()
    if train.sum() != 6399 or valid.sum() != 1601 or set(frame.loc[train, 'group_id']) & set(frame.loc[valid, 'group_id']):
        raise ValueError('Unexpected frozen grouped split')
    features, gammas, provenance = [], [], []
    for name in args.models:
        source = root / name
        public_source = REPOSITORY / 'output/experiment_record' / name
        if read_json(public_source / 'status.json')['status'] != 'completed':
            raise ValueError('An input reference is not completed')
        config = read_json(public_source / 'registration.json')
        x = features_for(config, frame)
        with np.load(source / 'head.npz', allow_pickle=False) as head:
            if 'training_weights' in head.files:
                raise ValueError('This control uses unweighted source feature references')
            np.testing.assert_allclose(x[train], head['train_features'], rtol=0, atol=1e-12)
            gammas.append(float(head['gamma']))
        features.append(x)
        provenance.append({'source': name, 'head_sha256': digest(source / 'head.npz'),
                           'cache_family': config['cache_family'], 'cache_key': config['feature_cache_key'],
                           'feature_dim': x.shape[1], 'manifest_sha256': config['manifest_sha256']})
    candidates = [[1/3, 1/3, 1/3], [.5, .2, .3]]
    alphas = [.01, .1, 1., 10.]
    run = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
    relative = Path('15_fixed_ensembles') / args.name / run
    out, public = root / relative, REPOSITORY / 'output/experiment_record' / relative
    out.mkdir(parents=True, exist_ok=False)
    registration = {'method': 'RBF kernels added before solving one common dual coefficient vector',
                    'components': provenance, 'gammas': gammas, 'kernel_weight_candidates': candidates,
                    'alphas': alphas, 'source_sha256': digest(__file__), 'split_sha256': digest(local_path('split_path')),
                    'selection': 'Eight preregistered kernel-weight/alpha candidates; unweighted original-price training GroupKFold3 MSE',
                    'centering': 'Fit-fold training target mean; no validation target use',
                    'test_images_used': False, 'course_scope': 'Frozen visual CLIP/BLIP/StreetCLIP references; eligibility pending confirmation',
                    'caveat': 'Source gammas previously chosen using training CV; selected training OOF score is not nested-unbiased'}
    write_json(public / 'registration.json', registration)
    write_json(out / 'registration.json', registration)
    def status(state, **kw):
        write_json(public / 'status.json', {'status': state, 'pid': os.getpid(), **kw})
    begun = time.perf_counter()
    try:
        y = frame.loc[train, 'price'].to_numpy() / 1000
        groups = frame.loc[train, 'group_id'].to_numpy()
        folds = list(GroupKFold(3).split(np.zeros((len(y), 1)), y, groups))
        rows, best = [], None
        with threadpool_limits(limits=4):
            kernels = [np.exp(-gamma * np.maximum(2 - 2 * (x[train] @ x[train].T), 0)) for x, gamma in zip(features, gammas)]
            for candidate_id, weights in enumerate(candidates):
                kernel = sum(w * k for w, k in zip(weights, kernels))
                oof = {alpha: np.full(len(y), np.nan) for alpha in alphas}
                for fold, (fit, held) in enumerate(folds):
                    assert not set(groups[fit]) & set(groups[held])
                    status('fitting', candidate=candidate_id, fold=fold + 1)
                    eigen, vectors = eigh(kernel[np.ix_(fit, fit)], check_finite=False, overwrite_a=True)
                    mean = float(y[fit].mean())
                    projected = vectors.T @ (y[fit] - mean)
                    held_vectors = kernel[np.ix_(held, fit)] @ vectors
                    for alpha in alphas:
                        oof[alpha][held] = held_vectors @ (projected / (np.maximum(eigen, 0) + alpha)) + mean
                    print(f'Additive visual kernel weights {candidate_id + 1}/2 fold {fold + 1}/3', flush=True)
                for alpha in alphas:
                    score = float(np.mean((np.maximum(oof[alpha], 0) - y)**2) * 1e6)
                    row = {'weight_candidate': candidate_id, 'alpha': alpha, 'training_oof_mse': score}
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(public / 'training_cv.csv', index=False)
                    if best is None or score < best['training_oof_mse']:
                        best, best_oof = row.copy(), oof[alpha].copy()
            weights, alpha = candidates[best['weight_candidate']], best['alpha']
            kernel = sum(w * k for w, k in zip(weights, kernels))
            mean = float(y.mean())
            kernel.flat[::len(y) + 1] += alpha
            coefficients = solve(kernel, y - mean, assume_a='pos', check_finite=False, overwrite_a=True)
            validation_kernel = sum(w * np.exp(-g * np.maximum(2 - 2 * (x[valid] @ x[train].T), 0))
                                    for w, g, x in zip(weights, gammas, features))
            raw = (validation_kernel @ coefficients + mean) * 1000
        metrics = save_predictions(out, frame.loc[valid], raw, 1300)
        tensors = {f'train_features_{i}': x[train] for i, x in enumerate(features)}
        np.savez_compressed(out / 'head.npz', **tensors, coefficients=coefficients, gammas=gammas,
                            kernel_weights=weights, alpha=alpha, target_mean=mean)
        pd.DataFrame({'imageid': frame.loc[train, 'imageid'], 'raw_oof_prediction': best_oof * 1000,
                      'oof_prediction': np.maximum(best_oof, 0) * 1000}).to_csv(out / 'selected_training_oof.csv', index=False)
        result = {'head': 'additive_rbf_kernel', **best, 'kernel_weights': weights, **metrics,
                  'seconds': time.perf_counter() - begun}
        write_json(public / 'results.json', result)
        status('completed')
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
