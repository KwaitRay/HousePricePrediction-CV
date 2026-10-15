"""One-factor train-only price-band weighting on the fixed whole-frame PE-G KRR."""
import argparse
import os
import time
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from common.metrics import save_predictions
from visual_kernel_submission import features_for

SOURCE = '19_image_embedding_reference/pe_core_g14_448_whole_kernel/20261014T090655Z_2d55c743'
BASELINE = '15_fixed_ensembles/pe_giant_whole_incremental_fusion/20261014T082727Z_c5152cef'
METHOD = 'Price-band weighted image RBF with fold-fitted normalized weights and weighted target mean'


def band_weights(price, strength):
    bands = np.digitize(price, [300., 700., 1300.], right=True)
    counts = np.bincount(bands, minlength=4)
    assert (counts > 0).all()
    levels = np.clip(np.sqrt(len(price)/(4*counts)), .8, 1.25)
    base = levels[bands]
    base /= base.mean()
    weights = (1-strength) + strength*base
    weights /= weights.mean()
    return weights, counts, levels


def fit(kernel, y, price, alpha, strength):
    weights, counts, levels = band_weights(price, strength)
    mean = float(np.average(y, weights=weights))
    matrix = kernel.copy()
    matrix.flat[::len(y)+1] += alpha/weights
    coefficients = solve(matrix, y-mean, assume_a='pos', check_finite=False)
    return coefficients, mean, weights, counts, levels


def preregister():
    root = local_path('records_root')
    public_source = REPOSITORY / 'output/experiment_record' / SOURCE
    assert read_json(public_source / 'status.json')['status'] == 'completed'
    source = read_json(public_source / 'registration.json')
    with np.load(root / SOURCE / 'head.npz', allow_pickle=False) as head:
        gamma, alpha = float(head['gamma']), float(head['alpha'])
    relative = Path('19_image_embedding_reference/pe_giant_whole_price_band_weighted_kernel') / (
        time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    out, public = root / relative, REPOSITORY / 'output/experiment_record' / relative
    out.mkdir(parents=True, exist_ok=False)
    registration = {'method': METHOD, 'source': SOURCE, 'baseline_fusion': BASELINE,
        'source_sha256': digest(__file__), 'split_sha256': digest(local_path('split_path')),
        'feature_cache_key': source['feature_cache_key'], 'cache_family': source['cache_family'],
        'manifest_sha256': source['manifest_sha256'], 'feature_dim': source['feature_dim'],
        'source_head_sha256': digest(root / SOURCE / 'head.npz'),
        'source_oof_sha256': digest(root / SOURCE / 'selected_training_oof.csv'),
        'source_predictions_sha256': digest(root / SOURCE / 'predictions.csv'),
        'baseline_ensemble_sha256': digest(root / BASELINE / 'ensemble.json'),
        'fixed_gamma': gamma, 'fixed_alpha': alpha, 'strengths': [0., .5, 1.],
        'bands': ['<=300', '(300,700]', '(700,1300]', '>1300'],
        'weight_formula': 'Within each fit partition, clip(sqrt(N/(4*n_band)), .8, 1.25), normalize mean; mix with ones by strength; renormalize',
        'kernel_equation': '(K + alpha*diag(1/w))c = y - weighted_training_mean; prediction=Knew*c+mean',
        'hypothesis': 'Whole-frame meta-crossfit shows low overprediction and high underprediction; bounded tail weighting may improve the readout on the current strong representation',
        'comparison': 'Same cached features, grouped folds, gamma and alpha; only objective weighting/associated prior mean change. Not a rerun of the failed from-scratch CNN weighting.',
        'selection': 'Unweighted original-price grouped three-fold OOF MSE, including strength zero exact replay control; validation once after selection',
        'control_tolerance': {'rtol': 1e-7, 'atol': 1e-4},
        'full_refit_adapter_supported': False, 'automatic_test_export': False,
        'test_images_used': False, 'test_labels_used': False, 'mllm_components': False,
        'caveat': 'Source kernel hyperparameters already selected; not unbiased nested CV; validation remains development data'}
    write_json(public / 'registration.json', registration)
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    write_json(public / 'status.json', {'status': 'preregistered'})
    print(out, flush=True)


def run(out):
    root = local_path('records_root')
    out = out.resolve()
    public = REPOSITORY / 'output/experiment_record' / out.relative_to(root)
    registration = read_json(public / 'registration.json')
    assert read_json(public / 'status.json')['status'] == 'preregistered'
    assert digest(__file__) == registration['source_sha256']
    assert digest(local_path('split_path')) == registration['split_sha256']
    for field, filename in [('source_head_sha256','head.npz'), ('source_oof_sha256','selected_training_oof.csv'), ('source_predictions_sha256','predictions.csv')]:
        assert digest(root / SOURCE / filename) == registration[field]
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    train, valid = frame.partition.eq('train').to_numpy(), frame.partition.eq('validation').to_numpy()
    assert train.sum() == 6399 and valid.sum() == 1601
    assert not set(frame.loc[train,'group_id']) & set(frame.loc[valid,'group_id'])
    source_registration = read_json(REPOSITORY / 'output/experiment_record' / SOURCE / 'registration.json')
    x = features_for(source_registration, frame)
    xt, xv = x[train], x[valid]
    price = frame.loc[train,'price'].to_numpy()
    y = price/1000
    gamma, alpha = registration['fixed_gamma'], registration['fixed_alpha']
    def status(state, **extra):
        write_json(public / 'status.json', {'status': state, 'pid': os.getpid(), **extra})
    try:
        status('fitting_cv')
        with threadpool_limits(2):
            kernel = np.exp(-gamma*np.maximum(2-2*xt@xt.T, 0))
            folds = list(GroupKFold(3).split(xt,y,frame.loc[train,'group_id'].to_numpy()))
            predictions, rows, fold_records = {}, [], []
            for strength in registration['strengths']:
                oof = np.full(len(y), np.nan)
                for i, (fit_rows, held) in enumerate(folds):
                    coefficients, mean, weights, counts, levels = fit(kernel[np.ix_(fit_rows,fit_rows)], y[fit_rows], price[fit_rows], alpha, strength)
                    oof[held] = kernel[np.ix_(held,fit_rows)]@coefficients+mean
                    fold_records.append({'strength': strength, 'fold': i+1, 'counts': counts.tolist(),
                        'raw_band_weights': levels.tolist(), 'weight_min': float(weights.min()), 'weight_max': float(weights.max())})
                assert np.isfinite(oof).all()
                predictions[strength] = oof
                row = {'strength': strength, 'training_oof_mse': float(np.mean((np.maximum(oof,0)*1000-price)**2))}
                rows.append(row)
                write_json(public / 'training_cv.json', rows)
                print(row, flush=True)
            saved = pd.read_csv(root / SOURCE / 'selected_training_oof.csv', dtype={'imageid': str})
            assert saved.imageid.tolist() == frame.loc[train,'imageid'].tolist()
            np.testing.assert_allclose(predictions[0.]*1000, saved.raw_oof_prediction, **registration['control_tolerance'])
            best = min(rows, key=lambda r:r['training_oof_mse'])
            strength = best['strength']
            coefficients, mean, weights, counts, levels = fit(kernel,y,price,alpha,strength)
            val_kernel = np.exp(-gamma*np.maximum(2-2*xv@xt.T, 0))
            raw = (val_kernel@coefficients+mean)*1000
            control_coefficients, control_mean, *_ = fit(kernel,y,price,alpha,0.)
            control = (val_kernel@control_coefficients+control_mean)*1000
        saved_val = pd.read_csv(root / SOURCE / 'predictions.csv', dtype={'imageid': str})
        assert saved_val.imageid.tolist() == frame.loc[valid,'imageid'].tolist()
        np.testing.assert_allclose(control, saved_val.raw_prediction, **registration['control_tolerance'])
        metrics = save_predictions(out, frame.loc[valid], raw, 1300)
        np.savez_compressed(out / 'head.npz', train_features=xt, coefficients=coefficients,
            gamma=gamma, alpha=alpha, target_mean=mean, training_weights=weights,
            strength=strength, kernel_type='price_band_weighted_rbf')
        pd.DataFrame({'imageid':frame.loc[train,'imageid'], 'raw_oof_prediction':predictions[strength]*1000,
            'oof_prediction':np.maximum(predictions[strength],0)*1000}).to_csv(out / 'selected_training_oof.csv', index=False)
        write_json(public / 'weight_folds.json', fold_records)
        write_json(public / 'results.json', {**best, **metrics, 'zero_weight_control_replay_passed': True,
            'control_oof_mse': rows[0]['training_oof_mse'], 'control_validation_mse': float(np.mean((np.maximum(control,0)-frame.loc[valid,'price'].to_numpy())**2)),
            'full_refit_adapter_supported': False, 'test_images_used': False})
        status('completed', selected_strength=strength)
        print({**best, **metrics}, flush=True)
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['preregister','run'], required=True)
    parser.add_argument('--protocol-dir', type=Path)
    args = parser.parse_args()
    if args.phase == 'preregister':
        preregister()
    elif args.protocol_dir:
        run(args.protocol_dir)
    else:
        parser.error('run requires protocol-dir')
