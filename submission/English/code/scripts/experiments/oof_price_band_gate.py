"""Train-only blending of weighted/unweighted heads by predicted price band."""
import argparse
import json
import time
import uuid
import numpy as np
import pandas as pd
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from common.metrics import save_predictions

BOUNDARIES = [300.0, 700.0, 1300.0]
MIN_GROUP_N = 100


def optimum_weight(base, weighted, truth):
    delta = weighted - base
    denominator = float(delta @ delta)
    return float(np.clip(((truth - base) @ delta) / denominator, 0, 1)) if denominator > 1e-12 else 0.0


def fit_gate(base_oof, weighted_oof, training_prices):
    global_weight = optimum_weight(base_oof, weighted_oof, training_prices)
    groups = np.digitize(base_oof, BOUNDARIES, right=True)
    weights, counts = [], []
    for band in range(4):
        mask = groups == band
        counts.append(int(mask.sum()))
        weights.append(optimum_weight(base_oof[mask], weighted_oof[mask], training_prices[mask])
                       if mask.sum() >= MIN_GROUP_N else global_weight)
    return {'boundaries': BOUNDARIES, 'weights': weights, 'training_band_counts': counts,
            'global_fallback_weight': global_weight, 'min_group_n': MIN_GROUP_N,
            'gate_input': 'Unweighted model prediction; never ground-truth price at prediction time'}


def apply_gate(base, weighted, parameters):
    groups = np.digitize(base, parameters['boundaries'], right=True)
    weights = np.asarray(parameters['weights'])[groups]
    return base + weights * (weighted - base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--weighted', required=True)
    parser.add_argument('--name', default='l14_multiview_price_band_gate')
    args = parser.parse_args()
    root = local_path('records_root')
    paths = [root / args.base, root / args.weighted]
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    train = frame[frame.partition.eq('train')].copy()
    valid = frame[frame.partition.eq('validation')].copy()
    if len(train) != 6399 or len(valid) != 1601 or set(train.group_id) & set(valid.group_id):
        raise ValueError('Unexpected frozen group split')
    for path in paths:
        public = REPOSITORY / 'output/experiment_record' / path.relative_to(root)
        if read_json(public / 'status.json')['status'] != 'completed':
            raise ValueError('A source experiment is not completed')
    oofs = []
    for path in paths:
        values = pd.read_csv(path / 'selected_training_oof.csv', dtype={'imageid': str})
        if not values.imageid.is_unique or set(values.imageid) != set(train.imageid):
            raise ValueError('Source OOF coverage differs')
        oofs.append(values.set_index('imageid').loc[train.imageid, 'oof_prediction'].to_numpy())
    if not np.isfinite(oofs).all():
        raise ValueError('Nonfinite source predictions')
    parameters = fit_gate(oofs[0], oofs[1], train.price.to_numpy())
    run = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
    relative = '15_fixed_ensembles/' + args.name + '/' + run
    out, public = root / relative, REPOSITORY / 'output/experiment_record' / relative
    out.mkdir(parents=True, exist_ok=False)
    registration = {'sources': [args.base, args.weighted], 'source_head_sha256': [digest(p / 'head.npz') for p in paths],
                    'source_oof_sha256': [digest(p / 'selected_training_oof.csv') for p in paths],
                    'source_sha256': digest(__file__), 'split_sha256': digest(local_path('split_path')),
                    'parameters': parameters, 'test_images_used': False,
                    'fit': 'Closed-form bounded squared-error blend fit independently in training OOF predicted-price bands',
                    'selection': 'Fixed bands and minimum support; no validation-fitted gate or weights',
                    'course_scope': 'Visual CLIP features only, eligibility pending confirmation',
                    'caveat': 'Meta training fit is not independent OOF; base hyperparameters already selected using these folds'}
    write_json(public / 'registration.json', registration)
    write_json(out / 'gate.json', registration)
    predictions = []
    for path in paths:
        values = pd.read_csv(path / 'predictions.csv', dtype={'imageid': str})
        if not values.imageid.is_unique or set(values.imageid) != set(valid.imageid):
            raise ValueError('Validation source coverage differs')
        aligned = values.set_index('imageid').loc[valid.imageid]
        np.testing.assert_allclose(aligned.price.to_numpy(), valid.price.to_numpy(), rtol=0, atol=0)
        predictions.append(aligned.prediction.to_numpy())
    pred = apply_gate(predictions[0], predictions[1], parameters)
    train_fit = apply_gate(oofs[0], oofs[1], parameters)
    metrics = save_predictions(out, valid, pred, 1300)
    # Avoid publishing fitted meta predictions as a new independent OOF feature source.
    pd.DataFrame({'imageid': train.imageid, 'prediction': train_fit}).to_csv(out / 'meta_training_predictions.csv', index=False)
    result = {'head': 'train_oof_price_band_gate', **metrics,
              'meta_training_fit_mse': float(np.mean((train_fit - train.price.to_numpy())**2)),
              'weighted_blend_per_predicted_band': parameters['weights']}
    write_json(public / 'results.json', result)
    slices = []
    for tier, mask in [('low', valid.price <= 300), ('middle', (valid.price > 300) & (valid.price <= 1300)),
                       ('high', valid.price > 1300)]:
        error = pred[mask] - valid.price.to_numpy()[mask]
        slices.append({'tier': tier, 'n': int(mask.sum()), 'mse': float(np.mean(error**2)),
                       'mae': float(np.mean(abs(error))), 'bias': float(np.mean(error))})
    write_json(public / 'price_slices.json', slices)
    write_json(public / 'status.json', {'status': 'completed'})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
