"""Independently recompute all exported prices from signed test features and full heads."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY


def main(out):
    out = out.resolve()
    public = REPOSITORY / 'output/experiment_record' / out.relative_to(local_path('records_root'))
    if read_json(public / 'status.json')['status'] != 'completed':
        raise ValueError('Test inference has not completed')
    registration = read_json(out / 'registration.json')
    checks = read_json(public / 'checks.json')
    launch = out / 'inference_source_at_launch.py'
    if digest(launch) != registration['inference_source_sha256']:
        raise ValueError('Inference source snapshot differs from its registered version')
    data = local_path('data_root')
    if digest(data / 'test.csv') != registration['test_csv_sha256'] or digest(data / 'train.csv') != registration['train_csv_sha256']:
        raise ValueError('Supplied CSV files changed')
    if digest(local_path('split_path')) != registration['split_sha256']:
        raise ValueError('Frozen development split changed')
    test = pd.read_csv(data / 'test.csv', dtype={'imageid': str})
    submission = pd.read_csv(out / 'predictions.csv', dtype={'imageid': str})
    if len(test) != 3000 or not test.imageid.is_unique or list(submission) != ['imageid', 'price'] or submission.imageid.tolist() != test.imageid.tolist():
        raise ValueError('Submission coverage, columns or order failed')
    if not np.isfinite(submission.price).all() or (submission.price < 0).any() or digest(out / 'predictions.csv') != checks['csv_sha256']:
        raise ValueError('Submission values or checksum failed')
    test_manifest = read_json(out / 'test_manifest.json')
    test_sha = digest(out / 'test_manifest.json')
    if test_manifest['imageids'] != test.imageid.tolist() or len(test_manifest['file_sha256']) != 3000 or test_sha != checks['test_manifest_sha256']:
        raise ValueError('Test image manifest differs')
    for name, expected in zip(test.imageid, test_manifest['file_sha256']):
        if digest(data / 'test' / name) != expected:
            raise ValueError('A supplied test image changed')
    outputs, audit = [], []
    for index, component in enumerate(registration['components']):
        family, key = component['cache_family'], component['cache_key']
        source_registration = REPOSITORY / 'output/experiment_record' / component['source'] / 'registration.json'
        if digest(source_registration) != component['source_registration_sha256']:
            raise ValueError('Frozen source registration changed')
        feature_manifest_path = local_path('cache_root') / family / key / 'manifest.json'
        manifest_sha = digest(feature_manifest_path)
        if manifest_sha != component['feature_manifest_sha256'] or not checks['labeled_feature_spotchecks'].get(family + ':' + key):
            raise ValueError('Encoder feature binding or labeled probe evidence failed')
        feature_source_sha = component.get('feature_inference_source_sha256', registration['inference_source_sha256'])
        reuse = component.get('feature_cache_reuse')
        if reuse:
            prior = (local_path('records_root') / reuse['reference']).resolve()
            prior.relative_to(local_path('records_root').resolve())
            prior_public = REPOSITORY / 'output/experiment_record' / prior.relative_to(local_path('records_root'))
            for filename, field in [('checks.json', 'checks_sha256'), ('independent_export_audit.json', 'audit_sha256'), ('registration.json', 'registration_sha256')]:
                if digest(prior_public / filename) != reuse[field]:
                    raise ValueError('Inherited cache evidence changed')
            prior_checks = read_json(prior_public / 'checks.json')
            prior_audit = read_json(prior_public / 'independent_export_audit.json')
            prior_registration = read_json(prior_public / 'registration.json')
            if not prior_audit['passed'] or digest(prior / 'predictions.csv') != prior_checks['csv_sha256']:
                raise ValueError('Inherited feature export was not independently verified')
            if prior_checks['test_manifest_sha256'] != test_sha or not prior_checks['labeled_feature_spotchecks'].get(family + ':' + key):
                raise ValueError('Inherited image or labeled probe binding differs')
            candidates = [c for c in prior_registration['components'] if c['cache_family'] == family and c['cache_key'] == key]
            if len(candidates) != 1 or candidates[0]['feature_manifest_sha256'] != manifest_sha:
                raise ValueError('Inherited encoder source differs')
            prior_source_sha = candidates[0].get('feature_inference_source_sha256', prior_registration['inference_source_sha256'])
            if feature_source_sha != prior_source_sha:
                raise ValueError('Inherited feature inference source differs')
        elif feature_source_sha != registration['inference_source_sha256']:
            raise ValueError('Different feature source requires explicit verified reuse evidence')
        batch = int(read_json(feature_manifest_path).get('batch_size', 4))
        cache_manifest = {'family': family, 'source_cache_key': key, 'source_manifest_sha256': manifest_sha,
            'test_manifest_sha256': test_sha, 'inference_source_sha256': feature_source_sha,
            'feature_dim': component['feature_dim'], 'batch_size': batch, 'test_labels_used': False}
        cache_key = hashlib.sha256(json.dumps(cache_manifest, sort_keys=True).encode()).hexdigest()
        cache = local_path('cache_root') / 'test_visual_features' / cache_key
        if read_json(cache / 'manifest.json') != cache_manifest:
            raise ValueError('Cached test feature contract changed')
        chunk_checksums = read_json(cache / 'chunk_sha256.json')
        chunks = []
        for offset in range(0, 3000, batch):
            count = min(batch, 3000 - offset)
            path = cache / f'{offset:05d}_{count}.npy'
            if digest(path) != chunk_checksums.get(path.name):
                raise ValueError('A test feature chunk changed')
            values = np.load(path, allow_pickle=False)
            if values.shape != (count, component['feature_dim']) or not np.isfinite(values).all():
                raise ValueError('Invalid test feature shape or values')
            np.testing.assert_allclose(np.linalg.norm(values, axis=1), np.ones(count), rtol=0, atol=1e-8)
            chunks.append(values)
        features = np.concatenate(chunks)
        if 'full_pca_file' in component:
            transform_path=out/component['full_pca_file']
            if digest(transform_path) != component['full_pca_sha256']:
                raise ValueError('Full-training PCA transform changed')
            with np.load(transform_path,allow_pickle=False) as transform, threadpool_limits(limits=2):
                features=(features-transform['mean'])@transform['vectors']
                features/=np.maximum(transform['variance'],float(transform['variance'][0])*1e-8)**(float(transform['whitening_power'])/2)
                feature_norms=np.sqrt(np.sum(features*features,axis=1,keepdims=True))
                if (feature_norms<=1e-12).any() or not np.isfinite(features).all():
                    raise ValueError('Invalid independently transformed PCA features')
                features/=feature_norms
        head_path = out / f'component_{index}_full_head.npz'
        if digest(head_path) != component['full_head_sha256']:
            raise ValueError('Full-training head changed')
        metric = None
        if 'full_metric_file' in component:
            path = out / component['full_metric_file']
            if digest(path) != component['full_metric_sha256']:
                raise ValueError('Supervised feature transform changed')
            metric = joblib.load(path)
        with np.load(head_path, allow_pickle=False) as saved:
            head = {name: saved[name] for name in saved.files}
        with threadpool_limits(limits=2):
            if head['train_features'].shape != (8000, component.get('head_feature_dim',component['feature_dim'])) or head['coefficients'].shape != (8000,):
                raise ValueError('Unexpected full-training coverage')
            parts = []
            for offset in range(0, 3000, 128):
                query = features[offset:offset + 128]
                distance = np.maximum(2 - 2 * (query @ head['train_features'].T), 0)
                gamma = float(head['image_gamma']) if 'image_gamma' in head else float(head['gamma'])
                kernel = np.exp(-gamma * distance)
                if metric is not None:
                    dimensions = int(head['components'])
                    scores = metric['scaler'].transform(metric['pls'].transform(query))[:, :dimensions]
                    kernel *= np.exp(-float(head['pls_gamma']) * cdist(scores, head['train_pls_scores'], 'sqeuclidean') / dimensions)
                parts.append((kernel @ head['coefficients'] + float(head['target_mean'])) * 1000)
            outputs.append(np.maximum(np.concatenate(parts), 0))
        audit.append({'component': index, 'source': component['source'], 'feature_rows': 3000,
            'cached_feature_chunks_verified': len(chunks), 'full_head_sha256': component['full_head_sha256']})
        print(f'Independent 3000-price replay completed for component {index + 1}', flush=True)
    weights = np.asarray(registration['weights'], dtype=np.float64)
    if weights.shape != (len(outputs),) or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError('Invalid frozen composition weights')
    prediction = np.column_stack(outputs) @ weights
    np.testing.assert_allclose(prediction, submission.price.to_numpy(), rtol=1e-10, atol=1e-7)
    write_json(public / 'independent_export_audit.json', {'passed': True, 'n': 3000,
        'all_raw_test_image_checksums_verified': True, 'all_test_feature_chunks_verified': True,
        'all_exported_prices_replayed': True, 'max_abs_csv_replay_error': float(np.max(np.abs(prediction - submission.price.to_numpy()))),
        'csv_sha256': digest(out / 'predictions.csv'), 'components': audit,
        'source_sha256': digest(__file__), 'scope': 'CPU audit of all cached feature rows and price formulas, with recorded fresh or explicitly verified inherited labeled probes; no new encoder extraction',
        'test_labels_used': False, 'kaggle_uploaded': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared-run', type=Path, required=True)
    main(parser.parse_args().prepared_run)
