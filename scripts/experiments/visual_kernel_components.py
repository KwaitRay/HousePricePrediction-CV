"""Explicit adapters for frozen image kernels and supervised PLS product kernels."""
from pathlib import Path
import joblib
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from common.config import digest, read_json
from common.local_paths import local_path, REPOSITORY

PLS_METHOD = 'Image RBF multiplied by fold-fitted supervised PLS-score RBF'


def feature_registration(registration):
    if registration.get('method') == 'Fold-fitted PCA partial-whitening image RBF with training-median bandwidth calibration':
        source = registration['source_kernel']
        public = REPOSITORY/'output/experiment_record'/source
        if read_json(public/'status.json')['status'] != 'completed':
            raise ValueError('PCA source is not completed')
        if digest(public/'registration.json') != registration['source_registration_sha256'] or digest(local_path('records_root')/source/'head.npz') != registration['source_head_sha256']:
            raise ValueError('PCA source binding changed')
        base = read_json(public/'registration.json')
        if base.get('cache_family') != 'pe_core_giant_whole_features' or registration['split_sha256'] != digest(local_path('split_path')):
            raise ValueError('Unsupported PCA source or split')
        return base
    if registration.get('method') == 'Price-band weighted image RBF with fold-fitted normalized weights and weighted target mean':
        raise ValueError('Price-band weighted KRR requires its explicit full-refit adapter; ordinary RBF refit is not equivalent')
    if registration.get('method') != PLS_METHOD:
        if 'cache_family' not in registration:
            raise ValueError('Unsupported kernel protocol; an explicit adapter is required')
        return registration
    if registration.get('mllm_components') is not False:
        raise ValueError('Only the registered visual-only PLS protocol is supported')
    if registration['split_sha256'] != digest(local_path('split_path')):
        raise ValueError('PLS source split changed')
    source = local_path('records_root') / registration['source']
    public = REPOSITORY / 'output/experiment_record' / registration['source']
    if read_json(public / 'status.json')['status'] != 'completed':
        raise ValueError('PLS image source is not completed')
    if digest(source / 'head.npz') != registration['source_head_sha256']:
        raise ValueError('PLS image source head changed')
    base = read_json(public / 'registration.json')
    if base['manifest_sha256'] != registration['source_feature_manifest_sha256']:
        raise ValueError('PLS image source manifest binding changed')
    family = base.get('cache_family')
    if family not in {'siglip_features', 'pe_core_multiview'}:
        raise ValueError('PLS adapter supports registered SigLIP single-view or PE multiview sources')
    if family == 'pe_core_multiview':
        for field in ('cache_family', 'feature_cache_key', 'manifest_sha256'):
            if registration.get(field) != base.get(field):
                raise ValueError('PE PLS feature source binding differs')
        manifest_path = local_path('cache_root') / family / base['feature_cache_key'] / 'manifest.json'
        manifest = read_json(manifest_path)
        if digest(manifest_path) != base['manifest_sha256']:
            raise ValueError('PE PLS source feature manifest changed')
        if manifest.get('mllm_components') is not False or manifest.get('test_images_used') is not False:
            raise ValueError('PE PLS requires frozen training-only visual features')
    return base


def scores(metric, values, components):
    return metric['scaler'].transform(metric['pls'].transform(values))[:, :components]


def kernel_for(head, values, metric=None):
    distance = np.maximum(2 - 2 * (values @ head['train_features'].T), 0)
    if 'gamma' in head:
        return np.exp(-float(head['gamma']) * distance)
    if not {'image_gamma', 'pls_gamma', 'components', 'train_pls_scores'} <= set(head):
        raise ValueError('Unknown kernel head format')
    kernel = np.exp(-float(head['image_gamma']) * distance)
    components = int(head['components'])
    if components:
        if metric is None:
            raise ValueError('The supervised metric is required for this kernel')
        query = scores(metric, values, components)
        kernel *= np.exp(-float(head['pls_gamma']) * cdist(query, head['train_pls_scores'], 'sqeuclidean') / components)
    return kernel


def replay_and_refit(source, registration, values, train, valid, y, path, reference_raw):
    """Replay the original validation head before fitting the frozen recipe on 8000 labels."""
    if registration.get('method') == 'Fold-fitted PCA partial-whitening image RBF with training-median bandwidth calibration':
        from pca_kernel_component import replay_and_refit as pca_refit
        return pca_refit(source,registration,values,train,valid,y,path,reference_raw)
    is_pls = registration.get('method') == PLS_METHOD
    metadata = {'kernel_type': 'pls_product_rbf' if is_pls else 'image_rbf'}
    with threadpool_limits(limits=2):
        with np.load(source / 'head.npz', allow_pickle=False) as previous:
            if 'training_weights' in previous.files:
                raise ValueError('Weighted kernels require a separate full-refit protocol')
            if is_pls != ('image_gamma' in previous.files):
                raise ValueError('Head format does not match the registered kernel method')
            np.testing.assert_allclose(values[train], previous['train_features'], rtol=0, atol=1e-12)
            metric = None
            components = int(previous['components']) if is_pls else 0
            if components:
                metric_path = source / 'feature_metric.joblib'
                metadata['reference_metric_sha256'] = digest(metric_path)
                metric = joblib.load(metric_path)
                if metric['pls'].n_components != 16 or not metric['pls'].scale or metric['pls'].max_iter != 500:
                    raise ValueError('Unexpected fitted PLS recipe')
                if components not in (4, 16):
                    raise ValueError('Unsupported selected PLS component count')
                np.testing.assert_allclose(scores(metric, values[train], components), previous['train_pls_scores'], rtol=0, atol=1e-10)
            raw = (kernel_for(previous, values[valid], metric) @ previous['coefficients'] + float(previous['target_mean'])) * 1000
            alpha = float(previous['alpha'])
            if is_pls:
                fields = {'image_gamma': float(previous['image_gamma']), 'pls_gamma': float(previous['pls_gamma']), 'components': components}
            else:
                fields = {'gamma': float(previous['gamma'])}
        np.testing.assert_allclose(raw, reference_raw, rtol=1e-7, atol=1e-4)
        # Labels enter the supervised transform only after all selected settings are frozen.
        full_metric = None
        if components:
            pls = PLSRegression(n_components=16, scale=True, max_iter=500).fit(values, y)
            scaler = StandardScaler().fit(pls.transform(values))
            full_metric = {'pls': pls, 'scaler': scaler}
            metric_path = path.with_name(path.stem.replace('_head', '_metric') + '.joblib')
            joblib.dump(full_metric, metric_path)
            metadata.update(full_metric_file=metric_path.name, full_metric_sha256=digest(metric_path), pls_fit_components=16)
        full_head = {'train_features': values, **fields}
        if is_pls:
            full_head['train_pls_scores'] = scores(full_metric, values, components) if components else np.empty((len(values), 0))
        kernel = kernel_for(full_head, values, full_metric)
        kernel.flat[::len(values) + 1] += alpha
        mean = float(y.mean())
        coefficients = solve(kernel, y - mean, assume_a='pos', check_finite=False, overwrite_a=True)
        np.savez_compressed(path, **full_head, coefficients=coefficients, alpha=alpha, target_mean=mean)
    metadata.update(**fields, alpha=alpha, feature_dim=values.shape[1], full_head_sha256=digest(path))
    return raw, metadata


def predict_component(out, index, component, features):
    path = out / f'component_{index}_full_head.npz'
    if digest(path) != component['full_head_sha256']:
        raise ValueError('Frozen full-training kernel changed')
    metric = None
    if 'full_pca_file' in component:
        from pca_kernel_component import apply_transform
        pca_path=out/component['full_pca_file']
        if digest(pca_path) != component['full_pca_sha256'] or digest(Path(__file__).with_name('pca_kernel_component.py')) != component['pca_adapter_sha256'] or digest(Path(__file__).with_name('pe_whole_pca_bandwidth_kernel.py')) != component['pca_recipe_source_sha256']:
            raise ValueError('Frozen PCA transform/implementation changed')
        with np.load(pca_path,allow_pickle=False) as transform:
            features=apply_transform(features,transform)
    if 'full_metric_file' in component:
        metric_path = out / component['full_metric_file']
        if digest(metric_path) != component['full_metric_sha256']:
            raise ValueError('Frozen full-training PLS metric changed')
        metric = joblib.load(metric_path)
    with np.load(path, allow_pickle=False) as head, threadpool_limits(limits=2):
        chunks = [(kernel_for(head, features[offset:offset + 128], metric) @ head['coefficients'] + float(head['target_mean'])) * 1000
                  for offset in range(0, len(features), 128)]
    return np.maximum(np.concatenate(chunks), 0)
