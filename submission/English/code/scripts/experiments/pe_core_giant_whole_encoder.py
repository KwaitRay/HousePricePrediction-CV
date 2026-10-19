"""Inference for the pinned original-plus-whole PE-G feature contract."""
from pathlib import Path
import numpy as np
from common.config import digest, read_json
from common.local_paths import local_path
from pe_core_giant_encoder import make_encoder as make_base_encoder
from pe_giant_whole_view import whole_view


def make_encoder(manifest):
    source = Path(__file__).with_name('pe_giant_whole_view.py')
    if digest(source) != manifest['source_sha256']:
        raise ValueError('Registered whole-frame transformation source changed')
    for filename in ('pe_core_giant_encoder.py', 'pe_core_giant_preflight.py'):
        if digest(Path(__file__).with_name(filename)) != manifest['dependency_sha256'][filename]:
            raise ValueError('Registered whole-frame encoder dependency changed')
    if (manifest['cache_family'] != 'pe_core_giant_whole_features'
            or manifest['feature_dim'] != 2560 or manifest['batch_size'] != 1
            or manifest['mllm_components'] is not False):
        raise ValueError('Unsupported whole-frame feature contract')
    folder = local_path('cache_root') / 'pe_core_giant_features' / manifest['original_cache_key']
    if digest(folder / 'manifest.json') != manifest['original_manifest_sha256']:
        raise ValueError('Original feature manifest changed')
    if digest(folder / 'chunk_sha256.json') != manifest['original_chunks_manifest_sha256']:
        raise ValueError('Original chunk manifest changed')
    original = read_json(folder / 'manifest.json')
    for field in ('model', 'revision', 'checkpoint_sha256', 'preprocess', 'precision', 'split_sha256'):
        if manifest[field] != original[field]:
            raise ValueError('Whole-frame and original encoder identities differ: ' + field)
    base = make_base_encoder(original)

    def encode(images):
        values = np.concatenate([base(images), base([whole_view(im) for im in images])], axis=1)
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        # Match extraction storage rounding before the regression adapter's F64 normalization.
        values = values.astype(np.float32).astype(np.float64)
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)

    return encode
