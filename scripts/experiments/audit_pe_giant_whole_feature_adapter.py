"""Verify all whole-frame cache bindings and fixed label-blind GPU probes."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from preparation.split import pixel_hash
from pe_core_giant_whole_encoder import make_encoder

REFERENCE = '19_image_embedding_reference/pe_core_g14_448_whole_view/20261014T082013Z_1058c142'


def main():
    public = REPOSITORY / 'output/experiment_record' / REFERENCE
    assert read_json(public / 'status.json')['status'] == 'completed'
    registration = read_json(public / 'registration.json')
    key = read_json(public / 'cost.json')['feature_cache_key']
    cache = local_path('cache_root') / registration['cache_family'] / key
    manifest = read_json(cache / 'manifest.json')
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str})
    assert len(frame) == 8000 and manifest['imageids'] == frame.imageid.tolist()
    assert manifest['split_sha256'] == digest(local_path('split_path'))
    assert manifest['pixel_hashes'] == frame.pixel_hash.tolist()
    protocol = {'source_sha256': digest(__file__), 'feature_manifest_sha256': digest(cache / 'manifest.json'),
        'encoder_source_sha256': digest(Path(__file__).with_name('pe_core_giant_whole_encoder.py')),
        'split_sha256': digest(local_path('split_path')), 'probe_seed': 2026, 'probe_n': 12,
        'tolerances': {'rtol': 1e-4, 'atol': 3e-6},
        'test_images_used': False, 'price_labels_used_for_probe_selection': False}
    assert not (public / 'encoder_adapter_gpu_audit.json').exists()
    write_json(public / 'encoder_adapter_gpu_registration.json', protocol)
    hashes = read_json(cache / 'chunk_sha256.json')
    original = local_path('cache_root') / 'pe_core_giant_features' / manifest['original_cache_key']
    assert digest(original / 'manifest.json') == manifest['original_manifest_sha256']
    assert digest(original / 'chunk_sha256.json') == manifest['original_chunks_manifest_sha256']
    original_hashes = read_json(original / 'chunk_sha256.json')
    assert len(hashes) == len(original_hashes) == 8000
    parts, max_recovery = [], 0.
    for i, row in frame.iterrows():
        if pixel_hash(local_path('data_root') / 'train' / row.imageid) != row.pixel_hash:
            raise ValueError('A supplied training image changed')
        path = cache / f'{i:05d}_1.npy'
        original_path = original / path.name
        assert digest(path) == hashes[path.name]
        assert digest(original_path) == original_hashes[path.name]
        value = np.load(path, allow_pickle=False)
        assert value.shape == (1, 2560) and value.dtype == np.float32 and np.isfinite(value).all()
        np.testing.assert_allclose(np.linalg.norm(value, axis=1), np.ones(1), rtol=0, atol=1e-5)
        base = np.load(original_path, allow_pickle=False).astype(np.float64)
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        recovered = value[:, :1280].astype(np.float64)
        recovered /= np.linalg.norm(recovered, axis=1, keepdims=True)
        error = float(np.max(abs(recovered - base)))
        assert error < 1e-6
        max_recovery = max(max_recovery, error)
        parts.append(value)
    values = np.concatenate(parts).astype(np.float64)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    indices = np.random.default_rng(2026).choice(8000, 12, replace=False)
    images = []
    for name in frame.imageid.iloc[indices]:
        with Image.open(local_path('data_root') / 'train' / name) as source:
            images.append(ImageOps.exif_transpose(source).convert('RGB'))
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    actual = make_encoder(manifest)(images)
    np.testing.assert_allclose(actual, values[indices], rtol=1e-4, atol=3e-6)
    result = {'passed': True, 'all_8000_training_pixel_hashes_verified': True,
        'all_8000_cache_chunks_verified': True, 'all_original_branches_verified': True,
        'max_original_branch_recovery_error': max_recovery,
        'imageids': frame.imageid.iloc[indices].tolist(), 'gpu_feature_probe_n': 12,
        'max_abs_feature_replay_error': float(np.max(abs(actual - values[indices]))),
        'encoder_adapter_source_sha256': protocol['encoder_source_sha256'],
        'feature_manifest_sha256': protocol['feature_manifest_sha256'], 'test_images_used': False}
    write_json(public / 'encoder_adapter_gpu_audit.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
