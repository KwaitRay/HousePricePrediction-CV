"""Keep the original PE-Core-G view, add only one fixed whole-frame letterbox."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from pe_core_giant_encoder import make_encoder

ORIGINAL = '19_image_embedding_reference/pe_core_g14_448/20261014T065137Z_b822c3d4'
BASELINE = '15_fixed_ensembles/pe_giant_incremental_fusion/20261014T065752Z_88efe38c'
FAMILY = 'pe_core_giant_whole_features'


def whole_view(image):
    return ImageOps.pad(image, (448, 448), method=Image.Resampling.BICUBIC, color=(128, 128, 128))


def public_path(out):
    return REPOSITORY / 'output/experiment_record' / out.relative_to(local_path('records_root'))


def preregister():
    original_public = REPOSITORY / 'output/experiment_record' / ORIGINAL
    assert read_json(original_public / 'status.json')['status'] == 'completed'
    assert read_json(original_public / 'encoder_adapter_gpu_audit.json')['passed']
    original_registration = read_json(original_public / 'registration.json')
    cache = local_path('cache_root') / 'pe_core_giant_features' / original_registration['cache_key']
    manifest = read_json(cache / 'manifest.json')
    assert manifest['feature_dim'] == 1280 and manifest['batch_size'] == 1
    assert manifest['mllm_components'] is False and manifest['test_images_used'] is False
    relative = Path('19_image_embedding_reference/pe_core_g14_448_whole_view') / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    out = local_path('records_root') / relative
    out.mkdir(parents=True, exist_ok=False)
    registration = {'method': 'Original PE-Core-G plus exactly one whole-frame letterbox view',
        'source_sha256': digest(__file__), 'original_reference': ORIGINAL,
        'original_cache_key': original_registration['cache_key'], 'original_manifest_sha256': digest(cache / 'manifest.json'),
        'original_chunks_manifest_sha256': digest(cache / 'chunk_sha256.json'),
        'dependency_sha256': {name: digest(Path(__file__).with_name(name)) for name in
            ('pe_core_giant_encoder.py', 'pe_core_giant_preflight.py', 'frozen_kernel_search.py')},
        'split_sha256': digest(local_path('split_path')), 'model': manifest['model'], 'revision': manifest['revision'],
        'cache_family': FAMILY, 'feature_dim': 2560, 'batch_size': 1,
        'views': ['Unchanged cached official original image vector', 'Whole EXIF RGB image letterboxed to 448 square, bicubic, RGB128 border, then unchanged official processor'],
        'view_weighting': 'Equal-weight concatenation of two unit 1280D vectors, then unit normalization, stored FP32',
        'comparison': 'One added view only; preserve original branch, checkpoint, precision, official processor and target. Feature dimension increases from 1280 to 2560.',
        'baseline_fusion': BASELINE, 'baseline_ensemble_sha256': digest(local_path('records_root') / BASELINE / 'ensemble.json'),
        'kernel_grid': {'gammas': [1., 4., 16.], 'alphas': [.01, .1, 1., 10.], 'group_folds': 3},
        'fusion_gate': {'minimum_meta_cv_relative_gain': .01, 'minimum_folds_won': 2, 'require_fixed_validation_improvement': True},
        'selection': 'Unweighted original-price training grouped OOF MSE, then fixed validation. Incremental fusion against fixed 53462.04 scheme only after ordinary kernel completes.',
        'caveat': 'Crop coverage does not prove price information was discarded; pad changes native detail scale. Base and development selection bias remains.',
        'test_images_used': False, 'test_labels_used': False, 'mllm_components': False,
        'automatic_test_export': False}
    write_json(public_path(out) / 'registration.json', registration)
    write_json(public_path(out) / 'status.json', {'status': 'preregistered'})
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    for name in registration['dependency_sha256']:
        (out / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    print(out, flush=True)


def extract(out):
    out = out.resolve()
    public = public_path(out)
    registration = read_json(public / 'registration.json')
    assert read_json(public / 'status.json')['status'] == 'preregistered'
    assert digest(__file__) == registration['source_sha256']
    assert digest(local_path('split_path')) == registration['split_sha256']
    for name, sha in registration['dependency_sha256'].items():
        assert digest(Path(__file__).with_name(name)) == sha
    def status(state, **extra):
        write_json(public / 'status.json', {'status': state, 'pid': os.getpid(), **extra})
    started = time.perf_counter()
    try:
        frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
        train = frame.partition.eq('train')
        assert len(frame) == 8000 and train.sum() == 6399
        assert not set(frame.loc[train, 'group_id']) & set(frame.loc[~train, 'group_id'])
        original_cache = local_path('cache_root') / 'pe_core_giant_features' / registration['original_cache_key']
        assert digest(original_cache / 'manifest.json') == registration['original_manifest_sha256']
        assert digest(original_cache / 'chunk_sha256.json') == registration['original_chunks_manifest_sha256']
        original_manifest = read_json(original_cache / 'manifest.json')
        assert original_manifest['imageids'] == frame.imageid.tolist()
        hashes = read_json(original_cache / 'chunk_sha256.json')
        original = []
        for offset in range(8000):
            path = original_cache / f'{offset:05d}_1.npy'
            assert digest(path) == hashes[path.name]
            value = np.load(path, allow_pickle=False)
            assert value.shape == (1, 1280) and np.isfinite(value).all()
            original.append(value)
        original = np.concatenate(original).astype(np.float64)
        original /= np.maximum(np.linalg.norm(original, axis=1, keepdims=True), 1e-12)
        manifest = {**registration, 'imageids': frame.imageid.tolist(), 'pixel_hashes': frame.pixel_hash.tolist(),
            'checkpoint_sha256': original_manifest['checkpoint_sha256'], 'preprocess': original_manifest['preprocess'],
            'precision': original_manifest['precision'], 'stored_feature': 'Two unit vectors concatenated, normalized and stored FP32',
            'course_scope': 'Generic image tower and provided images/labels only; no MLLM or additional SoCal data'}
        key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        cache = local_path('cache_root') / FAMILY / key
        cache.mkdir(parents=True, exist_ok=False)
        write_json(cache / 'manifest.json', manifest)
        write_json(out / 'data_manifest.json', manifest)
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        torch.cuda.reset_peak_memory_stats()
        status('loading_gpu')
        encode = make_encoder(original_manifest)
        def image_for(name):
            with Image.open(local_path('data_root') / 'train' / name) as image:
                return ImageOps.exif_transpose(image).convert('RGB')
        positions = np.flatnonzero(train.to_numpy())[:3]
        probes = [image_for(frame.imageid.iloc[i]) for i in positions]
        actual = encode(probes)
        np.testing.assert_allclose(actual, original[positions], rtol=1e-4, atol=3e-6)
        whole = encode([whole_view(image) for image in probes])
        np.testing.assert_allclose(whole, encode([whole_view(image) for image in probes]), rtol=1e-4, atol=3e-6)
        assert whole.shape == (3, 1280) and np.isfinite(whole).all()
        write_json(public / 'gpu_preflight.json', {'passed': True, 'probe_imageids': frame.imageid.iloc[positions].tolist(),
            'original_replay_max_abs_error': float(np.max(abs(actual - original[positions]))),
            'whole_repeat_passed': True, 'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
            'price_labels_used_to_choose_views': False})
        checksums, max_recovery_error = {}, 0.
        for offset, name in enumerate(frame.imageid):
            added = encode([whole_view(image_for(name))])
            features = np.concatenate([original[offset:offset + 1], added], axis=1)
            features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
            features = features.astype(np.float32)
            assert features.shape == (1, 2560) and np.isfinite(features).all()
            np.testing.assert_allclose(np.linalg.norm(features, axis=1), np.ones(1), rtol=0, atol=1e-5)
            recovered = features[:, :1280].astype(np.float64)
            recovered /= np.linalg.norm(recovered, axis=1, keepdims=True)
            delta = float(np.max(abs(recovered - original[offset:offset + 1])))
            assert delta < 1e-6
            max_recovery_error = max(max_recovery_error, delta)
            path = cache / f'{offset:05d}_1.npy'
            with path.open('xb') as stream:
                np.save(stream, features)
            checksums[path.name] = digest(path)
            if offset % 100 == 0:
                write_json(cache / 'chunk_sha256.json', checksums)
                status('extracting', images=offset + 1, total=8000, max_original_branch_recovery_error=max_recovery_error)
                print(f'PE-Core-G original + whole: {offset + 1}/8000', flush=True)
        write_json(cache / 'chunk_sha256.json', checksums)
        for name, sha in checksums.items():
            assert digest(cache / name) == sha
        del encode
        torch.cuda.empty_cache()
        write_json(public / 'cost.json', {'extraction_seconds': time.perf_counter() - started,
            'feature_cache_key': key, 'chunk_manifest_sha256': digest(cache / 'chunk_sha256.json'),
            'max_original_branch_recovery_error': max_recovery_error,
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30})
        source = Path(__file__).with_name('frozen_kernel_search.py')
        assert digest(source) == registration['dependency_sha256'][source.name]
        status('fitting_kernel', feature_extraction_completed=True)
        command = [sys.executable, '-X', 'utf8', str(source), '--name', 'pe_core_g14_448_whole_kernel',
            '--cache-family', FAMILY, '--cache-key', key]
        result = subprocess.run(command, check=False)
        write_json(public / 'kernel_queue.json', {'command': command, 'exit_code': result.returncode})
        if result.returncode:
            raise RuntimeError('PE-Core-G whole-view grouped kernel search failed')
        status('completed', feature_extraction_completed=True, price_model_evaluated=True)
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['preregister', 'extract'], required=True)
    parser.add_argument('--protocol-dir', type=Path)
    args = parser.parse_args()
    if args.phase == 'preregister':
        preregister()
    elif args.protocol_dir:
        extract(args.protocol_dir)
    else:
        parser.error('extract requires protocol-dir')
