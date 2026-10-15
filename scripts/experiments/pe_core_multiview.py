"""Preregister PE-Core original/whole/detail features without changing active exporters."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY

REFERENCE = '19_image_embedding_reference/pe_core_l14_336/20261014T024049Z_caec475a'
MODEL = 'timm/vit_pe_core_large_patch14_336.fb'
REVISION = 'e63206c8e3a0e9b699e40f31080eebd78fd2258e'
FAMILY = 'pe_core_multiview'


def views(images):
    whole = [ImageOps.pad(im, (336, 336), method=Image.Resampling.BICUBIC,
        color=(128, 128, 128)) for im in images]
    detail = [im.crop((round(im.width * .2), round(im.height * .2),
        round(im.width * .8), round(im.height * .8))) for im in images]
    return whole, detail


def public_path(out):
    return REPOSITORY / 'output/experiment_record' / out.relative_to(local_path('records_root'))


def preregister():
    reference = local_path('records_root') / REFERENCE / 'data_manifest.json'
    original = read_json(reference)
    if original['model'] != MODEL or original['revision'] != REVISION or original['feature_dim'] != 1024:
        raise ValueError('Only the completed pinned PE-Core image encoder is supported')
    if original.get('mllm_components') is not False or original.get('test_images_used') is not False:
        raise ValueError('Only provided training images and generic visual features are allowed')
    original_key = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    original_cache = local_path('cache_root') / 'pe_core_features' / original_key
    if digest(original_cache / 'manifest.json') != digest(reference):
        raise ValueError('Completed original PE feature manifest differs')
    relative = Path('19_image_embedding_reference/pe_core_l14_336_multiview') / (
        time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    out = local_path('records_root') / relative
    out.mkdir(parents=True, exist_ok=False)
    dependencies = {name: digest(Path(__file__).with_name(name)) for name in
        ('visual_kernel_inference.py', 'visual_kernel_submission.py')}
    registration = {'source_sha256': digest(__file__), 'dependency_sha256': dependencies,
        'original_reference': REFERENCE, 'original_manifest_sha256': digest(reference),
        'original_cache_key': original_key, 'split_sha256': digest(local_path('split_path')),
        'model': MODEL, 'revision': REVISION, 'cache_family': FAMILY, 'feature_dim': 3072,
        'views': ['Original official PE-Core preprocessing, unchanged cached full-image vector',
            'Whole EXIF image letterboxed to 336 square with RGB128 border',
            'Fixed center 60% width and height, official PE-Core preprocessing'],
        'view_weighting': 'Equal-weight concatenation of three unit 1024D vectors, then unit normalization',
        'precision': original['precision'], 'batch_size': 4,
        'next_kernel_protocol': {'gammas': [1., 4., 16.], 'alphas': [.01, .1, 1., 10.],
            'selection': 'Original unweighted clipped training OOF MSE, three-fold GroupKFold; validation only after selection'},
        'test_images_used': False, 'test_labels_used': False, 'text_modules_used': False,
        'mllm_components': False, 'course_scope': 'Generic pretrained visual tower and provided images only; no additional SoCal data',
        'scope': 'Feature extraction first; downstream kernel and fusion require an explicit registered adapter',
        'caveat': 'Full-image branch is retained; windows do not depend on price or errors; upsampling does not recover absent detail'}
    write_json(public_path(out) / 'registration.json', registration)
    write_json(public_path(out) / 'status.json', {'status': 'preregistered'})
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    for name in dependencies:
        (out / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    print(out, flush=True)
    return out


def extract(out):
    import torch
    from visual_kernel_inference import encoder
    from visual_kernel_submission import cache_array
    out = out.resolve()
    public = public_path(out)
    registration = read_json(public / 'registration.json')
    if digest(__file__) != registration['source_sha256'] or digest(local_path('split_path')) != registration['split_sha256']:
        raise ValueError('Preregistered source or split changed')
    for name, expected in registration['dependency_sha256'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Preregistered shared encoder dependency changed')
    if read_json(public / 'status.json')['status'] != 'preregistered':
        raise ValueError('No automatic repeat of a feature extraction')
    def status(state, **extra):
        write_json(public / 'status.json', {'status': state, 'pid': os.getpid(), **extra})
    started = time.perf_counter()
    try:
        frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
        train, valid = frame.partition.eq('train'), frame.partition.eq('validation')
        if len(frame) != 8000 or train.sum() != 6399 or valid.sum() != 1601 or set(frame.loc[train, 'group_id']) & set(frame.loc[valid, 'group_id']):
            raise ValueError('Unexpected frozen split')
        original, original_manifest = cache_array('pe_core_features', registration['original_cache_key'], frame)
        reference = local_path('records_root') / registration['original_reference'] / 'data_manifest.json'
        if digest(reference) != registration['original_manifest_sha256'] or original.shape != (8000, 1024):
            raise ValueError('Original full-image features changed')
        np.testing.assert_allclose(np.linalg.norm(original, axis=1), 1, rtol=0, atol=1e-5)
        manifest = {**registration, 'imageids': frame.imageid.tolist(), 'pixel_hashes': frame.pixel_hash.tolist(),
            'checkpoint_sha256': original_manifest['checkpoint_sha256'],
            'preprocess': original_manifest['preprocess'], 'stored_feature': 'All three views concatenated, normalized, float32'}
        key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        cache = local_path('cache_root') / FAMILY / key
        cache.mkdir(parents=True, exist_ok=False)
        write_json(cache / 'manifest.json', manifest)
        write_json(out / 'data_manifest.json', manifest)
        def images(names):
            result = []
            for name in names:
                with Image.open(local_path('data_root') / 'train' / name) as im:
                    result.append(ImageOps.exif_transpose(im).convert('RGB'))
            return result
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        torch.cuda.reset_peak_memory_stats()
        status('loading_vision')
        encode = encoder('pe_core_features', original_manifest)
        probe_positions = np.flatnonzero(train.to_numpy())[:4]
        probes = images(frame.imageid.iloc[probe_positions])
        reproduced = encode(probes)
        np.testing.assert_allclose(reproduced, original[probe_positions], rtol=1e-4, atol=1e-5)
        whole, detail = views(probes)
        detail_features = encode(detail)
        np.testing.assert_allclose(encode(detail), detail_features, rtol=1e-6, atol=1e-6)
        if encode(whole).shape != (4, 1024) or not np.isfinite(detail_features).all():
            raise ValueError('New views failed numerical preflight')
        write_json(public / 'gpu_preflight.json', {'passed': True,
            'full_image_probe_n': 4, 'original_feature_replay_passed': True,
            'detail_repeat_passed': True, 'batch_size': 4,
            'probe_imageids': frame.imageid.iloc[probe_positions].tolist(),
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20})
        for offset in range(0, 8000, 4):
            rows = frame.iloc[offset:offset + 4]
            whole, detail = views(images(rows.imageid))
            features = np.concatenate([original[offset:offset + len(rows)], encode(whole), encode(detail)], axis=1)
            features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
            features = features.astype(np.float32)
            if features.shape != (len(rows), 3072) or not np.isfinite(features).all():
                raise ValueError('Invalid combined PE-Core features')
            np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1, rtol=0, atol=1e-5)
            # The normalized original full-image branch must remain identifiable.
            reconstructed = features[:, :1024].astype(np.float64)
            reconstructed /= np.linalg.norm(reconstructed, axis=1, keepdims=True)
            np.testing.assert_allclose(reconstructed, original[offset:offset + len(rows)], rtol=1e-4, atol=1e-6)
            with (cache / f'{offset:05d}_{len(rows)}.npy').open('xb') as stream:
                np.save(stream, features)
            if offset % 400 == 0:
                status('extracting', images=offset + len(rows), total=8000)
                print(f'PE-Core full/whole/detail features {offset + len(rows)}/8000', flush=True)
        write_json(public / 'cost.json', {'extraction_seconds': time.perf_counter() - started,
            'feature_cache_key': key, 'feature_dim': 3072,
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20})
        status('completed', feature_cache_key=key, feature_extraction_completed=True,
            price_model_evaluated=False, next_phase='Registered multiview kernel adapter and OOF fusion')
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
        parser.error('extract requires protocol directory')
