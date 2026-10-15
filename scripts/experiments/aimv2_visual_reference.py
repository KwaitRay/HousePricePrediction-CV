"""Preregister one image-only AIMv2 readout; extract, audit, then grouped KRR."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
import transformers
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from preparation.split import pixel_hash
from aimv2_visual_encoder import REPO, REVISION, make_encoder

FAMILY = 'aimv2_features'
PREFLIGHT = '19_image_embedding_reference/aimv2_large_224_preflight/20261014T091339Z_8e4c72b3'
BASELINE = '15_fixed_ensembles/pe_giant_whole_incremental_fusion/20261014T082727Z_c5152cef'


def preregister():
    preflight = REPOSITORY / 'output/experiment_record' / PREFLIGHT
    checked = read_json(preflight / 'registration.json')
    results = read_json(preflight / 'results.json')
    assert read_json(preflight / 'status.json')['status'] == 'completed' and results['passed']
    assert results['strict_checkpoint_loading'] and results['repeat_max_abs_error'] == 0
    assert digest(Path(__file__).with_name('aimv2_visual_encoder.py')) == checked['encoder_source_sha256']
    folder = local_path('cache_root') / 'aimv2_weights' / REVISION
    provenance = read_json(folder / 'download.json')
    assert provenance['repo'] == REPO and provenance['revision'] == REVISION
    assert provenance['files']['model.safetensors']['publisher_lfs_verified']
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    assert len(frame) == 8000 and frame.partition.eq('train').sum() == 6399
    assert not set(frame.loc[frame.partition.eq('train'), 'group_id']) & set(frame.loc[frame.partition.eq('validation'), 'group_id'])
    native = Path(transformers.__file__).parent / 'models/aimv2/modeling_aimv2.py'
    manifest = {'model': REPO, 'revision': REVISION, 'cache_family': FAMILY, 'feature_dim': 1024, 'batch_size': 1,
        'checkpoint_sha256': digest(folder / 'model.safetensors'), 'config_sha256': digest(folder / 'config.json'),
        'preprocess': results['preprocess'], 'feature': checked['feature'], 'precision': checked['precision'],
        'source_sha256': digest(__file__), 'encoder_source_sha256': checked['encoder_source_sha256'],
        'kernel_source_sha256': digest(Path(__file__).with_name('frozen_kernel_search.py')),
        'native_implementation_sha256': digest(native), 'transformers_version': transformers.__version__,
        'torch_version': torch.__version__, 'preflight_reference': PREFLIGHT,
        'preflight_registration_sha256': digest(preflight / 'registration.json'),
        'preflight_results_sha256': digest(preflight / 'results.json'), 'license': provenance['license'],
        'imageids': frame.imageid.tolist(), 'pixel_hashes': frame.pixel_hash.tolist(),
        'split_sha256': digest(local_path('split_path')), 'train_csv_sha256': digest(local_path('data_root') / 'train.csv'),
        'mllm_components': False, 'text_modules_used': False, 'remote_code_executed': False,
        'test_images_used': False, 'test_labels_used': False,
        'baseline_fusion': BASELINE, 'baseline_ensemble_sha256': digest(local_path('records_root') / BASELINE / 'ensemble.json'),
        'hypothesis': 'Different generic image pretraining may complement PE-G whole-frame and existing PLS/NaFlex residuals',
        'comparison': 'Complete fixed encoder method; pretraining, resolution and processor differ. No causal isolation of training objective.',
        'selection': 'One predetermined mean-patch readout, no CLS, no layer/window sweep; grouped three-fold original-price OOF KRR',
        'kernel_grid': {'gammas': [1., 4., 16.], 'alphas': [.01, .1, 1., 10.]},
        'fusion_gate': {'minimum_meta_cv_relative_gain': .01, 'minimum_folds_won': 2, 'fixed_validation_improvement': True},
        'audit': {'probe_n': 12, 'seed': 2026, 'rtol': 1e-4, 'atol': 3e-6, 'label_blind_selection': True},
        'automatic_test_export': False, 'course_scope': 'Generic visual-only checkpoint and supplied images/labels only'}
    assert manifest['checkpoint_sha256'] == checked['checkpoint_sha256']
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    cache = local_path('cache_root') / FAMILY / key
    cache.mkdir(parents=True, exist_ok=False)
    write_json(cache / 'manifest.json', manifest)
    relative = Path('19_image_embedding_reference/aimv2_large_224_patchmean') / (
        time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    out, public = local_path('records_root') / relative, REPOSITORY / 'output/experiment_record' / relative
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'data_manifest.json', manifest)
    write_json(public / 'registration.json', {k: v for k, v in manifest.items() if k not in {'imageids', 'pixel_hashes'}} | {'cache_key': key})
    for name in ('aimv2_visual_reference.py', 'aimv2_visual_encoder.py', 'frozen_kernel_search.py'):
        (out / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    write_json(public / 'status.json', {'status': 'preregistered'})
    print(out, flush=True)
    return out


def extract(out):
    out = out.resolve()
    public = REPOSITORY / 'output/experiment_record' / out.relative_to(local_path('records_root'))
    registration = read_json(public / 'registration.json')
    assert read_json(public / 'status.json')['status'] == 'preregistered'
    assert digest(__file__) == registration['source_sha256']
    cache = local_path('cache_root') / FAMILY / registration['cache_key']
    manifest = read_json(cache / 'manifest.json')
    assert manifest == read_json(out / 'data_manifest.json')
    assert digest(local_path('split_path')) == manifest['split_sha256']
    assert digest(local_path('data_root') / 'train.csv') == manifest['train_csv_sha256']
    native = Path(transformers.__file__).parent / 'models/aimv2/modeling_aimv2.py'
    assert digest(native) == manifest['native_implementation_sha256']
    for name, field in [('aimv2_visual_encoder.py', 'encoder_source_sha256'), ('frozen_kernel_search.py', 'kernel_source_sha256')]:
        assert digest(Path(__file__).with_name(name)) == manifest[field]
    def status(state, **extra):
        write_json(public / 'status.json', {'status': state, 'pid': os.getpid(), **extra})
    def image_for(name):
        with Image.open(local_path('data_root') / 'train' / name) as image:
            return ImageOps.exif_transpose(image).convert('RGB')
    started = time.perf_counter()
    try:
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        torch.cuda.reset_peak_memory_stats()
        status('loading_gpu')
        folder = local_path('cache_root') / 'aimv2_weights' / REVISION
        encode, _ = make_encoder(folder, manifest)
        preflight = REPOSITORY / 'output/experiment_record' / PREFLIGHT
        checked = read_json(preflight / 'registration.json')
        expected = np.load(local_path('records_root') / PREFLIGHT / 'features.npy', allow_pickle=False)
        np.testing.assert_allclose(encode([image_for(n) for n in checked['probe_imageids']]), expected, rtol=1e-4, atol=3e-6)
        hashes = {}
        for i, name in enumerate(manifest['imageids']):
            assert pixel_hash(local_path('data_root') / 'train' / name) == manifest['pixel_hashes'][i]
            value = encode([image_for(name)]).astype(np.float32)
            assert value.shape == (1, 1024) and np.isfinite(value).all()
            np.testing.assert_allclose(np.linalg.norm(value, axis=1), np.ones(1), rtol=0, atol=1e-5)
            path = cache / f'{i:05d}_1.npy'
            with path.open('xb') as stream:
                np.save(stream, value)
            hashes[path.name] = digest(path)
            if i % 100 == 0:
                write_json(cache / 'chunk_sha256.json', hashes)
                status('extracting', images=i+1, total=8000)
                print(f'AIMv2 image patchmean: {i+1}/8000', flush=True)
        write_json(cache / 'chunk_sha256.json', hashes)
        assert len(hashes) == 8000
        for name, sha in hashes.items():
            assert digest(cache / name) == sha
        del encode
        torch.cuda.empty_cache()
        # Independently reload the same pinned adapter and re-encode fixed label-blind probes.
        indices = np.random.default_rng(2026).choice(8000, 12, replace=False)
        encode, _ = make_encoder(folder, manifest)
        actual = encode([image_for(manifest['imageids'][i]) for i in indices])
        expected = np.concatenate([np.load(cache / f'{i:05d}_1.npy', allow_pickle=False) for i in indices]).astype(np.float64)
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=3e-6)
        write_json(public / 'encoder_adapter_gpu_audit.json', {'passed': True,
            'all_8000_training_pixel_hashes_verified': True, 'all_8000_cache_chunks_verified': True,
            'gpu_feature_probe_n': 12, 'max_abs_feature_replay_error': float(np.max(abs(actual-expected))),
            'encoder_adapter_source_sha256': manifest['encoder_source_sha256'],
            'feature_manifest_sha256': digest(cache / 'manifest.json'), 'test_images_used': False})
        del encode
        torch.cuda.empty_cache()
        write_json(public / 'cost.json', {'extraction_and_audit_seconds': time.perf_counter()-started,
            'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30, 'feature_cache_key': registration['cache_key']})
        assert digest(Path(__file__).with_name('frozen_kernel_search.py')) == manifest['kernel_source_sha256']
        status('fitting_kernel', feature_extraction_completed=True)
        command = [sys.executable, '-X', 'utf8', str(Path(__file__).with_name('frozen_kernel_search.py')),
            '--cache-family', FAMILY, '--cache-key', registration['cache_key'], '--name', 'aimv2_large_224_patchmean_kernel']
        result = subprocess.run(command, check=False)
        write_json(public / 'kernel_queue.json', {'command': command, 'exit_code': result.returncode})
        if result.returncode:
            raise RuntimeError('AIMv2 grouped kernel search failed')
        status('completed', price_model_evaluated=True)
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
