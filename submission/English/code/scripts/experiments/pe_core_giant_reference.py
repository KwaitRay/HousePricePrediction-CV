"""Official frozen PE-Core-G image output followed by train-only grouped RBF CV."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import timm
from timm.data import create_transform
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from pe_core_giant_preflight import REVISION, build_meta, stream_load


def main():
    preflight = REPOSITORY / 'output/experiment_record/19_image_embedding_reference/pe_core_giant_preflight/20261014T064838Z_7e314d82'
    assert read_json(preflight / 'results.json')['passed']
    assert digest(Path(__file__).with_name('pe_core_giant_preflight.py')) == read_json(preflight / 'registration.json')['source_sha256']
    folder = local_path('cache_root') / 'pe_core_giant_weights' / REVISION
    provenance, config = read_json(folder / 'download.json'), read_json(folder / 'config.json')
    assert provenance['revision'] == REVISION
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    assert len(frame) == 8000 and frame.partition.eq('train').sum() == 6399
    assert not set(frame.loc[frame.partition.eq('train'), 'group_id']) & set(frame.loc[frame.partition.eq('validation'), 'group_id'])
    cfg = config['pretrained_cfg']
    processor = create_transform(input_size=cfg['input_size'], interpolation=cfg['interpolation'], mean=cfg['mean'],
        std=cfg['std'], crop_pct=cfg['crop_pct'], crop_mode=cfg['crop_mode'], is_training=False)
    kernel_source = Path(__file__).with_name('frozen_kernel_search.py')
    manifest = {'model': provenance['repo'], 'revision': REVISION, 'checkpoint_sha256': provenance['files']['model.safetensors']['sha256'],
        'imageids': frame.imageid.tolist(), 'pixel_hashes': frame.pixel_hash.tolist(), 'split_sha256': digest(local_path('split_path')),
        'feature_dim': 1280, 'batch_size': 1, 'preprocess': repr(processor), 'config_sha256': digest(folder / 'config.json'),
        'feature': 'Official learned PE-Core-G attention pooler and 1280D projection, FP32 L2 normalization',
        'precision': 'FP16 forward and parameters, official FP32 RoPE cast to FP16, FP32 normalized stored features',
        'source_sha256': digest(__file__), 'loader_source_sha256': digest(Path(__file__).with_name('pe_core_giant_preflight.py')),
        'kernel_search_source_sha256': digest(kernel_source), 'timm_version': timm.__version__,
        'mllm_components': False, 'text_modules_used': False, 'test_images_used': False,
        'course_scope': 'Generic pretrained visual features and provided labels only',
        'official_model_card': provenance['model_card']}
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    cache = local_path('cache_root') / 'pe_core_giant_features' / key
    cache.mkdir(parents=True, exist_ok=False)
    write_json(cache / 'manifest.json', manifest)
    relative = Path('19_image_embedding_reference/pe_core_g14_448') / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    out, public = local_path('records_root') / relative, REPOSITORY / 'output/experiment_record' / relative
    out.mkdir(parents=True, exist_ok=False)
    registration = {k: v for k, v in manifest.items() if k not in ('imageids', 'pixel_hashes')}
    registration.update(cache_family='pe_core_giant_features', cache_key=key,
        regression_grid={'gammas': [1., 4., 16.], 'alphas': [.01, .1, 1., 10.], 'training_group_folds': 3},
        selection='Original-price unweighted training grouped OOF MSE; fixed validation once after selection')
    write_json(public / 'registration.json', registration)
    write_json(out / 'data_manifest.json', manifest)
    for filename in ('pe_core_giant_reference.py', 'pe_core_giant_preflight.py', 'frozen_kernel_search.py'):
        (out / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    def status(state, **extra):
        write_json(public / 'status.json', {'status': state, 'pid': os.getpid(), **extra})
    print(public, flush=True)
    started = time.perf_counter()
    try:
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        checkpoint = folder / 'model.safetensors'
        assert digest(checkpoint) == manifest['checkpoint_sha256']
        status('loading_gpu')
        model = stream_load(build_meta(config), checkpoint)
        checksums = {}
        with torch.inference_mode():
            for offset, name in enumerate(frame.imageid):
                with Image.open(local_path('data_root') / 'train' / name) as source:
                    pixels = processor(ImageOps.exif_transpose(source).convert('RGB')).unsqueeze(0).to('cuda', torch.float16)
                value = model(pixels).float()
                assert value.shape == (1, 1280) and torch.isfinite(value).all()
                assert value.norm() > 0
                vector = (value / value.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)
                path = cache / f'{offset:05d}_1.npy'
                with path.open('xb') as stream:
                    np.save(stream, vector)
                checksums[path.name] = digest(path)
                if offset % 100 == 0:
                    write_json(cache / 'chunk_sha256.json', checksums)
                    status('extracting', images=offset + 1, total=8000)
                    print(f'PE-Core-G: {offset + 1}/8000', flush=True)
        write_json(cache / 'chunk_sha256.json', checksums)
        del model
        torch.cuda.empty_cache()
        write_json(public / 'cost.json', {'extraction_seconds': time.perf_counter() - started,
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30, 'cache_key': key})
        assert digest(kernel_source) == manifest['kernel_search_source_sha256']
        status('fitting_kernel')
        command = [sys.executable, '-X', 'utf8', str(kernel_source), '--name', 'pe_core_g14_448_kernel',
            '--cache-family', 'pe_core_giant_features', '--cache-key', key]
        result = subprocess.run(command, check=False)
        write_json(public / 'kernel_queue.json', {'command': command, 'exit_code': result.returncode})
        if result.returncode:
            raise RuntimeError('PE-Core-G kernel search failed')
        status('completed')
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
