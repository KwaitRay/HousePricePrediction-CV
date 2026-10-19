"""Frozen PE-Core image tower including its learned CLIP output projection."""
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
import timm
from timm.data import create_transform
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from visual_kernel_inference import encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-dir', type=Path, required=True)
    args = parser.parse_args()
    provenance = read_json(args.weights_dir / 'download.json')
    checkpoint = args.weights_dir / provenance['weight_file']
    if provenance['repo'] != 'timm/vit_pe_core_large_patch14_336.fb' or not provenance['files'][checkpoint.name]['publisher_lfs_verified'] or digest(checkpoint) != provenance['files'][checkpoint.name]['sha256']:
        raise ValueError('PE-Core image-only checkpoint not verified')
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    train, valid = frame.partition.eq('train').to_numpy(), frame.partition.eq('validation').to_numpy()
    if len(frame) != 8000 or train.sum() != 6399 or valid.sum() != 1601 or set(frame.loc[train, 'group_id']) & set(frame.loc[valid, 'group_id']):
        raise ValueError('Unexpected frozen split')
    config = read_json(args.weights_dir / 'config.json')
    data_config = config['pretrained_cfg']
    processor = create_transform(input_size=data_config['input_size'], interpolation=data_config['interpolation'],
        mean=data_config['mean'], std=data_config['std'], crop_pct=data_config['crop_pct'], crop_mode=data_config['crop_mode'], is_training=False)
    manifest = {'model': provenance['repo'], 'revision': provenance['revision'], 'checkpoint_sha256': digest(checkpoint),
        'imageids': frame.imageid.tolist(), 'pixel_hashes': frame.pixel_hash.tolist(), 'split_sha256': digest(local_path('split_path')),
        'preprocess': repr(processor), 'feature_dim': 1024, 'timm_version': timm.__version__,
        'feature': 'PE-Core attention pooler and learned 1024D projection, float32 L2 normalization',
        'precision': 'float16 vision forward, float32 normalized feature storage', 'batch_size': 4,
        'source_sha256': digest(__file__), 'encoder_source_sha256': digest(Path(__file__).with_name('visual_kernel_inference.py')),
        'test_images_used': False, 'text_modules_used': False, 'mllm_components': False,
        'course_scope': 'Generic frozen image tower, no price-task model or additional SoCal data',
        'official_model_card': provenance['model_card'],
        'caveat': 'Full method comparison; preserve projection head, not an ImageNet classifier; checkpoint and preprocessing differ from SigLIP2'}
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    cache = local_path('cache_root') / 'pe_core_features' / key
    cache.mkdir(parents=True, exist_ok=True)
    write_json(cache / 'manifest.json', manifest)
    run = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
    name = 'pe_core_l14_336'
    relative = Path('19_image_embedding_reference') / name / run
    out, public = local_path('records_root') / relative, REPOSITORY / 'output/experiment_record' / relative
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'data_manifest.json', manifest)
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    (out / 'encoder_source.py').write_bytes(Path(__file__).with_name('visual_kernel_inference.py').read_bytes())
    write_json(public / 'registration.json', {**{k: v for k, v in manifest.items() if k not in ['imageids', 'pixel_hashes']}, 'cache_key': key})
    def status(state, **extra):
        write_json(public / 'status.json', {'status': state, 'pid': os.getpid(), **extra})
    started = time.perf_counter()
    try:
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        status('loading_vision')
        encode = encoder('pe_core_features', manifest)
        for offset in range(0, 8000, 4):
            rows = frame.iloc[offset:offset + 4]
            path = cache / f'{offset:05d}_{len(rows)}.npy'
            if path.exists():
                raise FileExistsError('No reuse of unverified partial extraction')
            images = []
            for name_id in rows.imageid:
                with Image.open(local_path('data_root') / 'train' / name_id) as source:
                    images.append(ImageOps.exif_transpose(source).convert('RGB'))
            features = encode(images).astype(np.float32)
            if features.shape != (len(rows), 1024) or not np.isfinite(features).all() or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5):
                raise ValueError('Invalid image features')
            with path.open('xb') as stream:
                np.save(stream, features)
            if offset % 400 == 0:
                status('extracting', images=offset + len(rows), total=8000)
                print(f'PE-Core projected image features {offset + len(rows)}/8000', flush=True)
        del encode
        torch.cuda.empty_cache()
        write_json(public / 'cost.json', {'extraction_seconds': time.perf_counter() - started, 'feature_cache_key': key})
        status('fitting_kernel')
        command = [sys.executable, '-X', 'utf8', str(Path(__file__).with_name('frozen_kernel_search.py')),
            '--name', 'pe_core_l14_336_kernel', '--cache-family', 'pe_core_features', '--cache-key', key]
        result = subprocess.run(command, check=False)
        write_json(public / 'kernel_queue.json', {'status': 'completed' if result.returncode == 0 else 'failed', 'command': command, 'exit_code': result.returncode})
        if result.returncode:
            raise RuntimeError('PE-Core kernel search failed')
        status('completed')
        subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).with_name('improvement_summary.py'))], check=True)
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
