"""Frozen official NaFlex image-only features with an explicitly fixed patch budget."""
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
from transformers import Siglip2ImageProcessorFast
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from visual_kernel_inference import encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-dir', type=Path, required=True)
    parser.add_argument('--max-num-patches', type=int, choices=[256, 576, 1024], default=576)
    args = parser.parse_args()
    provenance = read_json(args.weights_dir / 'download.json')
    checkpoint = args.weights_dir / provenance['weight_file']
    if provenance['repo'] != 'google/siglip2-so400m-patch16-naflex' or not provenance['files'][checkpoint.name]['publisher_lfs_verified'] or digest(checkpoint) != provenance['files'][checkpoint.name]['sha256']:
        raise ValueError('Official NaFlex checkpoint not verified')
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    train, valid = frame.partition.eq('train').to_numpy(), frame.partition.eq('validation').to_numpy()
    if len(frame) != 8000 or train.sum() != 6399 or valid.sum() != 1601 or set(frame.loc[train, 'group_id']) & set(frame.loc[valid, 'group_id']):
        raise ValueError('Unexpected frozen split')
    processor = Siglip2ImageProcessorFast.from_pretrained(args.weights_dir, local_files_only=True, max_num_patches=args.max_num_patches)
    manifest = {'model': provenance['repo'], 'revision': provenance['revision'], 'checkpoint_sha256': digest(checkpoint),
        'imageids': frame.imageid.tolist(), 'pixel_hashes': frame.pixel_hash.tolist(), 'split_sha256': digest(local_path('split_path')),
        'preprocess': processor.to_dict(), 'max_num_patches': args.max_num_patches, 'feature_dim': 1152,
        'feature': 'Official NaFlex vision attention pooler, float32 L2 normalization',
        'precision': 'float16 vision forward, float32 normalized feature storage', 'batch_size': 4,
        'source_sha256': digest(__file__), 'encoder_source_sha256': digest(Path(__file__).with_name('visual_kernel_inference.py')),
        'test_images_used': False, 'text_modules_used': False, 'mllm_components': False,
        'course_scope': 'Generic frozen image tower, no price-task model or additional SoCal data',
        'official_model_card': provenance['model_card'],
        'caveat': 'Full method comparison: checkpoint, patch size and image preprocessing differ from fixed-resolution SigLIP2; more patches cannot recover absent native detail'}
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    cache = local_path('cache_root') / 'siglip2_naflex_features' / key
    cache.mkdir(parents=True, exist_ok=True)
    write_json(cache / 'manifest.json', manifest)
    run = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
    name = f'siglip2_naflex_{args.max_num_patches}'
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
        encode = encoder('siglip2_naflex_features', manifest)
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
            if features.shape != (len(rows), 1152) or not np.isfinite(features).all() or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5):
                raise ValueError('Invalid image features')
            with path.open('xb') as stream:
                np.save(stream, features)
            if offset % 400 == 0:
                status('extracting', images=offset + len(rows), total=8000)
                print(f'NaFlex {args.max_num_patches} patches: {offset + len(rows)}/8000', flush=True)
        del encode
        torch.cuda.empty_cache()
        write_json(public / 'cost.json', {'extraction_seconds': time.perf_counter() - started, 'feature_cache_key': key})
        status('fitting_kernel')
        command = [sys.executable, '-X', 'utf8', str(Path(__file__).with_name('frozen_kernel_search.py')),
            '--name', f'siglip2_naflex_{args.max_num_patches}_kernel', '--cache-family', 'siglip2_naflex_features', '--cache-key', key]
        result = subprocess.run(command, check=False)
        write_json(public / 'kernel_queue.json', {'status': 'completed' if result.returncode == 0 else 'failed', 'command': command, 'exit_code': result.returncode})
        if result.returncode:
            raise RuntimeError('NaFlex kernel search failed')
        status('completed')
        subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).with_name('improvement_summary.py'))], check=True)
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
