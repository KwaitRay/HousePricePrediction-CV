"""Keep original SigLIP2 view, add whole aspect-preserving and fixed detail views."""
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
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from visual_kernel_submission import cache_array
from visual_kernel_inference import encoder


def main():
    reference = local_path('records_root') / '19_image_embedding_reference/siglip2_so400m/20261013T151014Z_2c237eda/data_manifest.json'
    original_manifest = read_json(reference)
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    key = hashlib.sha256(json.dumps(original_manifest, sort_keys=True).encode()).hexdigest()
    original, _ = cache_array('siglip_features', key, frame)
    if original.shape != (8000, 1152) or digest(local_path('split_path')) != original_manifest['split_sha256']:
        raise ValueError('Original image features differ')
    run = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
    relative = Path('19_image_embedding_reference/siglip2_so400m_multiview') / run
    out, public = local_path('records_root') / relative, REPOSITORY / 'output/experiment_record' / relative
    out.mkdir(parents=True, exist_ok=False)
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    (out / 'encoder_source.py').write_bytes(Path(__file__).with_name('visual_kernel_inference.py').read_bytes())
    manifest = {'model': original_manifest['model'], 'revision': original_manifest['revision'],
        'checkpoint_sha256': original_manifest['checkpoint_sha256'], 'imageids': frame.imageid.tolist(),
        'split_sha256': digest(local_path('split_path')), 'original_manifest_sha256': digest(reference),
        'views': ['Original official square resize', 'Whole EXIF image letterboxed to 384 square, gray RGB128 border',
                  'Fixed center 60% width and height, official square preprocessing'],
        'view_size': 384, 'feature_dim': 1152, 'combined_feature_dim': 3456, 'batch_size': 4,
        'preprocess': original_manifest['preprocess'], 'precision': original_manifest['precision'],
        'extra_feature_storage': 'L2-normalized image embedding rounded to float32',
        'source_sha256': digest(__file__), 'encoder_source_sha256': digest(Path(__file__).with_name('visual_kernel_inference.py')),
        'test_images_used': False, 'text_modules_used': False,
        'selection': 'Training-only 3-fold GroupKFold kernel parameter selection',
        'course_scope': 'General-purpose frozen image tower; no MLLM, price-task checkpoint or extra SoCal data',
        'caveat': 'Original view remains present; upsampling cannot recover missing native image detail'}
    cache_key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    cache = local_path('cache_root') / 'siglip_multiview' / cache_key
    cache.mkdir(parents=True, exist_ok=True)
    write_json(cache / 'manifest.json', manifest)
    write_json(out / 'data_manifest.json', manifest)
    write_json(public / 'registration.json', {**{k: v for k, v in manifest.items() if k != 'imageids'}, 'cache_key': cache_key})
    def status(state, **extra):
        write_json(public / 'status.json', {'status': state, 'pid': os.getpid(), **extra})
    started = time.perf_counter()
    try:
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        encode = encoder('siglip_features', original_manifest)
        status('extracting', images=0, total=8000)
        for offset in range(0, 8000, 4):
            rows = frame.iloc[offset:offset + 4]
            path = cache / f'{offset:05d}_{len(rows)}.npy'
            if path.exists():
                raise FileExistsError('New runs never silently reuse an unverified partial cache')
            whole, center = [], []
            for name in rows.imageid:
                with Image.open(local_path('data_root') / 'train' / name) as source:
                    im = ImageOps.exif_transpose(source).convert('RGB')
                whole.append(ImageOps.pad(im, (384, 384), method=Image.Resampling.BICUBIC, color=(128, 128, 128)))
                center.append(im.crop((round(im.width * .2), round(im.height * .2), round(im.width * .8), round(im.height * .8))))
            values = np.stack([encode(whole), encode(center)], axis=1).astype(np.float32)
            if values.shape != (len(rows), 2, 1152) or not np.isfinite(values).all():
                raise ValueError('Invalid image view features')
            with path.open('xb') as stream:
                np.save(stream, values)
            if offset % 400 == 0:
                status('extracting', images=offset + len(rows), total=8000)
                print(f'SigLIP2 whole/detail views {offset + len(rows)}/8000', flush=True)
        del encode, original
        torch.cuda.empty_cache()
        write_json(public / 'cost.json', {'extraction_seconds': time.perf_counter() - started, 'feature_cache_key': cache_key})
        status('fitting_kernel')
        name = 'siglip2_so400m_multiview_kernel'
        command = [sys.executable, '-X', 'utf8', str(Path(__file__).with_name('frozen_kernel_search.py')),
            '--name', name, '--cache-family', 'siglip_multiview', '--cache-key', cache_key, '--original-manifest', str(reference)]
        write_json(public / 'kernel_queue.json', {'status': 'running', 'command': command})
        result = subprocess.run(command, check=False)
        write_json(public / 'kernel_queue.json', {'status': 'completed' if result.returncode == 0 else 'failed', 'exit_code': result.returncode})
        if result.returncode:
            raise RuntimeError('Multiview kernel search failed')
        status('completed')
        subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).with_name('improvement_summary.py'))], check=True)
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
