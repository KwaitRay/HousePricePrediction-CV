"""Pinned image-only Giant preflight and frozen-feature price reference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from transformers import SiglipImageProcessor
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from siglip_giant_encoder import REPO, REVISION, make_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=['preflight', 'experiment'], default='experiment')
    args = parser.parse_args()
    provenance = read_json(args.weights_dir / 'download.json')
    if provenance['repo'] != REPO or provenance['revision'] != REVISION or not provenance.get('sharded'):
        raise ValueError('Only the pinned official Giant checkpoint is supported')
    for shard in provenance['vision_weight_shards']:
        if not provenance['files'].get(shard, {}).get('publisher_lfs_verified'):
            raise ValueError('Official Giant weight download is not complete')
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    train, valid = frame.partition.eq('train'), frame.partition.eq('validation')
    if len(frame) != 8000 or train.sum() != 6399 or valid.sum() != 1601 or set(frame.loc[train, 'group_id']) & set(frame.loc[valid, 'group_id']):
        raise ValueError('Unexpected frozen split')
    processor = SiglipImageProcessor.from_pretrained(args.weights_dir, local_files_only=True)
    manifest = {'model': REPO, 'revision': REVISION, 'preprocess': processor.to_dict(),
        'checkpoint_index_sha256': digest(args.weights_dir / 'model.safetensors.index.json'),
        'checkpoint_shard_sha256': {name: provenance['files'][name]['sha256'] for name in provenance['vision_weight_shards']},
        'imageids': frame.imageid.tolist(), 'pixel_hashes': frame.pixel_hash.tolist(),
        'split_sha256': digest(local_path('split_path')), 'feature_dim': 1536, 'batch_size': 1,
        'precision': 'float16 image-only CUDA forward, float32 normalized feature storage',
        'feature': 'Official Giant vision pooler output, unit normalized',
        'source_sha256': digest(__file__), 'loader_source_sha256': digest(Path(__file__).with_name('siglip_giant_encoder.py')),
        'encoder_source_sha256': digest(Path(__file__).with_name('visual_kernel_inference.py')),
        'host_loading': 'Meta architecture, direct vision-tensor-to-fp16-CUDA streaming, no full CPU tower',
        'text_modules_used': False, 'mllm_components': False, 'test_images_used': False,
        'pretraining_dataset': 'WebLI generic vision-language pretraining', 'license': provenance['license'],
        'course_scope': 'Generic pretrained image encoder and supplied labels only; no MLLM, price-task checkpoint or extra SoCal data'}
    relative = Path('19_image_embedding_reference/siglip2_giant_384') / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    out, public = local_path('records_root') / relative, REPOSITORY / 'output/experiment_record' / relative
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'data_manifest.json', manifest)
    write_json(public / 'registration.json', {key: value for key, value in manifest.items() if key not in ('imageids', 'pixel_hashes')})
    for name in ('siglip_giant_reference.py', 'siglip_giant_encoder.py', 'visual_kernel_inference.py'):
        (out / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    def status(state, **extra):
        write_json(public / 'status.json', {'status': state, 'pid': os.getpid(), **extra})
    def images(names):
        values = []
        for name in names:
            with Image.open(local_path('data_root') / 'train' / name) as image:
                values.append(ImageOps.exif_transpose(image).convert('RGB'))
        return values
    started = time.perf_counter()
    try:
        status('loading_vision', phase=args.phase)
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        torch.cuda.reset_peak_memory_stats()
        encode = make_encoder(manifest)
        probe_ids = frame.loc[train, 'imageid'].iloc[:4].tolist()
        probe_images = images(probe_ids)
        probes = np.concatenate([encode([image]) for image in probe_images])
        if probes.shape != (4, 1536) or not np.isfinite(probes).all():
            raise ValueError('Giant feature preflight failed')
        np.testing.assert_allclose(np.linalg.norm(probes, axis=1), np.ones(4), rtol=0, atol=1e-8)
        np.testing.assert_allclose(encode([probe_images[0]])[0], probes[0], rtol=1e-6, atol=1e-6)
        write_json(public / 'gpu_preflight.json', {'passed': True, 'batch_size': 1, 'probe_imageids': probe_ids,
            'finite_unit_features': True, 'repeated_feature_replay_passed': True,
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20, 'text_modules_used': False})
        if args.phase == 'preflight':
            status('completed', phase='preflight', price_model_evaluated=False)
            return
        key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        cache = local_path('cache_root') / 'siglip2_giant_features' / key
        cache.mkdir(parents=True, exist_ok=True)
        write_json(cache / 'manifest.json', manifest)
        for offset in range(8000):
            path = cache / f'{offset:05d}_1.npy'
            if path.exists():
                raise FileExistsError('No silent reuse of a partial Giant feature cache')
            features = encode(images(frame.imageid.iloc[offset:offset + 1])).astype(np.float32)
            if features.shape != (1, 1536) or not np.isfinite(features).all() or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5):
                raise ValueError('Invalid Giant image features')
            with path.open('xb') as stream:
                np.save(stream, features)
            if offset % 200 == 0:
                status('extracting', images=offset + 1, total=8000)
                print(f'Giant image-only features {offset + 1}/8000', flush=True)
        del encode
        torch.cuda.empty_cache()
        write_json(public / 'cost.json', {'extraction_seconds': time.perf_counter() - started, 'feature_cache_key': key})
        status('fitting_kernel')
        command = [sys.executable, '-X', 'utf8', str(Path(__file__).with_name('frozen_kernel_search.py')),
            '--name', 'siglip2_giant_384_kernel', '--cache-family', 'siglip2_giant_features', '--cache-key', key]
        result = subprocess.run(command, check=False)
        write_json(public / 'kernel_queue.json', {'status': 'completed' if result.returncode == 0 else 'failed', 'command': command, 'exit_code': result.returncode})
        if result.returncode:
            raise RuntimeError('Giant kernel reference failed')
        status('completed')
        subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).with_name('improvement_summary.py'))], check=True)
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
