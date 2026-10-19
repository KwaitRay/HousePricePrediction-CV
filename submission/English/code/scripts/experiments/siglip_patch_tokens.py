"""Cache frozen full-image patch tokens for a price-supervised attention experiment."""
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
from safetensors import safe_open
from transformers import SiglipVisionModel, SiglipVisionConfig, SiglipImageProcessor
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from visual_kernel_submission import cache_array


def main():
    source = local_path('records_root') / '19_image_embedding_reference/siglip2_so400m/20261013T151014Z_2c237eda/data_manifest.json'
    original = read_json(source)
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    if original['imageids'] != frame.imageid.tolist() or original['split_sha256'] != digest(local_path('split_path')):
        raise ValueError('Frozen split changed')
    manifest = {'method': 'Frozen post-layernorm patch tokens, original full-image official SigLIP2 resize',
        'model': original['model'], 'revision': original['revision'], 'checkpoint_sha256': original['checkpoint_sha256'],
        'original_manifest_sha256': digest(source), 'preprocess': original['preprocess'], 'split_sha256': digest(local_path('split_path')),
        'imageids': frame.imageid.tolist(), 'shape': [8000, 729, 1152], 'dtype': 'float16', 'batch_size': 4,
        'source_sha256': digest(__file__), 'text_modules_used': False, 'mllm_components': False, 'test_images_used': False,
        'course_scope': 'Generic frozen image tower and provided images only; supervised pooling is trained separately on training labels'}
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    cache = local_path('cache_root') / 'siglip_patch_tokens' / key
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / 'patch_tokens.npy'
    run = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
    relative = Path('19_image_embedding_reference/siglip2_patch_tokens') / run
    public = REPOSITORY / 'output/experiment_record' / relative
    write_json(public / 'registration.json', {**{k: v for k, v in manifest.items() if k != 'imageids'}, 'cache_key': key})
    def status(state, **extra):
        write_json(public / 'status.json', {'status': state, 'pid': os.getpid(), **extra})
    try:
        status('extracting', images=0, total=8000)
        if path.exists():
            checks = read_json(cache / 'checks.json')
            if read_json(cache / 'manifest.json') != manifest or digest(path) != checks['array_sha256']:
                raise ValueError('Patch-token cache is incomplete or modified')
        else:
            weights = local_path('cache_root') / 'siglip2_weights' / original['revision']
            provenance = read_json(weights / 'download.json')
            checkpoint = weights / provenance['weight_file']
            if digest(checkpoint) != original['checkpoint_sha256']:
                raise ValueError('Official checkpoint changed')
            processor = SiglipImageProcessor.from_pretrained(weights, local_files_only=True)
            if processor.to_dict() != original['preprocess']:
                raise ValueError('Official image processor changed')
            config = read_json(weights / 'config.json')
            visual = SiglipVisionModel(SiglipVisionConfig(**config['vision_config']))
            with safe_open(checkpoint, framework='pt', device='cpu') as tensors:
                selected = {k: tensors.get_tensor(k) for k in tensors.keys() if k.startswith('vision_model.')}
            visual.load_state_dict(selected, strict=True)
            del selected
            torch.set_num_threads(2)
            torch.manual_seed(2026)
            visual = visual.half().cuda().eval().requires_grad_(False)
            original_key = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
            expected, _ = cache_array('siglip_features', original_key, frame)
            write_json(cache / 'manifest.json', manifest)
            storage = np.lib.format.open_memmap(path, mode='w+', dtype=np.float16, shape=tuple(manifest['shape']))
            started = time.perf_counter()
            for offset in range(0, 8000, 4):
                images = []
                for name in frame.imageid.iloc[offset:offset + 4]:
                    with Image.open(local_path('data_root') / 'train' / name) as source_image:
                        images.append(ImageOps.exif_transpose(source_image).convert('RGB'))
                pixels = processor(images=images, return_tensors='pt')['pixel_values'].half().cuda()
                with torch.inference_mode():
                    output = visual(pixel_values=pixels)
                tokens = output.last_hidden_state.cpu().numpy()
                if tokens.shape != (len(images), 729, 1152) or tokens.dtype != np.float16 or not np.isfinite(tokens).all():
                    raise ValueError('Invalid patch tokens')
                if offset < 8:
                    pooled = output.pooler_output.float()
                    pooled = (pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
                    np.testing.assert_allclose(pooled, expected[offset:offset + len(images)], rtol=1e-4, atol=3e-6)
                storage[offset:offset + len(images)] = tokens
                if offset % 400 == 0:
                    storage.flush()
                    status('extracting', images=offset + len(images), total=8000)
                    print(f'SigLIP2 full-image patch tokens {offset + len(images)}/8000', flush=True)
            storage.flush()
            del storage, visual, output, expected
            torch.cuda.empty_cache()
            write_json(cache / 'checks.json', {'array_sha256': digest(path), 'shape': manifest['shape'], 'dtype': 'float16',
                'all_chunks_finite': True, 'pooled_feature_replay_passed': True, 'seconds': time.perf_counter() - started})
        status('training_attention')
        command = [sys.executable, '-X', 'utf8', str(Path(__file__).with_name('siglip_price_attention.py')), '--cache-key', key]
        write_json(public / 'training_queue.json', {'status': 'running', 'command': command})
        result = subprocess.run(command, check=False)
        write_json(public / 'training_queue.json', {'status': 'completed' if result.returncode == 0 else 'failed', 'exit_code': result.returncode})
        if result.returncode:
            raise RuntimeError('Price-attention experiment failed')
        status('completed')
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
