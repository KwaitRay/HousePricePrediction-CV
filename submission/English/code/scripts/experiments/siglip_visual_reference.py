"""Frozen SigLIP 2 image tower only, followed by train-only price regressors."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from safetensors import safe_open
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from transformers import SiglipVisionModel, SiglipVisionConfig, SiglipImageProcessor
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from common.metrics import save_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    provenance = read_json(args.weights_dir / 'download.json')
    if provenance['repo'] != 'google/siglip2-so400m-patch14-384':
        raise ValueError('Only the preregistered official SigLIP 2 checkpoint is supported')
    weight = args.weights_dir / provenance['weight_file']
    if not provenance['files'][weight.name]['publisher_lfs_verified'] or digest(weight) != provenance['files'][weight.name]['sha256']:
        raise ValueError('Publisher checkpoint checksum not verified')
    config = read_json(args.weights_dir / 'config.json')
    if config['model_type'] != 'siglip':
        raise ValueError('This fixed-resolution SigLIP 2 uses the SiglipVisionModel architecture')
    processor = SiglipImageProcessor.from_pretrained(args.weights_dir, local_files_only=True)
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the registered extraction')
    visual = SiglipVisionModel(SiglipVisionConfig(**config['vision_config']))
    # Materialize only image tensors, never the downloaded checkpoint's text tower.
    with safe_open(weight, framework='pt', device='cpu') as tensors:
        keys = [k for k in tensors.keys() if k.startswith('vision_model.')]
        selected = {k: tensors.get_tensor(k) for k in keys}
    if not selected:
        raise ValueError('No vision tensors in official checkpoint')
    dropped = []
    position_key = 'vision_model.embeddings.position_ids'
    if position_key in selected and position_key not in visual.state_dict():
        if not torch.equal(selected[position_key], visual.vision_model.embeddings.position_ids):
            raise ValueError('Unexpected legacy position buffer')
        selected.pop(position_key)
        dropped.append(position_key)
    visual.load_state_dict(selected, strict=True)
    del selected
    visual = visual.half().cuda().eval().requires_grad_(False)
    split = local_path('split_path')
    frame = pd.read_csv(split, dtype={'imageid': str, 'group_id': str})
    train = frame.partition.eq('train').to_numpy()
    valid = frame.partition.eq('validation').to_numpy()
    if len(frame) != 8000 or not frame.imageid.is_unique or train.sum() != 6399 or valid.sum() != 1601 or set(frame.loc[train, 'group_id']) & set(frame.loc[valid, 'group_id']):
        raise ValueError('Unexpected frozen group split')
    manifest = {'model': provenance['repo'], 'revision': provenance['revision'], 'checkpoint_sha256': digest(weight),
                'imageids': frame.imageid.tolist(), 'pixel_hashes': frame.pixel_hash.tolist(), 'split_sha256': digest(split),
                'preprocess': processor.to_dict(), 'feature': 'Official vision pooler, float32 L2 normalization',
                'precision': 'float16 visual forward, float32 normalized cached features', 'feature_dim': 1152,
                'source_sha256': digest(__file__), 'text_modules_used': False, 'test_images_used': False,
                'vision_tensor_n': len(keys), 'legacy_buffers_verified': dropped,
                'model_card': provenance['model_card'], 'paper': 'https://arxiv.org/abs/2502.14786',
                'pretraining_dataset': 'WebLI, as documented in official model card',
                'course_scope': 'General-purpose frozen visual features; no MLLM, no price-task model or additional SoCal data',
                'caveat': 'A complete method comparison; pretraining, model, resolution and precision change together'}
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    cache = local_path('cache_root') / 'siglip_features' / key
    cache.mkdir(parents=True, exist_ok=True)
    write_json(cache / 'manifest.json', manifest)
    run = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
    relative = Path('19_image_embedding_reference/siglip2_so400m') / run
    out, public = local_path('records_root') / relative, REPOSITORY / 'output/experiment_record' / relative
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'data_manifest.json', manifest)
    write_json(public / 'registration.json', {**{k: v for k, v in manifest.items() if k not in ['imageids', 'pixel_hashes']},
                'cache_key': key, 'alphas': [1, 10, 100, 1000, 10000], 'selection': 'Training-only 3-fold GroupKFold'})
    def status(state, **kw):
        write_json(public / 'status.json', {'status': state, 'pid': os.getpid(), **kw})
    start = time.perf_counter()
    try:
        status('extracting', images=0, total=8000)
        chunks = []
        for offset in range(0, 8000, args.batch_size):
            rows = frame.iloc[offset:offset + args.batch_size]
            path = cache / f'{offset:05d}_{len(rows)}.npy'
            if path.exists():
                features = np.load(path, allow_pickle=False)
            else:
                images = []
                for name in rows.imageid:
                    with Image.open(local_path('data_root') / 'train' / name) as source:
                        images.append(ImageOps.exif_transpose(source).convert('RGB'))
                pixels = processor(images=images, return_tensors='pt')['pixel_values'].half().cuda()
                with torch.inference_mode():
                    features = visual(pixel_values=pixels).pooler_output.float()
                    features = (features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
                temporary = path.with_suffix('.tmp')
                with temporary.open('wb') as stream:
                    np.save(stream, features)
                temporary.replace(path)
            if features.shape != (len(rows), 1152) or not np.isfinite(features).all() or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5):
                raise ValueError('Invalid normalized visual features')
            chunks.append(features)
            if offset % 400 == 0:
                status('extracting', images=offset + len(rows), total=8000)
                print(f'SigLIP2 image features {offset + len(rows)}/8000', flush=True)
        x = np.concatenate(chunks).astype(np.float64)
        del visual
        torch.cuda.empty_cache()
        status('fitting', head='ridge')
        search = GridSearchCV(make_pipeline(StandardScaler(), Ridge(solver='cholesky')),
                        {'ridge__alpha': [1, 10, 100, 1000, 10000]}, cv=GroupKFold(3),
                        scoring='neg_mean_squared_error', n_jobs=1, error_score='raise')
        with threadpool_limits(limits=2):
            search.fit(x[train], frame.loc[train, 'price'] / 1000, groups=frame.loc[train, 'group_id'])
        pd.DataFrame(search.cv_results_).to_csv(public / 'training_cv.csv', index=False)
        joblib.dump(search.best_estimator_, out / 'head.joblib')
        metrics = save_predictions(out, frame.loc[valid], search.predict(x[valid]) * 1000, 1300)
        result = {'head': 'ridge', 'alpha': search.best_params_['ridge__alpha'],
                  'training_cv_mse': -search.best_score_ * 1e6, **metrics, 'seconds': time.perf_counter() - start}
        write_json(public / 'results.json', result)
        status('completed')
        print(json.dumps(result), flush=True)
        write_json(public / 'kernel_queue.json', {'status': 'starting', 'cache_key': key})
        job = subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).with_name('frozen_kernel_search.py')),
                            '--name', 'siglip2_so400m_kernel', '--cache-family', 'siglip_features', '--cache-key', key], check=False)
        write_json(public / 'kernel_queue.json', {'status': 'completed' if job.returncode == 0 else 'failed',
                                                'exit_code': job.returncode, 'cache_key': key})
        if job.returncode:
            raise RuntimeError('Kernel comparison failed; Ridge result preserved')
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
