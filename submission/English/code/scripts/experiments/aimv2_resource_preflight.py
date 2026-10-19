"""Resource/numerical preflight, with no price-based architecture selection."""
import argparse
import os
import time
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from aimv2_visual_encoder import REPO, REVISION, make_encoder


def main(folder):
    provenance = read_json(folder / 'download.json')
    assert provenance['revision'] == REVISION and provenance['repo'] == REPO
    assert provenance['files']['model.safetensors']['publisher_lfs_verified']
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str})
    names = frame.loc[frame.partition.eq('train'), 'imageid'].iloc[:3].tolist()
    relative = Path('19_image_embedding_reference/aimv2_large_224_preflight') / (
        time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    out, public = local_path('records_root') / relative, REPOSITORY / 'output/experiment_record' / relative
    out.mkdir(parents=True, exist_ok=False)
    registration = {'model': REPO, 'revision': REVISION, 'checkpoint_sha256': digest(folder / 'model.safetensors'),
        'config_sha256': digest(folder / 'config.json'), 'batch_size': 1, 'image_size': 224,
        'feature_dim': 1024, 'feature': 'Unit-normalized mean of all 256 post-norm image patch tokens',
        'precision': 'FP16 visual model and forward, FP32 patch mean and normalization, F64 renormalization',
        'source_sha256': digest(__file__), 'encoder_source_sha256': digest(Path(__file__).with_name('aimv2_visual_encoder.py')),
        'probe_imageids': names, 'split_sha256': digest(local_path('split_path')),
        'test_images_used': False, 'price_labels_used_for_selection': False, 'mllm_components': False,
        'remote_code_executed': False, 'automatic_feature_extraction': False, 'license': provenance['license']}
    write_json(public / 'registration.json', registration)
    for filename in ('aimv2_resource_preflight.py', 'aimv2_visual_encoder.py'):
        (out / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    write_json(public / 'status.json', {'status': 'loading_gpu', 'pid': os.getpid()})
    try:
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        encode, processor = make_encoder(folder)
        images = []
        for name in names:
            with Image.open(local_path('data_root') / 'train' / name) as image:
                images.append(ImageOps.exif_transpose(image).convert('RGB'))
        first = encode(images)
        second = encode(images)
        np.testing.assert_allclose(first, second, rtol=1e-4, atol=3e-6)
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), np.ones(3), rtol=0, atol=1e-8)
        np.save(out / 'features.npy', first, allow_pickle=False)
        write_json(public / 'results.json', {'passed': True, 'strict_checkpoint_loading': True,
            'feature_shape': list(first.shape), 'repeat_max_abs_error': float(np.max(abs(first-second))),
            'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
            'seconds': time.perf_counter()-started, 'preprocess': processor.to_dict(),
            'price_model_evaluated': False, 'test_images_used': False})
        write_json(public / 'status.json', {'status': 'completed', 'pid': os.getpid()})
        print('AIMV2_PREFLIGHT_PASSED', public, flush=True)
    except BaseException as exc:
        write_json(public / 'status.json', {'status': 'failed', 'pid': os.getpid(), 'error': repr(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-dir', type=Path, required=True)
    main(parser.parse_args().weights_dir)
