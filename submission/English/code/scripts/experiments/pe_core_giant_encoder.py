"""Inference adapter for the pinned batch-one PE-Core-G feature contract."""
from pathlib import Path
import numpy as np
import torch
from timm.data import create_transform
from common.config import digest, read_json
from common.local_paths import local_path
from pe_core_giant_preflight import REVISION, build_meta, stream_load


def make_encoder(manifest):
    loader = Path(__file__).with_name('pe_core_giant_preflight.py')
    if manifest['model'] != 'timm/vit_pe_core_gigantic_patch14_448.fb' or manifest['revision'] != REVISION:
        raise ValueError('Only the pinned generic PE-Core-G model is supported')
    if digest(loader) != manifest['loader_source_sha256']:
        raise ValueError('Registered PE-Core-G streaming loader changed')
    if manifest['batch_size'] != 1 or manifest['feature_dim'] != 1280 or manifest['mllm_components'] is not False:
        raise ValueError('PE-Core-G feature contract differs')
    folder = local_path('cache_root') / 'pe_core_giant_weights' / REVISION
    config = read_json(folder / 'config.json')
    if digest(folder / 'config.json') != manifest['config_sha256']:
        raise ValueError('Registered PE-Core-G configuration changed')
    provenance = read_json(folder / 'download.json')
    checkpoint = folder / 'model.safetensors'
    if not provenance['files'][checkpoint.name]['publisher_lfs_verified'] or digest(checkpoint) != manifest['checkpoint_sha256']:
        raise ValueError('Registered PE-Core-G checkpoint changed')
    cfg = config['pretrained_cfg']
    processor = create_transform(input_size=cfg['input_size'], interpolation=cfg['interpolation'],
        mean=cfg['mean'], std=cfg['std'], crop_pct=cfg['crop_pct'], crop_mode=cfg['crop_mode'], is_training=False)
    if repr(processor) != manifest['preprocess']:
        raise ValueError('Registered PE-Core-G preprocessing changed')
    model = stream_load(build_meta(config), checkpoint)

    def encode(images):
        rows = []
        # Preserve batch one, including when callers supply several probe images.
        with torch.inference_mode():
            for image in images:
                pixels = processor(image).unsqueeze(0).to('cuda', torch.float16)
                value = model(pixels).float()
                if value.shape != (1, 1280) or not torch.isfinite(value).all() or value.norm() <= 0:
                    raise ValueError('Invalid PE-Core-G output')
                row = (value / value.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)
                rows.append(row)
        values = np.concatenate(rows).astype(np.float64)
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)

    return encode
