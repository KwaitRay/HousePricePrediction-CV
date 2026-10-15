"""Pinned EVA02 CLIP image tower with its learned 768D projection preserved."""
from pathlib import Path
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
import torch
import timm
from timm.data import create_transform, resolve_model_data_config
from safetensors.torch import load_file
from torch.nn.attention import sdpa_kernel, SDPBackend
from common.config import digest, read_json

REPO = 'timm/eva02_large_patch14_clip_336.merged2b'
REVISION = 'a925908b5ecdd9b1eb34c8839aaf86cc6464b941'
NUMERICAL_POLICY = 'FP32, TF32 disabled, SDPA MATH only, deterministic Torch/CUDA, cuDNN benchmark off, CUBLAS workspace :4096:8; two fixed RGB128 336-square warmup forwards per load'
PRECISION = 'FP32 visual model/forward/math attention and unit normalization/storage; F64 renormalization'


def make_encoder(folder, manifest=None):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    folder = Path(folder)
    provenance = read_json(folder / 'download.json')
    checkpoint = folder / 'model.safetensors'
    if provenance['repo'] != REPO or provenance['revision'] != REVISION:
        raise ValueError('Only the pinned EVA02 CLIP publisher revision is supported')
    weight = provenance['files'][checkpoint.name]
    if not weight['publisher_lfs_verified'] or digest(checkpoint) != weight['sha256']:
        raise ValueError('EVA CLIP publisher checkpoint checksum differs')
    config = read_json(folder / 'config.json')
    if config['architecture'] != 'eva02_large_patch14_clip_336' or config['num_classes'] != 768:
        raise ValueError('Expected the fixed EVA02 CLIP tower and learned 768D projection')
    # Real CPU construction retains official positional/RoPE buffers before FP16 conversion.
    visual = timm.create_model(config['architecture'], pretrained=False,
        num_classes=config['num_classes'], img_size=336, global_pool=config['global_pool'],
        pretrained_cfg=config['pretrained_cfg'])
    processor = create_transform(**resolve_model_data_config(visual), is_training=False)
    if manifest is not None:
        if (manifest['model'] != REPO or manifest['revision'] != REVISION
                or manifest['feature_dim'] != 768 or manifest['batch_size'] != 1
                or manifest['mllm_components'] is not False):
            raise ValueError('Registered EVA CLIP feature contract differs')
        if (manifest['checkpoint_sha256'] != digest(checkpoint)
                or manifest['config_sha256'] != digest(folder / 'config.json')
                or manifest['preprocess'] != repr(processor)
                or manifest['encoder_source_sha256'] != digest(__file__)):
            raise ValueError('Registered EVA CLIP weights, processor or adapter changed')
    visual.load_state_dict(load_file(checkpoint, device='cpu'), strict=True)
    visual = visual.cuda().eval().requires_grad_(False)
    from PIL import Image
    warmup = processor(Image.new('RGB', (336, 336), (128, 128, 128))).unsqueeze(0).to('cuda', torch.float32)
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for _ in range(2):
            visual(warmup)
    del warmup

    def encode(images):
        rows = []
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for image in images:
                output = visual(processor(image).unsqueeze(0).to('cuda', torch.float32)).float()
                if output.shape != (1, 768) or not torch.isfinite(output).all() or output.norm() <= 0:
                    raise ValueError('Invalid projected EVA CLIP image vector')
                value = (output / output.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)
                rows.append(value)
        values = np.concatenate(rows).astype(np.float64)
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)

    return encode, processor
