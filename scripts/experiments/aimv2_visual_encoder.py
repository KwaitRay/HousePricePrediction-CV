"""Pinned Apple AIMv2 image-only inference using installed native Transformers."""
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import Aimv2VisionModel, Aimv2VisionConfig, CLIPImageProcessorFast
from common.config import digest, read_json

REPO = 'apple/aimv2-large-patch14-224'
REVISION = 'fcb5093a9c5b3efb7db5acee213849967fd18210'


def make_encoder(folder, manifest=None):
    folder = Path(folder)
    provenance = read_json(folder / 'download.json')
    checkpoint = folder / 'model.safetensors'
    if provenance['repo'] != REPO or provenance['revision'] != REVISION:
        raise ValueError('Only the pinned official AIMv2 revision is supported')
    weight = provenance['files'][checkpoint.name]
    if not weight['publisher_lfs_verified'] or digest(checkpoint) != weight['sha256']:
        raise ValueError('Official AIMv2 checkpoint checksum differs')
    config_json = read_json(folder / 'config.json')
    if config_json['model_type'] != 'aimv2_vision_model' or config_json['use_head']:
        raise ValueError('Expected the native image-only AIMv2 trunk without an extra head')
    processor = CLIPImageProcessorFast.from_pretrained(folder, local_files_only=True)
    if manifest is not None:
        if (manifest['model'] != REPO or manifest['revision'] != REVISION
                or manifest['feature_dim'] != 1024 or manifest['batch_size'] != 1
                or manifest['mllm_components'] is not False):
            raise ValueError('Registered AIMv2 feature contract differs')
        if (digest(checkpoint) != manifest['checkpoint_sha256']
                or digest(folder / 'config.json') != manifest['config_sha256']
                or processor.to_dict() != manifest['preprocess']
                or digest(__file__) != manifest['encoder_source_sha256']):
            raise ValueError('Registered AIMv2 weights, configuration or adapter changed')
    # Explicit strict loading catches every missing/unexpected checkpoint key.
    model = Aimv2VisionModel(Aimv2VisionConfig(**config_json)).half()
    model.load_state_dict(load_file(checkpoint, device='cpu'), strict=True)
    model = model.cuda().eval().requires_grad_(False)

    def encode(images):
        rows = []
        with torch.inference_mode():
            for image in images:
                inputs = processor(images=[image], return_tensors='pt', device='cpu')
                output = model(pixel_values=inputs['pixel_values'].to('cuda', torch.float16))
                tokens = output.last_hidden_state.float()
                if tokens.shape != (1, 256, 1024) or not torch.isfinite(tokens).all():
                    raise ValueError('Expected 256 AIMv2 image patch tokens')
                # No CLS token exists in this checkpoint; mean of all post-norm image patches.
                value = tokens.mean(dim=1)
                if value.norm() <= 0:
                    raise ValueError('Zero AIMv2 feature')
                value = (value / value.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)
                rows.append(value)
        values = np.concatenate(rows).astype(np.float64)
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)

    return encode, processor
