"""Pinned PE-Core-G checkpoint shape audit and batch-one GPU resource probe."""
from pathlib import Path
import argparse
import time
import uuid
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import timm
from timm.data import create_transform
from safetensors import safe_open
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY

REVISION = 'fb096c09d13bd2e3827aed4dbdffb8a6b8e0c10b'


def build_meta(config):
    with torch.device('meta'):
        model = timm.create_model(config['architecture'], pretrained=False,
            num_classes=config['num_classes'], img_size=config['pretrained_cfg']['input_size'][-1])
    assert all(p.is_meta for p in model.parameters())
    return model


def stream_load(model, checkpoint):
    """Avoid simultaneously materializing the 7.5GB FP32 model in RAM."""
    expected = model.state_dict()
    with safe_open(str(checkpoint), framework='pt', device='cpu') as reader:
        if set(reader.keys()) != set(expected):
            raise ValueError('Checkpoint keys differ from the official Meta architecture')
        for key, tensor in expected.items():
            if list(reader.get_slice(key).get_shape()) != list(tensor.shape):
                raise ValueError(f'Checkpoint shape differs: {key}')
        for key in reader.keys():
            module_path, _, leaf = key.rpartition('.')
            module = model.get_submodule(module_path) if module_path else model
            value = reader.get_tensor(key).to(device='cuda', dtype=torch.float16)
            if leaf in module._parameters:
                module._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
            elif leaf in module._buffers:
                module._buffers[leaf] = value
            else:
                raise ValueError(f'Unexpected state entry: {key}')
    # Nonpersistent RoPE cache is absent from the checkpoint. Generate the official
    # formula in FP32, then cast as an ordinary CPU-build -> half -> CUDA would.
    meta_buffers = [name for name, buf in model.named_buffers() if buf.is_meta]
    if meta_buffers != ['rope.pos_embed']:
        raise ValueError(f'Unexpected nonpersistent Meta buffers: {meta_buffers}')
    rope = model.rope._get_pos_embed_values(model.rope.feat_shape, device='cpu', dtype=torch.float32)
    assert rope.shape == model.rope.pos_embed.shape and torch.isfinite(rope).all()
    assert torch.count_nonzero(rope) > 0
    model.rope.pos_embed = rope.to(device='cuda', dtype=torch.float16)
    assert not any(p.is_meta for p in model.parameters())
    assert not any(b.is_meta for b in model.buffers())
    return model.eval()


def main(shape_only=False):
    weights = local_path('cache_root') / 'pe_core_giant_weights' / REVISION
    config = read_json(weights / 'config.json')
    provenance = read_json(weights / 'download.json')
    checkpoint = weights / 'model.safetensors'
    assert provenance['repo'] == 'timm/vit_pe_core_gigantic_patch14_448.fb'
    assert provenance['revision'] == REVISION
    assert provenance['files'][checkpoint.name]['publisher_lfs_verified']
    assert digest(weights / 'config.json') == provenance['files']['config.json']['sha256']
    relative = Path('19_image_embedding_reference/pe_core_giant_preflight') / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    public = REPOSITORY / 'output/experiment_record' / relative
    write_json(public / 'registration.json', {'model': provenance['repo'], 'revision': REVISION,
        'source_sha256': digest(__file__), 'shape_only': shape_only, 'batch_size': 1,
        'probe_selection': 'First three training rows in frozen split, no prices consulted',
        'split_sha256': digest(local_path('split_path')), 'checkpoint_sha256': provenance['files'][checkpoint.name]['sha256'],
        'precision': 'FP16 parameters and forward; FP32 official RoPE then FP16 cast; FP32 output normalization',
        'mllm_components': False, 'test_images_used': False, 'text_modules_used': False})
    write_json(public / 'status.json', {'status': 'checking_shapes'})
    print(public, flush=True)
    try:
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        model = build_meta(config)
        state = model.state_dict()
        with safe_open(str(checkpoint), framework='pt', device='cpu') as reader:
            assert set(reader.keys()) == set(state)
            for key, tensor in state.items():
                assert list(reader.get_slice(key).get_shape()) == list(tensor.shape), key
        write_json(public / 'shape_audit.json', {'passed': True, 'tensors': len(state),
            'parameters': sum(p.numel() for p in model.parameters()),
            'meta_buffers': [{'name': n, 'shape': list(b.shape)} for n, b in model.named_buffers()]})
        if shape_only:
            write_json(public / 'status.json', {'status': 'completed', 'gpu_preflight_completed': False})
            return
        assert torch.cuda.is_available()
        assert digest(checkpoint) == provenance['files'][checkpoint.name]['sha256']
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        write_json(public / 'status.json', {'status': 'loading_gpu'})
        started = time.perf_counter()
        model = stream_load(model, checkpoint)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - started
        cfg = config['pretrained_cfg']
        transform = create_transform(input_size=cfg['input_size'], interpolation=cfg['interpolation'],
            mean=cfg['mean'], std=cfg['std'], crop_pct=cfg['crop_pct'], crop_mode=cfg['crop_mode'], is_training=False)
        frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str})
        names = frame.loc[frame.partition.eq('train'), 'imageid'].iloc[:3].tolist()
        outputs, timing, norms = [], [], []
        write_json(public / 'status.json', {'status': 'probing_gpu'})
        with torch.inference_mode():
            for name in names + names[:1]:
                with Image.open(local_path('data_root') / 'train' / name) as source:
                    value = transform(ImageOps.exif_transpose(source).convert('RGB')).unsqueeze(0).to('cuda', torch.float16)
                torch.cuda.synchronize()
                began = time.perf_counter()
                vector = model(value).float()
                torch.cuda.synchronize()
                timing.append(time.perf_counter() - began)
                assert vector.shape == (1, 1280) and torch.isfinite(vector).all()
                norm = vector.norm(dim=-1, keepdim=True)
                assert (norm > 0).all()
                norms.append(float(norm.item()))
                outputs.append((vector / norm).cpu().numpy()[0])
        np.testing.assert_allclose(outputs[0], outputs[-1], rtol=1e-4, atol=3e-6)
        results = {'passed': True, 'gpu': torch.cuda.get_device_name(), 'imageids': names,
            'preprocess': repr(transform), 'load_seconds': load_seconds, 'forward_seconds': timing,
            'raw_feature_norms': norms, 'repeat_max_abs_error': float(np.max(abs(outputs[0] - outputs[-1]))),
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
            'scope': 'Resource and numerical probe only; no price evaluation or full extraction'}
        write_json(public / 'results.json', results)
        write_json(public / 'status.json', {'status': 'completed', 'gpu_preflight_completed': True})
        print(results, flush=True)
    except BaseException as exc:
        write_json(public / 'status.json', {'status': 'failed', 'error': repr(exc), 'accuracy_conclusion': None})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape-only', action='store_true')
    main(parser.parse_args().shape_only)
