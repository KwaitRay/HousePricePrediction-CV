"""Zero-LoRA equivalence and gradient/memory preflight; no price training."""
import os
import time
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from timm.data import create_transform
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from pe_core_giant_preflight import REVISION, build_meta, stream_load

ORIGINAL = '19_image_embedding_reference/pe_core_g14_448/20261014T065137Z_b822c3d4'


class LowRankLinear(torch.nn.Module):
    def __init__(self, base, rank=4):
        super().__init__()
        assert isinstance(base, torch.nn.Linear)
        self.base = base.requires_grad_(False)
        self.a = torch.nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        torch.nn.init.normal_(self.a, std=.01)
        self.scale = 1.  # alpha=rank=4

    def forward(self, x):
        delta = torch.nn.functional.linear(torch.nn.functional.linear(x.float(), self.a), self.b)
        return self.base(x) + (delta*self.scale).to(x.dtype)


def unit(value):
    value = value.float()
    value = value / value.norm(dim=-1, keepdim=True)
    value = value.double()
    return value / value.norm(dim=-1, keepdim=True)


def main():
    folder = local_path('cache_root') / 'pe_core_giant_weights' / REVISION
    config = read_json(folder / 'config.json')
    provenance = read_json(folder / 'download.json')
    checkpoint = folder / 'model.safetensors'
    assert provenance['files'][checkpoint.name]['publisher_lfs_verified']
    assert digest(checkpoint) == provenance['files'][checkpoint.name]['sha256']
    original_reg = read_json(REPOSITORY / 'output/experiment_record' / ORIGINAL / 'registration.json')
    cache = local_path('cache_root') / 'pe_core_giant_features' / original_reg['cache_key']
    manifest = read_json(cache / 'manifest.json')
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str})
    indices = np.flatnonzero(frame.partition.eq('train').to_numpy())[:3]
    names = frame.imageid.iloc[indices].tolist()
    relative = Path('19_image_embedding_reference/pe_giant_lora_resource_preflight') / (
        time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    out, public = local_path('records_root') / relative, REPOSITORY / 'output/experiment_record' / relative
    out.mkdir(parents=True, exist_ok=False)
    registration = {'source_sha256': digest(__file__), 'rank': 4, 'alpha': 4, 'blocks': 'Last two only',
        'target_modules': ['attn.qkv', 'attn.proj'], 'base_parameters_frozen': True,
        'adapter_precision': 'FP32 parameters and operations, delta cast to frozen FP16 base output',
        'base_checkpoint_sha256': digest(checkpoint), 'feature_manifest_sha256': digest(cache / 'manifest.json'),
        'split_sha256': digest(local_path('split_path')), 'probe_imageids': names,
        'loader_source_sha256': digest(Path(__file__).with_name('pe_core_giant_preflight.py')),
        'gradient_objective': 'First coordinate of unit image feature, no price targets and no optimizer step',
        'tolerance': {'rtol': 1e-4, 'atol': 3e-6}, 'price_model_evaluated': False,
        'test_images_used': False, 'mllm_components': False, 'automatic_training': False}
    assert registration['loader_source_sha256'] == manifest['loader_source_sha256']
    write_json(public / 'registration.json', registration)
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    def status(state, **extra):
        write_json(public / 'status.json', {'status': state, 'pid': os.getpid(), **extra})
    try:
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        torch.cuda.reset_peak_memory_stats()
        status('loading_gpu')
        model = stream_load(build_meta(config), checkpoint).eval().requires_grad_(False)
        cfg = config['pretrained_cfg']
        processor = create_transform(input_size=cfg['input_size'], interpolation=cfg['interpolation'],
            mean=cfg['mean'], std=cfg['std'], crop_pct=cfg['crop_pct'], crop_mode=cfg['crop_mode'], is_training=False)
        assert repr(processor) == manifest['preprocess']
        inputs = []
        for name in names:
            with Image.open(local_path('data_root') / 'train' / name) as image:
                inputs.append(processor(ImageOps.exif_transpose(image).convert('RGB')).unsqueeze(0).to('cuda', torch.float16))
        expected = np.concatenate([np.load(cache / f'{i:05d}_1.npy', allow_pickle=False) for i in indices]).astype(np.float64)
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        with torch.no_grad():
            base = torch.cat([unit(model(x)) for x in inputs]).cpu().numpy()
        np.testing.assert_allclose(base, expected, **registration['tolerance'])
        modules = []
        for i in range(len(model.blocks)-2, len(model.blocks)):
            for leaf in ('qkv','proj'):
                name = f'blocks.{i}.attn.{leaf}'
                module = model.get_submodule(name)
                model.get_submodule(f'blocks.{i}.attn').__setattr__(leaf, LowRankLinear(module))
                modules.append(name)
        model.eval()
        with torch.no_grad():
            modified = torch.cat([unit(model(x)) for x in inputs]).cpu().numpy()
        np.testing.assert_allclose(modified, expected, **registration['tolerance'])
        began = time.perf_counter()
        model.zero_grad(set_to_none=True)
        value = unit(model(inputs[0]))
        np.testing.assert_allclose(value.detach().cpu().numpy(), expected[:1], **registration['tolerance'])
        value[:,0].sum().backward()
        gradients = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                gradients[name] = float(parameter.grad.float().norm())
            else:
                assert parameter.grad is None
        assert sum(gradients.values()) > 0
        result = {'passed': True, 'injected_modules': modules,
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'base_feature_replay_error': float(np.max(abs(base-expected))),
            'zero_adapter_feature_replay_error': float(np.max(abs(modified-expected))),
            'gradient_norms': gradients, 'forward_backward_seconds': time.perf_counter()-began,
            'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
            'price_model_evaluated': False, 'optimizer_step_performed': False, 'test_images_used': False}
        write_json(public / 'results.json', result)
        status('completed')
        print('PE_G_LORA_PREFLIGHT_PASSED', public, flush=True)
        print(result, flush=True)
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
