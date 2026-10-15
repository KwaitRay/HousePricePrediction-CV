"""Probe two-view zero-LoRA prices and real-target gradients; no optimizer step."""
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
from pe_giant_lora_preflight import LowRankLinear, unit
from pe_giant_whole_view import whole_view

SOURCE = '19_image_embedding_reference/pe_core_g14_448_whole_kernel/20261014T090655Z_2d55c743'


def main():
    root = local_path('records_root')
    source_registration = read_json(REPOSITORY / 'output/experiment_record' / SOURCE / 'registration.json')
    cache = local_path('cache_root') / source_registration['cache_family'] / source_registration['feature_cache_key']
    manifest = read_json(cache / 'manifest.json')
    assert digest(cache / 'manifest.json') == source_registration['manifest_sha256']
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str})
    indices = np.flatnonzero(frame.partition.eq('train').to_numpy())[:3]
    names = frame.imageid.iloc[indices].tolist()
    relative = Path('19_image_embedding_reference/pe_lora_prototype_preflight') / (
        time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    out, public = root / relative, REPOSITORY / 'output/experiment_record' / relative
    out.mkdir(parents=True, exist_ok=False)
    registration = {'source_sha256': digest(__file__), 'source_kernel': SOURCE,
        'source_head_sha256': digest(root / SOURCE / 'head.npz'),
        'feature_manifest_sha256': digest(cache / 'manifest.json'),
        'split_sha256': digest(local_path('split_path')), 'probe_imageids': names,
        'base_model': manifest['model'], 'revision': REVISION, 'rank': 4, 'adapter_alpha': 4,
        'frozen_price_head': 'Original 6399-training-label RBF head and original teacher feature bank; learn query encoder only',
        'feature_policy': 'Original view + fixed whole view; concatenate unit vectors, normalize, round FP32 as cache, renormalize F64',
        'dependencies_sha256': {name:digest(Path(__file__).with_name(name)) for name in
            ['pe_core_giant_preflight.py','pe_giant_lora_preflight.py','pe_giant_whole_view.py']},
        'feature_tolerance': {'rtol':1e-4,'atol':3e-6}, 'price_tolerance': {'rtol':1e-7,'atol':1e-4},
        'purpose': 'Numerical/gradient/resource check on fixed three training images; no accuracy selection',
        'price_model_evaluated': False, 'optimizer_step_performed': False, 'test_images_used': False}
    write_json(public / 'registration.json', registration)
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    torch.cuda.reset_peak_memory_stats()
    folder = local_path('cache_root') / 'pe_core_giant_weights' / REVISION
    config = read_json(folder / 'config.json')
    checkpoint = folder / 'model.safetensors'
    assert digest(checkpoint) == manifest['checkpoint_sha256']
    model = stream_load(build_meta(config), checkpoint).eval().requires_grad_(False)
    cfg = config['pretrained_cfg']
    processor = create_transform(input_size=cfg['input_size'], interpolation=cfg['interpolation'],
        mean=cfg['mean'],std=cfg['std'],crop_pct=cfg['crop_pct'],crop_mode=cfg['crop_mode'],is_training=False)
    assert repr(processor) == manifest['preprocess']
    for i in range(len(model.blocks)-2,len(model.blocks)):
        for leaf in ('qkv','proj'):
            parent = model.get_submodule(f'blocks.{i}.attn')
            setattr(parent,leaf,LowRankLinear(getattr(parent,leaf)))
    model.eval()
    with np.load(root / SOURCE / 'head.npz',allow_pickle=False) as head:
        anchors = torch.as_tensor(head['train_features'],device='cuda',dtype=torch.float64)
        coefficients = torch.as_tensor(head['coefficients'],device='cuda',dtype=torch.float64)
        gamma, mean = float(head['gamma']), float(head['target_mean'])
    assert anchors.shape == (6399,2560) and not anchors.requires_grad
    def query(image):
        pixels = [processor(im).unsqueeze(0).to('cuda',torch.float16) for im in [image,whole_view(image)]]
        value = torch.cat([unit(model(x)) for x in pixels],dim=1)
        value = value/value.norm(dim=-1,keepdim=True)
        value = value.float().double()
        return value/value.norm(dim=-1,keepdim=True)
    def price(value):
        distances = torch.clamp(2-2*value@anchors.T,min=0)
        return (torch.exp(-gamma*distances)@coefficients+mean)*1000
    maximum_feature, maximum_price, gradients = 0.,0.,[]
    began = time.perf_counter()
    for index,name in zip(indices,names):
        with Image.open(local_path('data_root')/'train'/name) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
        expected = np.load(cache/f'{index:05d}_1.npy',allow_pickle=False).astype(np.float64)
        expected /= np.linalg.norm(expected,axis=1,keepdims=True)
        expected_tensor = torch.as_tensor(expected,device='cuda',dtype=torch.float64)
        with torch.no_grad():
            expected_price = price(expected_tensor)
        model.zero_grad(set_to_none=True)
        value = query(image)
        actual_price = price(value)
        np.testing.assert_allclose(value.detach().cpu().numpy(),expected,**registration['feature_tolerance'])
        np.testing.assert_allclose(actual_price.detach().cpu().numpy(),expected_price.cpu().numpy(),**registration['price_tolerance'])
        maximum_feature = max(maximum_feature,float(np.max(abs(value.detach().cpu().numpy()-expected))))
        maximum_price = max(maximum_price,float(torch.max(abs(actual_price.detach()-expected_price))))
        # Real supplied training price, used only to verify the intended gradient path.
        loss = (actual_price/1000-float(frame.price.iloc[index])/1000).square().mean()
        loss.backward()
        total = 0.
        for parameter in model.parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                total += float(parameter.grad.float().norm())
            else:
                assert parameter.grad is None
        assert total > 0
        gradients.append(total)
    result = {'passed': True, 'max_abs_feature_replay_error':maximum_feature,
        'max_abs_price_replay_error':maximum_price,'gradient_norm_sums':gradients,
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'seconds':time.perf_counter()-began,
        'price_model_evaluated':False,'optimizer_step_performed':False,'test_images_used':False}
    write_json(public/'results.json',result)
    write_json(public/'status.json',{'status':'completed'})
    print('LORA_PROTOTYPE_PREFLIGHT_PASSED',public,flush=True)
    print(result,flush=True)


if __name__=='__main__':
    main()
