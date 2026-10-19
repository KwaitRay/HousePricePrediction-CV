"""Frozen PE-G block40 patch means, official crop and whole view, explicit runtime."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from timm.data import create_transform
from PIL import Image
from common.config import read_json,digest
from common.local_paths import local_path
from pe_core_giant_preflight import REVISION,build_meta,stream_load
from pe_giant_whole_view import whole_view

MODEL='timm/vit_pe_core_gigantic_patch14_448.fb'
CHECKPOINT_SHA='a0a499d4a7c6d35bb5f11aa2b076eaacde06e66a08176eb3b3d362d80eb91bd8'

def make_encoder():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    folder=local_path('cache_root')/'pe_core_giant_weights'/REVISION
    config=read_json(folder/'config.json');provenance=read_json(folder/'download.json')
    assert provenance['repo']==MODEL and provenance['revision']==REVISION
    assert provenance['files']['model.safetensors']['publisher_lfs_verified']
    assert digest(folder/'model.safetensors')==CHECKPOINT_SHA==provenance['files']['model.safetensors']['sha256']
    assert config['architecture']=='vit_pe_core_gigantic_patch14_448' and config['num_features']==1536
    model=stream_load(build_meta(config),folder/'model.safetensors').eval().requires_grad_(False)
    assert len(model.blocks)==50 and model.num_prefix_tokens==0
    cfg=config['pretrained_cfg']
    processor=create_transform(input_size=cfg['input_size'],interpolation=cfg['interpolation'],
        mean=cfg['mean'],std=cfg['std'],crop_pct=cfg['crop_pct'],crop_mode=cfg['crop_mode'],is_training=False)
    def tensors(image):return [processor(im).unsqueeze(0).to('cuda',torch.float16) for im in [image,whole_view(image)]]
    def token_mean(tokens):
        assert tokens.shape==(1,1024,1536) and torch.isfinite(tokens).all()
        value=tokens.float().mean(dim=1)
        assert torch.isfinite(value).all() and value.norm()>0
        return value/value.norm(dim=-1,keepdim=True)
    def encode(images):
        rows=[]
        with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
            for image in images:
                views=[]
                for pixels in tensors(image):
                    features=model.forward_intermediates(pixels,indices=[39],norm=False,stop_early=True,output_fmt='NLC',intermediates_only=True)
                    assert len(features)==1
                    views.append(token_mean(features[0]).cpu().numpy())
                value=np.concatenate(views,axis=1).astype(np.float64)
                value/=np.linalg.norm(value,axis=1,keepdims=True)
                value=value.astype(np.float32).astype(np.float64)
                rows.append(value/np.linalg.norm(value,axis=1,keepdims=True))
        return np.concatenate(rows)
    def hook_control(images):
        errors=[];captured=[]
        def hook(module,args,output):captured.append(token_mean(output).cpu().numpy())
        handle=model.blocks[39].register_forward_hook(hook)
        try:
            with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
                for image in images:
                    for pixels in tensors(image):
                        captured.clear();output=model(pixels)
                        assert output.shape==(1,1280) and len(captured)==1
                        full=captured[0].copy();captured.clear()
                        intermediate=model.forward_intermediates(pixels,indices=[39],norm=False,stop_early=True,output_fmt='NLC',intermediates_only=True)
                        early=token_mean(intermediate[0]).cpu().numpy()
                        np.testing.assert_allclose(full,early,rtol=1e-4,atol=3e-6)
                        errors.append(float(abs(full-early).max()))
        finally:handle.remove()
        return errors
    # Fixed neutral-image warm-up, independent of prices and candidate outcomes.
    encode([Image.new('RGB',(448,448),(128,128,128))]*2)
    return encode,hook_control,repr(processor)
