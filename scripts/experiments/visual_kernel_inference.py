"""Inference keyed by exact registered encoder/preprocessing, never family alone."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY


def normalize(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def encoder(family, manifest):
    import torch
    import open_clip
    from PIL import Image, ImageOps
    from visual_kernel_submission import original_clip_cache
    multi = family in {'clip_multiview', 'siglip_multiview'}
    source_family = 'siglip_features' if family == 'siglip_multiview' else family
    source = original_clip_cache(manifest, 'siglip_features' if family == 'siglip_multiview' else 'clip_features')[1] if multi else manifest
    if family == 'pe_core_multiview':
        from pe_core_multiview import views
        if digest(Path(__file__).with_name('pe_core_multiview.py')) != manifest['source_sha256']:
            raise ValueError('Registered PE multiview transformation source changed')
        original_folder = local_path('cache_root') / 'pe_core_features' / manifest['original_cache_key']
        original_path = original_folder / 'manifest.json'
        original = read_json(original_path)
        if digest(original_path) != manifest['original_manifest_sha256'] or original['model'] != manifest['model'] or original['revision'] != manifest['revision']:
            raise ValueError('Registered PE full-image source changed')
        if manifest['feature_dim'] != 3072 or manifest['batch_size'] != 4 or original.get('mllm_components') is not False:
            raise ValueError('Unsupported PE multiview feature contract')
        base_encode = encoder('pe_core_features', original)
        def encode(images):
            whole, detail = views(images)
            return normalize(np.concatenate([base_encode(images), base_encode(whole), base_encode(detail)], axis=1))
        return encode
    if family == 'pe_core_giant_whole_features':
        from pe_core_giant_whole_encoder import make_encoder
        return make_encoder(manifest)
    if family == 'pe_core_giant_features':
        from pe_core_giant_encoder import make_encoder
        return make_encoder(manifest)
    if family == 'siglip2_giant_features':
        from siglip_giant_encoder import make_encoder
        return make_encoder(manifest)
    if family == 'pe_core_features':
        import timm
        from timm.data import resolve_model_data_config, create_transform
        from safetensors.torch import load_file
        weights_dir = local_path('cache_root') / 'pe_core_weights' / source['revision']
        provenance = read_json(weights_dir / 'download.json')
        checkpoint = weights_dir / provenance['weight_file']
        if digest(checkpoint) != source['checkpoint_sha256']:
            raise ValueError('Registered PE-Core checkpoint changed')
        config = read_json(weights_dir / 'config.json')
        visual = timm.create_model(config['architecture'], pretrained=False, num_classes=config['num_classes'],
            img_size=config['pretrained_cfg']['input_size'][-1], pretrained_cfg=config['pretrained_cfg'])
        visual.load_state_dict(load_file(checkpoint, device='cpu'), strict=True)
        processor = create_transform(**resolve_model_data_config(visual), is_training=False)
        if repr(processor) != source['preprocess']:
            raise ValueError('Registered PE-Core preprocessing changed')
        visual = visual.half().cuda().eval().requires_grad_(False)
        def encode(images):
            with torch.inference_mode():
                value = visual(torch.stack([processor(im) for im in images]).half().cuda()).float()
                value = (value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
            return normalize(value)
        return encode
    if family == 'siglip2_naflex_features':
        from safetensors import safe_open
        from transformers import Siglip2VisionModel, Siglip2VisionConfig, Siglip2ImageProcessorFast
        weights_dir = local_path('cache_root') / 'siglip2_naflex_weights' / source['revision']
        provenance = read_json(weights_dir / 'download.json')
        checkpoint = weights_dir / provenance['weight_file']
        if digest(checkpoint) != source['checkpoint_sha256']:
            raise ValueError('Registered NaFlex checkpoint changed')
        processor = Siglip2ImageProcessorFast.from_pretrained(weights_dir, local_files_only=True, max_num_patches=source['max_num_patches'])
        if processor.to_dict() != source['preprocess']:
            raise ValueError('Registered NaFlex image processor changed')
        config = read_json(weights_dir / 'config.json')
        if config['model_type'] != 'siglip2':
            raise ValueError('NaFlex requires the Siglip2 architecture')
        visual = Siglip2VisionModel(Siglip2VisionConfig(**config['vision_config']))
        with safe_open(checkpoint, framework='pt', device='cpu') as tensors:
            state = {k: tensors.get_tensor(k) for k in tensors.keys() if k.startswith('vision_model.')}
        visual.load_state_dict(state, strict=True)
        del state
        visual = visual.half().cuda().eval().requires_grad_(False)
        def encode(images):
            inputs = processor(images=images, return_tensors='pt', device='cpu')
            inputs = {k: v.to(device='cuda', dtype=torch.float16) if k == 'pixel_values' else v.cuda() for k, v in inputs.items()}
            with torch.inference_mode():
                value = visual(**inputs).pooler_output.float()
                value = (value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
            return normalize(value)
        return encode
    if family == 'dinov2_features':
        from safetensors import safe_open
        from transformers import Dinov2Model, Dinov2Config, BitImageProcessor
        weights_dir = local_path('cache_root') / 'dinov2_weights' / source['revision']
        provenance = read_json(weights_dir / 'download.json')
        checkpoint = weights_dir / provenance['weight_file']
        if digest(checkpoint) != source['checkpoint_sha256']:
            raise ValueError('Registered DINOv2 checkpoint changed')
        processor = BitImageProcessor.from_pretrained(weights_dir, local_files_only=True)
        if processor.to_dict() != source['preprocess']:
            raise ValueError('Registered DINOv2 preprocessing changed')
        visual = Dinov2Model(Dinov2Config(**read_json(weights_dir / 'config.json')))
        if checkpoint.suffix == '.safetensors':
            with safe_open(checkpoint, framework='pt', device='cpu') as tensors:
                state = {k: tensors.get_tensor(k) for k in tensors.keys()}
        else:
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        visual.load_state_dict(state, strict=True)
        del state
        visual = visual.half().cuda().eval().requires_grad_(False)
        def encode(images):
            pixels = processor(images=images, return_tensors='pt')['pixel_values'].half().cuda()
            with torch.inference_mode():
                tokens = visual(pixel_values=pixels).last_hidden_state.float()
                parts = [tokens[:, 0], tokens[:, 1:].mean(dim=1)]
                parts = [(value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu().numpy() for value in parts]
            return normalize(np.concatenate(parts, axis=1))
        return encode
    if family == 'siglip_multiview':
        base_encode = encoder('siglip_features', source)
        def encode(images):
            size = int(manifest['view_size'])
            whole = [ImageOps.pad(im, (size, size), method=Image.Resampling.BICUBIC, color=(128, 128, 128)) for im in images]
            center = [im.crop((round(im.width * .2), round(im.height * .2), round(im.width * .8), round(im.height * .8))) for im in images]
            return normalize(np.concatenate([base_encode(images), base_encode(whole), base_encode(center)], axis=1))
        return encode
    projection = None
    if family in {'clip_features', 'clip_multiview'}:
        model_name = source['model']
        weights = local_path('cache_root') / 'clip_weights' / (model_name + '.pt')
        if digest(weights) != source['checkpoint_sha256']:
            raise ValueError('Registered CLIP weights changed')
        cfg = open_clip.get_pretrained_cfg(model_name, 'openai')
        model, _, processor = open_clip.create_model_and_transforms(model_name, pretrained=str(weights),
            force_quick_gelu=True, weights_only=False, image_mean=cfg['mean'], image_std=cfg['std'],
            image_interpolation=cfg['interpolation'], image_resize_mode=cfg['resize_mode'])
        if repr(processor) != source['preprocess'] or source['precision'] != 'float32':
            raise ValueError('Registered CLIP preprocessing or precision changed')
        visual = model.visual.float().cuda().eval().requires_grad_(False)
        del model
        size = visual.image_size
        size = (size, size) if isinstance(size, int) else tuple(size)

        def encode(images):
            buckets = [images]
            if multi:
                buckets += [[ImageOps.pad(im, size, method=Image.Resampling.BICUBIC, color=(128, 128, 128)) for im in images],
                    [im.crop((round(im.width * .2), round(im.height * .2), round(im.width * .8), round(im.height * .8))) for im in images]]
            parts = []
            with torch.inference_mode():
                for bucket in buckets:
                    value = visual(torch.stack([processor(im) for im in bucket]).cuda()).float()
                    parts.append((value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu().numpy())
            return normalize(np.concatenate(parts, axis=1))
    else:
        folders = {'siglip_features': 'siglip2_weights', 'streetclip_features': 'streetclip_weights', 'blip_features': 'blip_weights'}
        if family not in folders:
            raise ValueError('Unsupported visual family; no MLLM components accepted')
        weights_dir = local_path('cache_root') / folders[family] / source['revision']
        provenance = read_json(weights_dir / 'download.json')
        weights = weights_dir / provenance['weight_file']
        if digest(weights) != source['checkpoint_sha256']:
            raise ValueError('Registered vision checkpoint changed')
        config = read_json(weights_dir / 'config.json')
        if family == 'siglip_features':
            from safetensors import safe_open
            from transformers import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel
            processor = SiglipImageProcessor.from_pretrained(weights_dir, local_files_only=True)
            visual = SiglipVisionModel(SiglipVisionConfig(**config['vision_config']))
            with safe_open(weights, framework='pt', device='cpu') as tensors:
                state = {k: tensors.get_tensor(k) for k in tensors.keys() if k.startswith('vision_model.')}
            visual.load_state_dict(state, strict=True)
            del state
            visual = visual.half().cuda().eval().requires_grad_(False)
            if source['precision'] != 'float16 visual forward, float32 normalized cached features':
                raise ValueError('Registered SigLIP precision changed')
        else:
            state = torch.load(weights, map_location='cpu', weights_only=True)
            if family == 'blip_features':
                from transformers import BlipImageProcessor, BlipVisionConfig, BlipVisionModel
                processor = BlipImageProcessor.from_pretrained(weights_dir, local_files_only=True)
                visual = BlipVisionModel(BlipVisionConfig(**config['vision_config']))
                visual.load_state_dict({k[len('vision_model.'):]: v for k, v in state.items() if k.startswith('vision_model.')}, strict=True)
            else:
                from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel
                processor = CLIPImageProcessor.from_pretrained(weights_dir, local_files_only=True)
                visual = CLIPVisionModel(CLIPVisionConfig(**config['vision_config']))
                selected = {k: v for k, v in state.items() if k.startswith('vision_model.')}
                position_key = 'vision_model.embeddings.position_ids'
                if position_key in selected and position_key not in visual.state_dict():
                    if not torch.equal(selected[position_key], visual.vision_model.embeddings.position_ids):
                        raise ValueError('Unexpected StreetCLIP legacy position buffer')
                    selected.pop(position_key)
                visual.load_state_dict(selected, strict=True)
                projection = torch.nn.Linear(config['vision_config']['hidden_size'], config['projection_dim'], bias=False)
                projection.weight.data.copy_(state['visual_projection.weight'])
                projection = projection.float().cuda().eval().requires_grad_(False)
                del selected
            del state
            visual = visual.float().cuda().eval().requires_grad_(False)
        if processor.to_dict() != source['preprocess']:
            raise ValueError('Registered vision preprocessing changed')

        def encode(images):
            pixels = processor(images=images, return_tensors='pt')['pixel_values'].to(device='cuda', dtype=next(visual.parameters()).dtype)
            with torch.inference_mode():
                value = visual(pixel_values=pixels).pooler_output
                value = projection(value) if projection is not None else value
                value = value.float()
                value = (value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
            return normalize(value)
    return encode


def infer(out):
    import torch
    from PIL import Image, ImageOps
    from visual_kernel_submission import features_for
    from visual_kernel_components import feature_registration, predict_component
    out = Path(out).resolve()
    public = REPOSITORY / 'output/experiment_record' / out.relative_to(local_path('records_root'))
    prepared, registration = read_json(out / 'prepared.json'), read_json(out / 'registration.json')
    if not prepared['reference_replay_passed'] or not prepared['weights_frozen'] or prepared['full_train_n'] != 8000:
        raise ValueError('Full-label refit is not verified')
    if (out / 'predictions.csv').exists():
        raise FileExistsError('Completed prediction artifacts are immutable')
    adapter = Path(__file__).with_name('visual_kernel_components.py')
    if 'component_adapter_sha256' in registration and digest(adapter) != registration['component_adapter_sha256']:
        raise ValueError('Registered kernel adapter changed after full refit')
    (out / 'inference_source_at_launch.py').write_bytes(Path(__file__).read_bytes())
    data_root = local_path('data_root')
    if digest(data_root / 'train.csv') != registration['train_csv_sha256'] or digest(data_root / 'test.csv') != registration['test_csv_sha256']:
        raise ValueError('Registered label/test ID CSV changed')
    test = pd.read_csv(data_root / 'test.csv', dtype={'imageid': str})
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    if len(test) != 3000 or not test.imageid.is_unique or set(test.imageid) & set(frame.imageid):
        raise ValueError('Invalid test coverage')
    indices = np.random.default_rng(2026).choice(8000, 8, replace=False)
    def images(folder, names):
        result = []
        for name in names:
            with Image.open(data_root / folder / name) as source:
                result.append(ImageOps.exif_transpose(source).convert('RGB'))
        return result
    probes = images('train', frame.imageid.iloc[indices])
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    registration.update(inference_source_sha256=digest(__file__), test_images_used=True,
        test_images_role='Inference only; no training, target access, model selection or calibration')
    write_json(out / 'registration.json', registration)
    write_json(public / 'registration.json', registration)
    write_json(out / 'test_manifest.json', {'imageids': test.imageid.tolist(),
        'file_sha256': [digest(data_root / 'test' / name) for name in test.imageid], 'test_labels_used': False})
    test_manifest_sha = digest(out / 'test_manifest.json')
    def status(state, **extra):
        write_json(public / 'status.json', {'status': state, 'pid': os.getpid(), **extra})
    try:
        outputs, spotchecks = [], {}
        for index, component in enumerate(registration['components']):
            family, key = component['cache_family'], component['cache_key']
            manifest_path = local_path('cache_root') / family / key / 'manifest.json'
            source_registration = read_json(REPOSITORY / 'output/experiment_record' / component['source'] / 'registration.json')
            if 'source_registration_sha256' in component and digest(REPOSITORY / 'output/experiment_record' / component['source'] / 'registration.json') != component['source_registration_sha256']:
                raise ValueError('Selected component registration changed')
            source_registration = feature_registration(source_registration)
            if digest(manifest_path) != source_registration['manifest_sha256']:
                raise ValueError('Selected feature manifest changed')
            manifest = read_json(manifest_path)
            source_batch = int(manifest.get('batch_size', 4))
            if source_batch not in (1, 2, 4):
                raise ValueError('Unsupported registered inference batch size')
            encode = encoder(family, manifest)
            # Match training extraction batch size, including fp16 image towers.
            probe_values = np.concatenate([encode(probes[start:start + source_batch]) for start in range(0, 8, source_batch)])
            expected = features_for(source_registration, frame)
            np.testing.assert_allclose(probe_values, expected[indices], rtol=1e-4, atol=3e-6)
            del expected
            spotchecks[family + ':' + key] = True
            cache_manifest = {'family': family, 'source_cache_key': key, 'source_manifest_sha256': digest(manifest_path),
                'test_manifest_sha256': test_manifest_sha, 'inference_source_sha256': digest(__file__),
                'feature_dim': component['feature_dim'], 'batch_size': source_batch, 'test_labels_used': False}
            cache_key = hashlib.sha256(json.dumps(cache_manifest, sort_keys=True).encode()).hexdigest()
            cache = local_path('cache_root') / 'test_visual_features' / cache_key
            cache.mkdir(parents=True, exist_ok=True)
            checksum_path = cache / 'chunk_sha256.json'
            checksums = read_json(checksum_path) if checksum_path.exists() else {}
            write_json(cache / 'manifest.json', cache_manifest)
            parts = []
            for offset in range(0, 3000, source_batch):
                names = test.imageid.iloc[offset:offset + source_batch]
                path = cache / f'{offset:05d}_{len(names)}.npy'
                if path.exists():
                    if path.name not in checksums or digest(path) != checksums[path.name]:
                        raise ValueError('Incomplete or modified test feature chunk')
                    value = np.load(path, allow_pickle=False)
                else:
                    value = encode(images('test', names))
                    with path.open('xb') as stream:
                        np.save(stream, value)
                    checksums[path.name] = digest(path)
                    write_json(checksum_path, checksums)
                if value.shape != (len(names), component['feature_dim']) or not np.isfinite(value).all() or not np.allclose(np.linalg.norm(value, axis=1), 1, atol=1e-8):
                    raise ValueError('Invalid normalized test features')
                parts.append(value)
                if offset % 200 == 0:
                    status('inferring_test', component=index, family=family, cache_key=key, images=offset + len(names), total=3000)
                    print(f'Component {index + 1}: {family} {offset + len(names)}/3000', flush=True)
            del encode
            torch.cuda.empty_cache()
            features = np.concatenate(parts)
            outputs.append(predict_component(out, index, component, features))
        prediction = np.stack(outputs, axis=1) @ np.asarray(registration['weights'])
        if prediction.shape != (3000,) or not np.isfinite(prediction).all() or (prediction < 0).any():
            raise ValueError('Invalid prediction coverage/value')
        csv = out / 'predictions.csv'
        pd.DataFrame({'imageid': test.imageid, 'price': prediction}).to_csv(csv, index=False, encoding='utf-8', mode='x')
        restored = pd.read_csv(csv, dtype={'imageid': str})
        if list(restored) != ['imageid', 'price'] or restored.imageid.tolist() != test.imageid.tolist() or len(restored) != 3000:
            raise ValueError('Submission schema/order failed')
        write_json(public / 'checks.json', {'n': 3000, 'finite_nonnegative': True, 'test_order_matches': True,
            'csv_sha256': digest(csv), 'min_price': float(prediction.min()), 'max_price': float(prediction.max()),
            'reference_replay_passed': True, 'test_images_used': True, 'test_images_role': 'Inference only',
            'test_labels_used': False, 'mllm_components': False, 'labeled_feature_spotchecks': spotchecks,
            'test_manifest_sha256': test_manifest_sha, 'kaggle_uploaded': False, 'price_unit': 'thousand USD'})
        status('completed', test_inference_completed=True)
        print('Verified visual-only Kaggle candidate:', csv, flush=True)
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise
