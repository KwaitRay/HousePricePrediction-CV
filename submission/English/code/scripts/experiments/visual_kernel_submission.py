"""Freeze a visual-only ensemble, refit on 8000 labels, then export 3000 prices."""
import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import solve
from threadpoolctl import threadpool_limits
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY

L14_REFERENCE = Path('13_clip_optimization/ViT_L_14/20261013T113728Z_b33981a4')
FAMILIES = {'clip_features', 'clip_multiview', 'blip_features', 'streetclip_features', 'siglip_features', 'siglip_multiview', 'dinov2_features', 'siglip2_naflex_features', 'pe_core_features', 'siglip2_giant_features', 'pe_core_multiview', 'pe_core_giant_features', 'pe_core_giant_whole_features','pe_giant_mid40_features'}


def original_clip_cache(manifest, family='clip_features'):
    matches = [p.parent for p in (local_path('cache_root') / family).glob('*/manifest.json')
               if digest(p) == manifest['original_manifest_sha256']]
    if len(matches) != 1:
        raise ValueError('Expected one exact original CLIP feature manifest')
    return matches[0].name, read_json(matches[0] / 'manifest.json')


def cache_array(family, key, frame):
    folder = local_path('cache_root') / family / key
    manifest = read_json(folder / 'manifest.json')
    if manifest['imageids'] != frame.imageid.tolist() or manifest['split_sha256'] != digest(local_path('split_path')):
        raise ValueError('Feature cache and frozen split differ')
    array = np.concatenate([np.load(p, allow_pickle=False) for p in sorted(folder.glob('*.npy'))]).astype(np.float64)
    if len(array) != 8000 or not np.isfinite(array).all():
        raise ValueError('Invalid frozen feature cache')
    return array, manifest


def features_for(registration, frame):
    family, key = registration['cache_family'], registration['feature_cache_key']
    if family not in FAMILIES:
        raise ValueError('Only registered frozen visual components are supported; no MLLM features')
    values, manifest = cache_array(family, key, frame)
    if digest(local_path('cache_root') / family / key / 'manifest.json') != registration['manifest_sha256']:
        raise ValueError('Selected source feature manifest changed')
    if family in {'clip_multiview', 'siglip_multiview'}:
        original_family = 'siglip_features' if family == 'siglip_multiview' else 'clip_features'
        original_key, original_manifest = original_clip_cache(manifest, original_family)
        original, _ = cache_array(original_family, original_key, frame)
        dim = original.shape[1]
        if values.shape != (8000, 2, dim):
            raise ValueError('Unexpected multi-view shape')
        values = np.concatenate([original, values.reshape(8000, 2 * dim)], axis=1)
    if values.ndim != 2:
        raise ValueError('Expected two-dimensional image features')
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return values


def prepare(reference):
    from visual_kernel_components import feature_registration, replay_and_refit
    root = local_path('records_root')
    reference = reference.resolve()
    relative = reference.relative_to(root)
    public_reference = REPOSITORY / 'output/experiment_record' / relative
    if read_json(public_reference / 'status.json')['status'] != 'completed':
        raise ValueError('Ensemble reference is not completed')
    ensemble = read_json(reference / 'ensemble.json')
    weights = np.asarray(ensemble['weights'], dtype=np.float64)
    if len(weights) != len(ensemble['models']) or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError('Invalid fixed ensemble weights')
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    train = frame.partition.eq('train').to_numpy()
    valid = frame.partition.eq('validation').to_numpy()
    if len(frame) != 8000 or train.sum() != 6399 or valid.sum() != 1601 or set(frame.loc[train, 'group_id']) & set(frame.loc[valid, 'group_id']):
        raise ValueError('Unexpected development split')
    data_root = local_path('data_root')
    test = pd.read_csv(data_root / 'test.csv', dtype={'imageid': str})
    if len(test) != 3000 or not test.imageid.is_unique or set(test.imageid) & set(frame.imageid):
        raise ValueError('Unexpected test coverage')
    run = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
    out = root / '08_final/visual_kernel_ensemble' / run
    public = REPOSITORY / 'output/experiment_record/08_final/visual_kernel_ensemble' / run
    out.mkdir(parents=True, exist_ok=False)
    registration = {'method': 'Fixed visual-only convex kernel ensemble; full-label refit', 'reference': str(relative),
                    'reference_ensemble_sha256': digest(reference / 'ensemble.json'), 'weights': weights.tolist(),
                    'models': ensemble['models'], 'source_sha256': digest(__file__),
                    'reference_validation_mse': read_json(public_reference / 'results.json')['mse'],
                    'split_sha256': digest(local_path('split_path')), 'train_csv_sha256': digest(data_root / 'train.csv'),
                    'test_csv_sha256': digest(data_root / 'test.csv'), 'train_n': 8000, 'test_n': 3000,
                    'price_unit': 'thousand USD', 'target_scale': 1000, 'mllm_components': False,
                    'selection': 'Components, gamma, alpha and blend weights frozen before test inference',
                    'scope': 'Exploratory visual-only candidate; source course eligibility unresolved; not uploaded',
                    'test_images_used': False, 'test_labels_used': False, 'components': [],
                    'component_adapter_sha256': digest(Path(__file__).with_name('visual_kernel_components.py'))}
    (out / 'preparation_source.py').write_bytes(Path(__file__).read_bytes())
    (out / 'component_adapter_source.py').write_bytes(Path(__file__).with_name('visual_kernel_components.py').read_bytes())
    write_json(out / 'registration.json', registration)
    write_json(public / 'registration.json', registration)
    write_json(public / 'status.json', {'status': 'fitting_full', 'pid': os.getpid()})
    try:
        replay = []
        y = frame.price.to_numpy(dtype=np.float64) / 1000
        for index, name in enumerate(ensemble['models']):
            source = root / name
            source_public = REPOSITORY / 'output/experiment_record' / name
            if read_json(source_public / 'status.json')['status'] != 'completed':
                raise ValueError('Source kernel is not completed')
            if digest(source / 'head.npz') != ensemble['model_sha256'][index] or digest(source / 'predictions.csv') != ensemble['prediction_sha256'][index]:
                raise ValueError('Selected component artifacts changed')
            source_registration = read_json(source_public / 'registration.json')
            feature_source = feature_registration(source_registration)
            x = features_for(feature_source, frame)
            saved = pd.read_csv(source / 'predictions.csv', dtype={'imageid': str})
            if saved.imageid.tolist() != frame.loc[valid, 'imageid'].tolist():
                raise ValueError('Reference validation IDs differ')
            path = out / f'component_{index}_full_head.npz'
            old_raw, details = replay_and_refit(source, source_registration, x, train, valid, y, path, saved.raw_prediction.to_numpy())
            replay.append(np.maximum(old_raw, 0))
            registration['components'].append({'source': name, 'cache_family': feature_source['cache_family'],
                    'cache_key': feature_source['feature_cache_key'], 'source_registration_sha256': digest(source_public / 'registration.json'),
                    'feature_manifest_sha256': feature_source['manifest_sha256'], **details, 'reference_replay_passed': True})
            write_json(out / 'registration.json', registration)
            write_json(public / 'registration.json', registration)
            write_json(public / 'status.json', {'status': 'fitting_full', 'pid': os.getpid(), 'components_fitted': index + 1})
            print(f'Full-label kernel {index + 1}/{len(weights)} fitted; validation reference reproduced', flush=True)
            del x
        expected = pd.read_csv(reference / 'predictions.csv', dtype={'imageid': str})
        if expected.imageid.tolist() != frame.loc[valid, 'imageid'].tolist():
            raise ValueError('Ensemble validation IDs differ')
        np.testing.assert_allclose(np.stack(replay, axis=1) @ weights, expected.prediction.to_numpy(), rtol=1e-7, atol=1e-4)
        write_json(out / 'prepared.json', {'reference_replay_passed': True, 'full_train_n': 8000, 'weights_frozen': True})
        write_json(public / 'status.json', {'status': 'prepared', 'pid': os.getpid(), 'test_inference_completed': False})
        print('Prepared full-label visual candidate:', out, flush=True)
        return out
    except BaseException as exc:
        write_json(public / 'status.json', {'status': 'failed', 'pid': os.getpid(), 'error': repr(exc)})
        raise


def infer(out):
    # Resolve each component by its exact source cache, including mixed CLIP resolutions.
    from visual_kernel_inference import infer as registered_infer
    return registered_infer(out)


def legacy_infer(out):
    out = out.resolve()
    public = REPOSITORY / 'output/experiment_record' / out.relative_to(local_path('records_root'))
    prepared, registration = read_json(out / 'prepared.json'), read_json(out / 'registration.json')
    if not prepared['reference_replay_passed'] or not prepared['weights_frozen'] or prepared['full_train_n'] != 8000:
        raise ValueError('Candidate full refit has not been verified')
    data_root = local_path('data_root')
    if digest(data_root / 'train.csv') != registration['train_csv_sha256'] or digest(data_root / 'test.csv') != registration['test_csv_sha256']:
        raise ValueError('Source label/test ID CSV changed')
    test = pd.read_csv(data_root / 'test.csv', dtype={'imageid': str})
    import torch
    from PIL import Image, ImageOps
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    indices = np.random.default_rng(2026).choice(8000, 8, replace=False)
    probes = []
    for name in frame.imageid.iloc[indices]:
        with Image.open(data_root / 'train' / name) as source:
            probes.append(ImageOps.exif_transpose(source).convert('RGB'))
    spotchecks = {}
    registration['inference_source_sha256'] = digest(__file__)
    registration['test_images_used'] = True
    registration['test_images_role'] = 'Inference only; no training, target access, model selection or calibration'
    write_json(out / 'registration.json', registration)
    write_json(public / 'registration.json', registration)
    write_json(out / 'test_manifest.json', {'imageids': test.imageid.tolist(),
                'file_sha256': [digest(data_root / 'test' / name) for name in test.imageid],
                'test_labels_used': False})
    write_json(public / 'status.json', {'status': 'inferring_test', 'pid': os.getpid()})
    try:
        # Three model families; shared CLIP original view is computed only once.
        required = set(c['cache_family'] for c in registration['components'])
        test_features = {}
        if required & {'clip_features', 'clip_multiview'}:
            import open_clip
            checkpoint = local_path('cache_root') / 'clip_weights/ViT-L-14.pt'
            original_manifest = read_json(local_path('records_root') / L14_REFERENCE / 'data_manifest.json')
            if digest(checkpoint) != original_manifest['checkpoint_sha256']:
                raise ValueError('CLIP checkpoint changed')
            cfg = open_clip.get_pretrained_cfg('ViT-L-14', 'openai')
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=str(checkpoint),
                        force_quick_gelu=True, weights_only=False, image_mean=cfg['mean'], image_std=cfg['std'],
                        image_interpolation=cfg['interpolation'], image_resize_mode=cfg['resize_mode'])
            if repr(preprocess) != original_manifest['preprocess']:
                raise ValueError('CLIP selected preprocessing differs')
            visual = model.visual.float().cuda().eval().requires_grad_(False)
            del model
            probe_buckets = [[], [], []]
            for im in probes:
                w, h = im.size
                images = [im, ImageOps.pad(im, (224, 224), method=Image.Resampling.BICUBIC, color=(128, 128, 128)),
                          im.crop((round(w * .2), round(h * .2), round(w * .8), round(h * .8)))]
                for bucket, image in zip(probe_buckets, images):
                    bucket.append(preprocess(image))
            probe_values = []
            with torch.inference_mode():
                for bucket in probe_buckets:
                    values = visual(torch.stack(bucket).cuda()).float()
                    probe_values.append((values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu().numpy())
            original_key = hashlib.sha256(json.dumps(original_manifest, sort_keys=True).encode()).hexdigest()
            expected_original, _ = cache_array('clip_features', original_key, frame)
            np.testing.assert_allclose(probe_values[0], expected_original[indices], rtol=1e-4, atol=3e-6)
            spotchecks['clip_features'] = True
            if 'clip_multiview' in required:
                multi = next(c for c in registration['components'] if c['cache_family'] == 'clip_multiview')
                expected_multi, _ = cache_array('clip_multiview', multi['cache_key'], frame)
                np.testing.assert_allclose(np.stack(probe_values[1:], axis=1), expected_multi[indices], rtol=1e-4, atol=3e-6)
                spotchecks['clip_multiview'] = True
                del expected_multi
            del expected_original
            original, full, center = [], [], []
            for offset in range(0, len(test), 4):
                views = [[], [], []]
                for name in test.imageid.iloc[offset:offset + 4]:
                    with Image.open(data_root / 'test' / name) as source:
                        im = ImageOps.exif_transpose(source).convert('RGB')
                    w, h = im.size
                    images = [im, ImageOps.pad(im, (224, 224), method=Image.Resampling.BICUBIC, color=(128, 128, 128)),
                              im.crop((round(w * .2), round(h * .2), round(w * .8), round(h * .8)))]
                    for bucket, image in zip(views, images):
                        bucket.append(preprocess(image))
                with torch.inference_mode():
                    for bucket, saved in zip(views, [original, full, center]):
                        x = visual(torch.stack(bucket).cuda()).float()
                        saved.append((x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu().numpy())
                if offset % 400 == 0:
                    print(f'CLIP test views {offset + len(views[0])}/3000', flush=True)
            a, b, c = [np.concatenate(parts).astype(np.float64) for parts in [original, full, center]]
            test_features['clip_features'] = a
            test_features['clip_multiview'] = np.concatenate([a, b, c], axis=1)
            del visual
            torch.cuda.empty_cache()
        for family, weight_folder in [('blip_features', 'blip_weights'), ('streetclip_features', 'streetclip_weights')]:
            if family not in required:
                continue
            component = next(c for c in registration['components'] if c['cache_family'] == family)
            manifest = read_json(local_path('cache_root') / family / component['cache_key'] / 'manifest.json')
            revision = manifest['revision']
            weights = local_path('cache_root') / weight_folder / revision
            provenance = read_json(weights / 'download.json')
            checkpoint = weights / provenance['weight_file']
            if digest(checkpoint) != manifest['checkpoint_sha256']:
                raise ValueError(f'{family} checkpoint changed')
            config = read_json(weights / 'config.json')
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if family == 'blip_features':
                from transformers import BlipVisionConfig, BlipVisionModel, BlipImageProcessor
                processor = BlipImageProcessor.from_pretrained(weights, local_files_only=True)
                visual = BlipVisionModel(BlipVisionConfig(**config['vision_config']))
                visual.load_state_dict({k[len('vision_model.'):]: v for k, v in state.items() if k.startswith('vision_model.')}, strict=True)
                projection = None
            else:
                from transformers import CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor
                processor = CLIPImageProcessor.from_pretrained(weights, local_files_only=True)
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
            del state
            visual = visual.float().cuda().eval().requires_grad_(False)
            if processor.to_dict() != manifest['preprocess']:
                raise ValueError(f'{family} selected processor changed')
            with torch.inference_mode():
                pixels = processor(images=probes, return_tensors='pt')['pixel_values'].cuda()
                values = visual(pixel_values=pixels).pooler_output
                values = projection(values) if projection is not None else values
                values = values.float()
                values = (values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
            expected, _ = cache_array(family, component['cache_key'], frame)
            np.testing.assert_allclose(values, expected[indices], rtol=1e-4, atol=3e-6)
            spotchecks[family] = True
            del expected
            parts = []
            for offset in range(0, len(test), 4):
                images = []
                for name in test.imageid.iloc[offset:offset + 4]:
                    with Image.open(data_root / 'test' / name) as source:
                        images.append(ImageOps.exif_transpose(source).convert('RGB'))
                pixels = processor(images=images, return_tensors='pt')['pixel_values'].cuda()
                with torch.inference_mode():
                    x = visual(pixel_values=pixels).pooler_output
                    x = projection(x) if projection is not None else x
                    x = x.float()
                    parts.append((x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu().numpy())
                if offset % 400 == 0:
                    print(f'{family} test features {offset + len(images)}/3000', flush=True)
            test_features[family] = np.concatenate(parts).astype(np.float64)
            del visual, projection
            torch.cuda.empty_cache()
        outputs = []
        if not required.issubset(spotchecks):
            raise ValueError('A visual encoder has not passed independent labeled-image replay')
        for index, component in enumerate(registration['components']):
            features = test_features[component['cache_family']].copy()
            if features.shape != (3000, component['feature_dim']) or not np.isfinite(features).all():
                raise ValueError('Invalid test feature shape/values')
            features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
            path = out / f'component_{index}_full_head.npz'
            if digest(path) != component['full_head_sha256']:
                raise ValueError('Frozen full-training kernel changed')
            with np.load(path, allow_pickle=False) as head:
                chunks = []
                with threadpool_limits(limits=4):
                    for offset in range(0, 3000, 128):
                        distances = np.maximum(2 - 2 * (features[offset:offset + 128] @ head['train_features'].T), 0)
                        chunks.append((np.exp(-float(head['gamma']) * distances) @ head['coefficients'] + float(head['target_mean'])) * 1000)
                outputs.append(np.maximum(np.concatenate(chunks), 0))
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
                    'test_labels_used': False, 'mllm_components': False,
                    'labeled_feature_spotchecks': spotchecks, 'test_manifest_sha256': digest(out / 'test_manifest.json'),
                    'kaggle_uploaded': False, 'price_unit': 'thousand USD'})
        write_json(public / 'status.json', {'status': 'completed', 'pid': os.getpid(), 'test_inference_completed': True})
        print('Verified visual-only Kaggle candidate:', csv, flush=True)
    except BaseException as exc:
        write_json(public / 'status.json', {'status': 'failed', 'pid': os.getpid(), 'error': repr(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'infer'], required=True)
    parser.add_argument('--reference-dir', type=Path)
    parser.add_argument('--prepared-run', type=Path)
    args = parser.parse_args()
    if args.phase == 'prepare' and args.reference_dir and not args.prepared_run:
        prepare(args.reference_dir)
    elif args.phase == 'infer' and args.prepared_run and not args.reference_dir:
        infer(args.prepared_run)
    else:
        parser.error('prepare requires --reference-dir; infer requires --prepared-run')
