"""Reuse signed shared encoder caches; extract only new frozen image features."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from visual_kernel_components import feature_registration, predict_component
from visual_kernel_submission import features_for
from visual_kernel_inference import encoder
from audit_visual_submission import main as independent_audit


def main(out, reuse_reference):
    out, reuse_reference = out.resolve(), reuse_reference.resolve()
    root = local_path('records_root')
    public = REPOSITORY / 'output/experiment_record' / out.relative_to(root)
    prior_public = REPOSITORY / 'output/experiment_record' / reuse_reference.relative_to(root)
    registration = read_json(out / 'registration.json')
    for filename, expected_sha in registration.get('inference_dependency_sha256', {}).items():
        assert digest(Path(__file__).parent / filename) == expected_sha
    priorreg = read_json(reuse_reference / 'registration.json')
    priorchecks = read_json(prior_public / 'checks.json')
    assert read_json(prior_public / 'independent_export_audit.json')['passed']
    assert digest(reuse_reference / 'predictions.csv') == priorchecks['csv_sha256']
    assert read_json(public / 'kernel_adapter_audit.json')['passed']
    assert read_json(out / 'prepared.json')['reference_replay_passed']
    assert digest(Path(__file__).with_name('visual_kernel_components.py')) == registration['component_adapter_sha256']
    assert not (out / 'predictions.csv').exists()
    for key, filename in [('train_csv_sha256', 'train.csv'), ('test_csv_sha256', 'test.csv')]:
        assert registration[key] == priorreg[key] == digest(local_path('data_root') / filename)
    assert registration['split_sha256'] == priorreg['split_sha256'] == digest(local_path('split_path'))
    source_sha = digest(__file__)
    (out / 'inference_source_at_launch.py').write_bytes(Path(__file__).read_bytes())
    registration.update(inference_source_sha256=source_sha, encoder_dispatch_source_sha256=digest(Path(__file__).with_name('visual_kernel_inference.py')),
        test_images_used=True, test_images_role='Inference only; shared signed features reused, new image tower frozen',
        scope='Generic visual-only candidate; explicitly mixed verified-cache reuse and fresh encoder inference')
    test = pd.read_csv(local_path('data_root') / 'test.csv', dtype={'imageid': str})
    assert len(test) == 3000 and test.imageid.is_unique
    test_manifest = {'imageids': test.imageid.tolist(),
        'file_sha256': [digest(local_path('data_root') / 'test' / name) for name in test.imageid], 'test_labels_used': False}
    assert test_manifest == read_json(reuse_reference / 'test_manifest.json')
    write_json(out / 'test_manifest.json', test_manifest)
    test_sha = digest(out / 'test_manifest.json')
    assert test_sha == priorchecks['test_manifest_sha256']
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
    indices = np.random.default_rng(2026).choice(8000, 8, replace=False)
    def images(folder, names):
        result = []
        for name in names:
            with Image.open(local_path('data_root') / folder / name) as source:
                result.append(ImageOps.exif_transpose(source).convert('RGB'))
        return result
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    probes, outputs, spotchecks, evidence = images('train', frame.imageid.iloc[indices]), [], {}, []
    def status(state, **extra):
        write_json(public / 'status.json', {'status': state, 'pid': os.getpid(), **extra})
    def save_registration():
        write_json(out / 'registration.json', registration)
        write_json(public / 'registration.json', registration)
    save_registration()
    try:
        for i, component in enumerate(registration['components']):
            family, key = component['cache_family'], component['cache_key']
            source_path = REPOSITORY / 'output/experiment_record' / component['source'] / 'registration.json'
            assert digest(source_path) == component['source_registration_sha256']
            feature_source = feature_registration(read_json(source_path))
            manifest_path = local_path('cache_root') / family / key / 'manifest.json'
            manifest = read_json(manifest_path)
            assert digest(manifest_path) == component['feature_manifest_sha256'] == feature_source['manifest_sha256']
            batch = int(manifest.get('batch_size', 4))
            matches = [c for c in priorreg['components'] if c['cache_family'] == family and c['cache_key'] == key]
            encode = None
            if matches:
                assert len(matches) == 1 and matches[0]['feature_manifest_sha256'] == component['feature_manifest_sha256']
                assert priorchecks['labeled_feature_spotchecks'][family + ':' + key]
                component['feature_inference_source_sha256'] = matches[0].get('feature_inference_source_sha256', priorreg['inference_source_sha256'])
                component['feature_cache_reuse'] = {'reference': str(reuse_reference.relative_to(root)),
                    'checks_sha256': digest(prior_public / 'checks.json'), 'audit_sha256': digest(prior_public / 'independent_export_audit.json'),
                    'registration_sha256': digest(prior_public / 'registration.json')}
                evidence.append({'family': family, 'cache_key': key, 'fresh_runtime_probe': False,
                    'inherited_probe_reference': str(reuse_reference.relative_to(root))})
            else:
                contracts = {
                    'pe_core_giant_features': ('pe_core_g14_448/20261014T065137Z_b822c3d4', 'pe_core_giant_encoder.py'),
                    'pe_core_giant_whole_features': ('pe_core_g14_448_whole_view/20261014T082013Z_1058c142', 'pe_core_giant_whole_encoder.py'),
                }
                if family not in contracts:
                    raise ValueError('Only explicitly registered and audited new PE-Core-G features are allowed')
                audit_reference, adapter_filename = contracts[family]
                audit_path = REPOSITORY / 'output/experiment_record/19_image_embedding_reference' / audit_reference / 'encoder_adapter_gpu_audit.json'
                gpu_audit = read_json(audit_path)
                assert gpu_audit['passed'] and gpu_audit['feature_manifest_sha256'] == digest(manifest_path)
                assert gpu_audit['encoder_adapter_source_sha256'] == digest(Path(__file__).with_name(adapter_filename))
                encode = encoder(family, manifest)
                actual = encode(probes)
                expected = features_for(feature_source, frame)
                np.testing.assert_allclose(actual, expected[indices], rtol=1e-4, atol=3e-6)
                probe_error = float(np.max(abs(actual - expected[indices])))
                del expected
                component['feature_inference_source_sha256'] = source_sha
                evidence.append({'family': family, 'cache_key': key, 'fresh_runtime_probe': True,
                    'max_abs_training_probe_error': probe_error})
            save_registration()
            contract = {'family': family, 'source_cache_key': key, 'source_manifest_sha256': digest(manifest_path),
                'test_manifest_sha256': test_sha, 'inference_source_sha256': component['feature_inference_source_sha256'],
                'feature_dim': component['feature_dim'], 'batch_size': batch, 'test_labels_used': False}
            cachekey = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
            cache = local_path('cache_root') / 'test_visual_features' / cachekey
            if encode is None:
                assert read_json(cache / 'manifest.json') == contract
            else:
                cache.mkdir(parents=True, exist_ok=False)
                write_json(cache / 'manifest.json', contract)
            checksum_path = cache / 'chunk_sha256.json'
            checksums = read_json(checksum_path) if checksum_path.exists() else {}
            chunks = []
            for offset in range(0, 3000, batch):
                names = test.imageid.iloc[offset:offset + batch]
                path = cache / f'{offset:05d}_{len(names)}.npy'
                if encode is None:
                    assert digest(path) == checksums[path.name]
                    value = np.load(path, allow_pickle=False)
                else:
                    value = encode(images('test', names))
                    with path.open('xb') as stream:
                        np.save(stream, value)
                    checksums[path.name] = digest(path)
                    if offset % 100 == 0:
                        write_json(checksum_path, checksums)
                assert value.shape == (len(names), component['feature_dim']) and np.isfinite(value).all()
                np.testing.assert_allclose(np.linalg.norm(value, axis=1), np.ones(len(names)), rtol=0, atol=1e-8)
                chunks.append(value)
                if offset % 100 == 0:
                    status('inferring_test', component=i, family=family, images=offset + len(names), total=3000)
                    print(f'Component {i + 1}: {family} {offset + len(names)}/3000', flush=True)
            if encode is not None:
                write_json(checksum_path, checksums)
                del encode
                torch.cuda.empty_cache()
            outputs.append(predict_component(out, i, component, np.concatenate(chunks)))
            spotchecks[family + ':' + key] = True
        prediction = np.column_stack(outputs) @ np.asarray(registration['weights'])
        assert prediction.shape == (3000,) and np.isfinite(prediction).all() and (prediction >= 0).all()
        pd.DataFrame({'imageid': test.imageid, 'price': prediction}).to_csv(out / 'predictions.csv', index=False, mode='x')
        write_json(public / 'checks.json', {'n': 3000, 'finite_nonnegative': True, 'test_order_matches': True,
            'csv_sha256': digest(out / 'predictions.csv'), 'test_manifest_sha256': test_sha,
            'labeled_feature_spotchecks': spotchecks, 'labeled_feature_spotchecks_evidence': evidence,
            'test_labels_used': False, 'mllm_components': False, 'kaggle_uploaded': False, 'price_unit': 'thousand USD'})
        status('completed', test_inference_completed=True)
        independent_audit(out)
        print('AUDITED_CSV', out / 'predictions.csv', flush=True)
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared-run', type=Path, required=True)
    parser.add_argument('--reuse-reference-run', type=Path, required=True)
    args = parser.parse_args()
    main(args.prepared_run, args.reuse_reference_run)
