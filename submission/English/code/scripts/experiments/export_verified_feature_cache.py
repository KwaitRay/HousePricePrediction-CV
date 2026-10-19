"""Apply frozen new heads to immutable features from an audited earlier export.

No new encoder execution is claimed; runtime probe evidence is inherited explicitly.
The failed fresh-extraction run remains untouched.
"""
from pathlib import Path
import hashlib
import json
import shutil
import time
import uuid
import numpy as np
import pandas as pd
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from visual_kernel_components import predict_component
from audit_visual_submission import main as independent_audit

FAILED = '08_final/visual_kernel_ensemble/20261014T064058Z_16e1be61'
PRIOR = '08_final/visual_kernel_ensemble/20261014T044646Z_7d1af3eb'
EXPERIMENT = '15_fixed_ensembles/pe_pls_incremental_fusion/20261014T063915Z_573c1b3a'


def main():
    root = local_path('records_root')
    pubroot = REPOSITORY / 'output/experiment_record'
    old, prior = root / FAILED, root / PRIOR
    oldreg = read_json(old / 'registration.json')
    priorreg = read_json(prior / 'registration.json')
    checks = read_json(pubroot / PRIOR / 'checks.json')
    audit_file = pubroot / PRIOR / 'independent_export_audit.json'
    assert read_json(audit_file)['passed']
    assert digest(prior / 'predictions.csv') == checks['csv_sha256']
    assert read_json(old / 'prepared.json')['reference_replay_passed']
    assert read_json(pubroot / FAILED / 'kernel_adapter_audit.json')['passed']
    assert oldreg['split_sha256'] == priorreg['split_sha256'] == digest(local_path('split_path'))
    for key, filename in [('train_csv_sha256', 'train.csv'), ('test_csv_sha256', 'test.csv')]:
        assert oldreg[key] == priorreg[key] == digest(local_path('data_root') / filename)
    relative = Path('08_final/visual_kernel_ensemble') / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    out, public = root / relative, pubroot / relative
    out.mkdir(parents=True, exist_ok=False)
    registration = dict(oldreg)
    registration.update(inference_source_sha256=priorreg['inference_source_sha256'],
        cache_export_source_sha256=digest(__file__), full_refit_source=FAILED,
        verified_feature_export_source=PRIOR, prior_independent_audit_sha256=digest(audit_file),
        test_images_used=True, test_images_role='Reuse audited frozen test features for inference only',
        scope='Generic visual encoder candidate; cache-only export, no new runtime encoder probe claimed')
    for path in old.glob('component_*'):
        shutil.copyfile(path, out / path.name)
    for name in ('prepared.json', 'preparation_source.py', 'component_adapter_source.py'):
        shutil.copyfile(old / name, out / name)
    shutil.copyfile(prior / 'inference_source_at_launch.py', out / 'inference_source_at_launch.py')
    shutil.copyfile(prior / 'test_manifest.json', out / 'test_manifest.json')
    (out / 'cache_export_source.py').write_bytes(Path(__file__).read_bytes())
    write_json(out / 'registration.json', registration)
    write_json(public / 'registration.json', registration)
    write_json(public / 'status.json', {'status': 'replaying_verified_test_cache'})
    test = pd.read_csv(local_path('data_root') / 'test.csv', dtype={'imageid': str})
    test_manifest = read_json(out / 'test_manifest.json')
    test_sha = digest(out / 'test_manifest.json')
    assert test_sha == checks['test_manifest_sha256']
    assert test_manifest['imageids'] == test.imageid.tolist()
    for name, sha in zip(test.imageid, test_manifest['file_sha256']):
        assert digest(local_path('data_root') / 'test' / name) == sha
    outputs, reuse, probes = [], [], {}
    for i, component in enumerate(registration['components']):
        family, key = component['cache_family'], component['cache_key']
        matching = [c for c in priorreg['components'] if c['cache_family'] == family and c['cache_key'] == key]
        assert len(matching) == 1
        original = matching[0]
        assert original['feature_manifest_sha256'] == component['feature_manifest_sha256']
        manifest_path = local_path('cache_root') / family / key / 'manifest.json'
        manifest_sha = digest(manifest_path)
        assert manifest_sha == component['feature_manifest_sha256']
        assert digest(pubroot / component['source'] / 'registration.json') == component['source_registration_sha256']
        assert checks['labeled_feature_spotchecks'][family + ':' + key]
        batch = int(read_json(manifest_path).get('batch_size', 4))
        contract = {'family': family, 'source_cache_key': key, 'source_manifest_sha256': manifest_sha,
            'test_manifest_sha256': test_sha, 'inference_source_sha256': priorreg['inference_source_sha256'],
            'feature_dim': component['feature_dim'], 'batch_size': batch, 'test_labels_used': False}
        cachekey = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        cache = local_path('cache_root') / 'test_visual_features' / cachekey
        assert read_json(cache / 'manifest.json') == contract
        chunk_sha = read_json(cache / 'chunk_sha256.json')
        chunks = []
        for offset in range(0, 3000, batch):
            count = min(batch, 3000 - offset)
            path = cache / f'{offset:05d}_{count}.npy'
            assert digest(path) == chunk_sha[path.name]
            values = np.load(path, allow_pickle=False)
            assert values.shape == (count, component['feature_dim']) and np.isfinite(values).all()
            np.testing.assert_allclose(np.linalg.norm(values, axis=1), np.ones(count), rtol=0, atol=1e-8)
            chunks.append(values)
        outputs.append(predict_component(out, i, component, np.concatenate(chunks)))
        probes[family + ':' + key] = True
        reuse.append({'family': family, 'training_cache_key': key, 'test_cache_key': cachekey,
            'prior_export': PRIOR, 'prior_probe_evidence_sha256': digest(pubroot / PRIOR / 'checks.json'),
            'all_chunk_hashes_verified': True, 'fresh_runtime_probe': False})
        print(f'Frozen cache and new head replayed {i + 1}/3', flush=True)
    prediction = np.column_stack(outputs) @ np.asarray(registration['weights'])
    assert prediction.shape == (3000,) and np.isfinite(prediction).all() and (prediction >= 0).all()
    pd.DataFrame({'imageid': test.imageid, 'price': prediction}).to_csv(out / 'predictions.csv', index=False, mode='x')
    write_json(public / 'checks.json', {'n': 3000, 'finite_nonnegative': True, 'test_order_matches': True,
        'csv_sha256': digest(out / 'predictions.csv'), 'test_manifest_sha256': test_sha,
        'labeled_feature_spotchecks': probes, 'labeled_feature_spotchecks_evidence': reuse,
        'fresh_runtime_encoder_spotchecks': False, 'test_labels_used': False, 'mllm_components': False,
        'reference_replay_passed': True, 'kaggle_uploaded': False, 'price_unit': 'thousand USD'})
    write_json(public / 'status.json', {'status': 'completed', 'test_inference_completed': True})
    independent_audit(out)
    write_json(pubroot / EXPERIMENT / 'export_queue.json', {'status': 'completed', 'prepared_run': str(out),
        'independent_audit_passed': True, 'failed_fresh_extraction': FAILED, 'feature_reuse_source': PRIOR})
    (public / '实验报告.md').write_text('# PE PLS 三项融合缓存导出\n\n参考验证 MSE **56454.60**，公榜待反馈。\n\n全量 8000 标签重拟合三个冻结回归头，继承已成功审计的 044646 同源测试特征缓存，未声称重新编码成功。原图、缓存分块和来源哈希全部复核；新 PLS 距离变换由全量训练标签拟合。旧新 PLS 验证回放、全量核线性方程检查和 3000 条价格独立回放均通过。\n\n新鲜 NaFlex 编码抽检差异导致的失败保留于 064058，不放宽阈值。此次明确复用原有编码器输出，运行缓存导出脚本快照与原始特征推理源码分别保留。格式为 imageid,price，价格单位千美元；未上传 Kaggle。\n', encoding='utf-8')
    print('AUDITED_CSV', out / 'predictions.csv', flush=True)


if __name__ == '__main__':
    main()
