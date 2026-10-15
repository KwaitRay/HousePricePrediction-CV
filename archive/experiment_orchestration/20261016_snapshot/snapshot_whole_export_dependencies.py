import sys
from pathlib import Path
sys.path.insert(0, str(Path('scripts/experiments').resolve()))
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY

relative = Path('08_final/visual_kernel_ensemble/20261014T091115Z_286fc436')
out = local_path('records_root') / relative
public = REPOSITORY / 'output/experiment_record' / relative
files = ['visual_kernel_inference.py', 'pe_core_giant_whole_encoder.py', 'pe_giant_whole_view.py',
    'pe_core_giant_encoder.py', 'pe_core_giant_preflight.py', 'audit_visual_submission.py',
    'infer_mixed_verified_features.py']
registration = read_json(out / 'registration.json')
assert read_json(public / 'kernel_adapter_audit.json')['passed']
assert not (out / 'predictions.csv').exists()
hashes = {}
for name in files:
    source = Path('scripts/experiments') / name
    destination = out / ('verified_dependency_' + name)
    with destination.open('xb') as stream:
        stream.write(source.read_bytes())
    hashes[name] = digest(source)
    assert digest(destination) == hashes[name]
registration['inference_dependency_sha256'] = hashes
write_json(out / 'registration.json', registration)
write_json(public / 'registration.json', registration)
write_json(public / 'dependency_snapshots.json', {'files_sha256': hashes,
    'verified_before_test_inference': True, 'preserve_frozen_dependencies': True})
print('Frozen inference dependencies snapshotted:', len(hashes))
