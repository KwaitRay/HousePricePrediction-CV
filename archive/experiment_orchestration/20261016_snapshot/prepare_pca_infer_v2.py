"""Preserve failed pre-inference launch binding and bind the narrow path-join fix."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
from common.config import write_json,read_json,digest
RUN=Path('08_final/visual_kernel_ensemble/20261015T034010Z_0c94a543')
out=ROOT/'.local/experiment_record'/RUN;pub=ROOT/'output/experiment_record'/RUN
assert read_json(pub/'status.json')['status']=='prepared' and not (out/'predictions.csv').exists()
original=out/'registration.json';saved=out/'inference_launch_attempt1_registration.json'
assert not saved.exists();saved.write_bytes(original.read_bytes())
write_json(pub/'inference_attempt1_failure.json',{'status':'failed_before_test_reading','session':70026,
    'error':"ValueError: Invalid name 'common/config.py'",'stage':'Dependency path digest before main inference try block',
    'previous_registration_sha256':digest(saved),'model_refit_repeated':False,'csv_generated':False,
    'fix':'Use Path(__file__).parent / relative_filename, supporting nested common source paths',
    'old_inference_snapshot_sha256':digest(out/'inference_dependency_snapshots/infer_mixed_verified_features.py'),
    'corrected_inference_source_sha256':digest(ROOT/'scripts/experiments/infer_mixed_verified_features.py')})
s=(ROOT/'.local/launch_pca_infer.py').read_text(encoding='utf-8')
assert s.count("out/'inference_dependency_snapshots'")==1
s=s.replace("out/'inference_dependency_snapshots'","out/'inference_dependency_snapshots_v2'")
target=ROOT/'.local/launch_pca_infer_v2.py';assert not target.exists()
target.write_text(s,encoding='utf-8');compile(s,str(target),'exec')
