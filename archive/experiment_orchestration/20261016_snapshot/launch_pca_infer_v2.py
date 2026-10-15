"""Bind all inference sources and reuse only the independently verified 091115 test features."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
from common.config import read_json,write_json,digest
from common.local_paths import local_path
from infer_mixed_verified_features import main
reference=ROOT/'output/experiment_record/15_fixed_ensembles/pe_whole_pca_incremental_fusion/20261015T033810Z_8aa6ec62/prepared_export.json'
prepared=read_json(reference);assert prepared['full_refit_audit_passed'] and not prepared['test_inference_completed']
out=Path(prepared['prepared_run'])
public=ROOT/'output/experiment_record'/out.relative_to(local_path('records_root'))
assert read_json(public/'status.json')['status']=='prepared'
reg=read_json(out/'registration.json')
assert reg['component_adapter_sha256']==digest(ROOT/'scripts/experiments/visual_kernel_components.py')
names=['infer_mixed_verified_features.py','visual_kernel_components.py','pca_kernel_component.py',
    'pe_whole_pca_bandwidth_kernel.py','visual_kernel_submission.py','visual_kernel_inference.py',
    'audit_visual_submission.py','common/config.py','common/local_paths.py']
bindings={name:digest(ROOT/'scripts/experiments'/name) for name in names}
snapshot=out/'inference_dependency_snapshots_v2';snapshot.mkdir(exist_ok=False)
for name in names:
    target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes((ROOT/'scripts/experiments'/name).read_bytes())
reg['inference_dependency_sha256']=bindings
write_json(out/'registration.json',reg);write_json(public/'registration.json',reg)
main(out,local_path('records_root')/'08_final/visual_kernel_ensemble/20261014T091115Z_286fc436')
