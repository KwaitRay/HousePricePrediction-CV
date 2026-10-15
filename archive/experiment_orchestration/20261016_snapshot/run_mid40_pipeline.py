"""One registered sequence: extract, fresh audit, train-only KRR, fixed incremental fusion."""
from pathlib import Path
import sys,subprocess,os
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
from common.config import read_json,write_json,digest
from common.local_paths import local_path
RUN=Path('19_image_embedding_reference/pe_giant_mid40/20261015T035803Z_f0de64dc')
out=local_path('records_root')/RUN;pub=ROOT/'output/experiment_record'/RUN
assert read_json(pub/'status.json')['status']=='preregistered'
assert not (pub/'pipeline_binding.json').exists()
names=['pe_giant_mid40_reference.py','pe_giant_mid40_encoder.py','pe_core_giant_preflight.py',
       'pe_giant_whole_view.py','frozen_kernel_search.py','visual_kernel_submission.py',
       'pe_giant_incremental_fusion.py','encoder_oof_fusion.py','common/config.py','common/local_paths.py','common/metrics.py','preparation/split.py']
binding={'controller_source_sha256':digest(__file__),'dependency_sha256':{n:digest(ROOT/'scripts/experiments'/n) for n in names},
    'source_registration_sha256':digest(pub/'registration.json'),'automatic_export':False}
write_json(pub/'pipeline_binding.json',binding)
(out/'pipeline_source.py').write_bytes(Path(__file__).read_bytes())
for name in names:
    target=out/'pipeline_dependency_snapshots'/name;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes((ROOT/'scripts/experiments'/name).read_bytes())
def status(state,**extra):write_json(pub/'pipeline_status.json',{'status':state,'controller_pid':os.getpid(),**extra})
def check_sources():
    assert digest(__file__)==binding['controller_source_sha256']
    for n,sha in binding['dependency_sha256'].items():assert digest(ROOT/'scripts/experiments'/n)==sha
def command(phase,args):
    check_sources()
    child=subprocess.Popen([sys.executable,'-X','utf8',*args],cwd=ROOT)
    status('running',phase=phase,child_pid=child.pid)
    code=child.wait()
    if code:raise RuntimeError(f'{phase} exited {code}; no automatic restart')
try:
    from pe_giant_incremental_fusion import preregister,run
    from encoder_oof_fusion import resolve_encoder
    reg=read_json(pub/'registration.json')
    spec={'name':'pe_giant_mid40_kernel','family':'pe_giant_mid40_features','model':reg['model'],'revision':reg['revision']}
    fusion=preregister(reg['baseline_fusion'],spec,'pe_giant_mid40_incremental_fusion')
    write_json(pub/'planned_fusion.json',{'protocol_dir':str(fusion),'spec':spec})
    command('extract',['scripts/experiments/pe_giant_mid40_reference.py','--phase','extract','--protocol-dir',str(out)])
    assert read_json(pub/'status.json')['status']=='features_extracted'
    command('fresh_gpu_audit',['scripts/experiments/pe_giant_mid40_reference.py','--phase','audit','--protocol-dir',str(out)])
    assert read_json(pub/'encoder_adapter_gpu_audit.json')['passed']
    reg=read_json(pub/'registration.json')
    command('grouped_kernel',['scripts/experiments/frozen_kernel_search.py','--cache-key',reg['cache_key'],
        '--cache-family',spec['family'],'--name',spec['name']])
    kernel=resolve_encoder(spec)
    check_sources();status('running',phase='fixed_incremental_fusion',kernel=kernel)
    run(fusion)
    result=read_json(ROOT/'output/experiment_record'/fusion.relative_to(local_path('records_root'))/'results.json')
    status('completed',kernel=kernel,fusion=str(fusion.relative_to(local_path('records_root'))),
        fusion_export_gate_passed=result['export_eligible'],automatic_test_export=False)
    print({'completed':True,'kernel':kernel,'fusion':str(fusion),'results':result},flush=True)
except BaseException as exc:
    status('failed',error=repr(exc));raise
