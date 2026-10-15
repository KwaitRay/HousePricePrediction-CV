"""Record user-requested stopped state; no model loading or restart."""
from pathlib import Path
import sys,json,hashlib,datetime
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('19_image_embedding_reference/pe_giant_mid40/20261015T035803Z_f0de64dc')
public=ROOT/'output/experiment_record'/RUN
reg=json.loads((public/'registration.json').read_text(encoding='utf-8'))
cache=ROOT/'.local/cache'/reg['cache_family']/reg['cache_key']
checks=json.loads((cache/'chunk_sha256.json').read_text(encoding='utf-8'))
for name,sha in checks.items():assert hashlib.sha256((cache/name).read_bytes()).hexdigest()==sha
complete=len(checks)*100
for name in ['status.json','pipeline_status.json']:
    original=public/('before_user_pause_'+name)
    assert not original.exists();original.write_bytes((public/name).read_bytes())
record={'status':'paused_by_user','reason':'User requested GPU for other work; no automatic restart',
    'requested_date':'2026-10-15','recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'stopped_pids':[46944,34456,4672],'verified_complete_chunks':len(checks),'verified_saved_images':complete,
    'total_images':8000,'cache_key':reg['cache_key'],'all_saved_chunk_hashes_verified':True,
    'latest_user_reported_public_mse':47291.34608,'latest_feedback_run':'20261015T034010Z_0c94a543',
    'latest_feedback_csv_sha256':'c39009ec65fd89a0f531991d1cfc26a2ecf4095e512ce9905b89c38da06b8da3',
    'resume_requires_user_instruction':True,'resume_implementation_not_yet_added':True}
for name in ['user_pause.json','status.json','pipeline_status.json']:
    (public/name).write_text(json.dumps(record,indent=2),encoding='utf-8')
for relative in ['design/model_and_experiment_plan.md','output/experiment_record/improvement_to_50000/最新进展与接续.md']:
    path=ROOT/relative;s=path.read_text(encoding='utf-8')
    note=f'**用户已要求暂停实验以使用GPU（2026-10-15）：所有实验进程已停止，任务状态paused，不自动恢复。第40块已核验保存{complete}/8000图、{len(checks)}块，详见运行user_pause.json；恢复需要明确指令及续跑适配。最新用户公榜反馈为034010：47291.34608（SHA开头c39009ec），低于40000仍未达到。下方运行中/待反馈描述为暂停前快照。**\n\n'
    path.write_text(note+s,encoding='utf-8')
print(json.dumps(record))
