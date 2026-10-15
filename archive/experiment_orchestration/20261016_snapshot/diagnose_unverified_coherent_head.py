"""Provisional readout diagnostic on quarantined cache; never eligible for export."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from common.config import read_json,write_json,digest
from common.local_paths import local_path
from pe_lora_coherent_readout import fit_head,predict
RUN=Path('20_price_lora/pe_giant_lora_coherent_readout/20261014T144957Z_d58a2814')
public=ROOT/'output/experiment_record'/RUN
reg=read_json(public/'registration.json')
manifest=read_json(public/'feature_manifest.json')
cache=local_path('cache_root')/reg['feature_family']/manifest['cache_key']
assert read_json(public/'status.json')['status']=='failed'
registration={'source_sha256':digest(__file__),'failed_status_sha256':digest(public/'status.json'),
    'feature_manifest_sha256':digest(cache/'manifest.json'),'chunks_sha256':digest(cache/'chunk_sha256.json'),
    'purpose':'Resource prioritization only: investigate coherent head using numerically unverified cached features. This does not repair failed source replay or produce a candidate.',
    'gamma':4,'alpha':1,'train_n':6399,'validation_n':1601,'fixed_blend_weights':True,
    'new_model_oof_generated':False,'export_eligible':False,'screen_gate_applicable':False,
    'test_images_used':False,'tolerance_relaxed':False}
assert not (public/'unverified_readout_diagnostic_registration.json').exists()
write_json(public/'unverified_readout_diagnostic_registration.json',registration)
checks=read_json(cache/'chunk_sha256.json')
for name,sha in checks.items():assert digest(cache/name)==sha
x=np.concatenate([np.load(cache/name,allow_pickle=False) for name in sorted(checks)])
frame=pd.read_csv(local_path('split_path'),dtype={'imageid':str,'group_id':str})
train=frame.partition.eq('train').to_numpy();valid=frame.partition.eq('validation').to_numpy()
assert not set(frame.loc[train,'group_id'])&set(frame.loc[valid,'group_id'])
assert train.sum()==6399 and valid.sum()==1601 and manifest['imageids']==frame.imageid.tolist()
with threadpool_limits(limits=2):
    c,mean=fit_head(x[train],frame.loc[train,'price'].to_numpy(),4.,1.)
    q=np.maximum(predict(x[valid],x[train],c,mean,4.),0)
ensemble=read_json(local_path('records_root')/reg['baseline_fusion']/'ensemble.json')
p=np.zeros(1601)
for r,w,sha in zip(ensemble['models'],ensemble['weights'],ensemble['prediction_sha256']):
    if not w:continue
    if Path(r)==Path(reg['original_teacher_kernel']):p+=w*q
    else:
        file=local_path('records_root')/r/'predictions.csv';assert digest(file)==sha
        t=pd.read_csv(file,dtype={'imageid':str});assert t.imageid.tolist()==frame.loc[valid,'imageid'].tolist()
        p+=w*t.prediction.to_numpy()
y=frame.loc[valid,'price'].to_numpy()
result={'provisional_unverified_features':True,'not_a_candidate':True,
    'mse':float(np.mean((p-y)**2)),'branch_mse':float(np.mean((q-y)**2)),
    'mae':float(np.mean(abs(p-y))),'tail_mse':float(np.mean((p[y>1300]-y[y>1300])**2)),
    'baseline_mse':reg['baseline_validation_mse'],'export_eligible':False,'screen_passed':None,
    'source_replay_still_failed':True,'no_new_model_oof':True,
    'scope':'Descriptive diagnostic only. No successful numerical control, adoption, test export or source replay claim.'}
write_json(public/'unverified_readout_diagnostic.json',result)
print(result)
