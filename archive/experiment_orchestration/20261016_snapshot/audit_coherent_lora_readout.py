"""Independent CPU bank-membership, equation, prediction and fixed-blend audit."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
from common.local_paths import local_path
from common.config import read_json,write_json,digest

RUN=Path('20_price_lora/pe_giant_lora_coherent_readout/20261014T144957Z_d58a2814')
public=ROOT/'output/experiment_record'/RUN
local=local_path('records_root')/RUN
assert read_json(public/'status.json')['status']=='completed'
reg=read_json(public/'registration.json')
binding=read_json(public/'execution_binding.json')
result=read_json(public/'results.json')
assert digest(public/'registration.json')==binding['registration_sha256']
assert digest(ROOT/reg['execution_source_path'])==binding['source_sha256']
assert digest(local/'source.py')==binding['source_sha256']
for name,sha in binding['dependency_sha256'].items():
    assert digest(ROOT/'scripts/experiments'/name)==sha
    assert digest(local/'dependency_snapshots'/name)==sha
assert digest(local/'source_adapter.safetensors')==binding['adapter_sha256']==result['adapter_sha256']
assert digest(local_path('split_path'))==reg['split_sha256']
frame=pd.read_csv(local_path('split_path'),dtype={'imageid':str,'group_id':str})
train,valid=[frame.partition.eq(p).to_numpy() for p in ['train','validation']]
assert train.sum()==6399 and valid.sum()==1601 and len(frame)==8000
assert not set(frame.loc[train,'group_id'])&set(frame.loc[valid,'group_id'])
manifest=read_json(public/'feature_manifest.json')
cache=local_path('cache_root')/reg['feature_family']/manifest['cache_key']
actual_manifest=read_json(cache/'manifest.json')
assert manifest=={**actual_manifest,'cache_key':manifest['cache_key']}
assert manifest['imageids']==frame.imageid.tolist() and manifest['pixel_hashes']==frame.pixel_hash.tolist()
assert manifest['supervised_encoder'] is True and manifest['new_model_oof_generated'] is False
assert manifest['adapter_sha256']==binding['adapter_sha256']
assert manifest['execution_binding_sha256']==digest(public/'execution_binding.json')
checks=read_json(cache/'chunk_sha256.json')
assert len(checks)==80
parts=[]
for start in range(0,8000,100):
    name=f'{start:05d}_100.npy'
    assert digest(cache/name)==checks[name]
    part=np.load(cache/name,allow_pickle=False)
    assert part.shape==(100,2560) and part.dtype==np.float64 and np.isfinite(part).all()
    np.testing.assert_allclose(np.linalg.norm(part,axis=1),1,rtol=1e-10,atol=1e-12)
    parts.append(part)
x=np.concatenate(parts)
del parts
assert digest(local/'head.npz')==result['head_sha256']
with np.load(local/'head.npz',allow_pickle=False) as head:
    anchors,c=head['train_features'].copy(),head['coefficients'].copy()
    gamma,alpha,mean=[float(head[k]) for k in ['gamma','alpha','target_mean']]
assert gamma==reg['gamma']==4 and alpha==reg['alpha']==1
assert anchors.shape==(6399,2560) and c.shape==(6399,) and np.isfinite(c).all()
np.testing.assert_array_equal(anchors,x[train])
ytrain=frame.loc[train,'price'].to_numpy()
np.testing.assert_allclose(mean,np.mean(ytrain/1000),rtol=1e-12)

def kernel_prices(query):
    predictions=[]
    for i in range(0,len(query),128):
        distance=np.maximum(2-2*(query[i:i+128]@anchors.T),0)
        predictions.extend(((np.exp(-gamma*distance)@c+mean)*1000).tolist())
    return np.asarray(predictions)

with threadpool_limits(limits=2):
    train_predictions=kernel_prices(anchors)
    expected=ytrain-1000*alpha*c
    np.testing.assert_allclose(train_predictions,expected,rtol=1e-7,atol=1e-4)
    query=kernel_prices(x[valid])
branch=pd.read_csv(local/'coherent_branch/predictions.csv',dtype={'imageid':str,'group_id':str})
final=pd.read_csv(local/'predictions.csv',dtype={'imageid':str,'group_id':str})
for table in [branch,final]:
    assert table.imageid.tolist()==frame.loc[valid,'imageid'].tolist()
    assert table.group_id.tolist()==frame.loc[valid,'group_id'].tolist()
    np.testing.assert_array_equal(table.price.to_numpy(),frame.loc[valid,'price'].to_numpy())
    assert np.isfinite(table.prediction).all() and (table.prediction>=0).all()
np.testing.assert_allclose(query,branch.raw_prediction.to_numpy(),rtol=1e-10,atol=1e-7)
np.testing.assert_allclose(np.maximum(query,0),branch.prediction.to_numpy(),rtol=1e-10,atol=1e-7)
baseline_root=local_path('records_root')/reg['baseline_fusion']
assert digest(baseline_root/'ensemble.json')==reg['baseline_ensemble_sha256']
ensemble=read_json(baseline_root/'ensemble.json')
own=[i for i,r in enumerate(ensemble['models']) if Path(r)==Path(reg['original_teacher_kernel'])]
assert len(own)==1
own=own[0]
other=np.zeros(1601)
for i,(run,w) in enumerate(zip(ensemble['models'],ensemble['weights'])):
    if not w or i==own: continue
    file=local_path('records_root')/run/'predictions.csv'
    assert digest(file)==ensemble['prediction_sha256'][i]
    t=pd.read_csv(file,dtype={'imageid':str})
    assert t.imageid.tolist()==frame.loc[valid,'imageid'].tolist()
    other+=w*t.prediction.to_numpy()
p=other+ensemble['weights'][own]*np.maximum(query,0)
np.testing.assert_allclose(p,final.prediction.to_numpy(),rtol=1e-10,atol=1e-7)
y=frame.loc[valid,'price'].to_numpy()
error=p-y
np.testing.assert_allclose(final.raw_prediction.to_numpy(),p,rtol=1e-10,atol=1e-7)
for table in [branch,final]:
    residual=table.prediction.to_numpy()-y
    np.testing.assert_allclose(table.error.to_numpy(),residual,rtol=1e-10,atol=1e-8)
    np.testing.assert_allclose(table.squared_error.to_numpy(),residual**2,rtol=1e-10,atol=1e-6)
branch_error=np.maximum(query,0)-y
for key,value in {'n':1601,'mse':float(np.mean(branch_error**2)),
    'mae':float(np.mean(abs(branch_error))), 'tail_mse':float(np.mean(branch_error[y>1300]**2))}.items():
    np.testing.assert_allclose(value,result['branch_metrics'][key],rtol=1e-10,atol=1e-7)
metrics={'n':1601,'mse':float(np.mean(error**2)),'mae':float(np.mean(abs(error))),
    'bias':float(error.mean()),'tail_mse':float(np.mean(error[y>1300]**2))}
for key in ['mse','mae','tail_mse','n']:
    np.testing.assert_allclose(metrics[key],result[key],rtol=1e-10,atol=1e-7)
base=pd.read_csv(baseline_root/'predictions.csv',dtype={'imageid':str})
assert base.imageid.tolist()==final.imageid.tolist()
b=base.prediction.to_numpy()
baseline_mse=float(np.mean((b-y)**2))
np.testing.assert_allclose(baseline_mse,result['baseline_validation_mse'],rtol=1e-10)
assert result['screen_passed']==(metrics['mse']<baseline_mse*.99)
slices=[]
for name,mask in [('le300',y<=300),('300_700',(y>300)&(y<=700)),('700_1300',(y>700)&(y<=1300)),('gt1300',y>1300)]:
    bmse,cmse=[float(np.mean((v[mask]-y[mask])**2)) for v in [b,p]]
    slices.append({'band':name,'n':int(mask.sum()),'baseline_mse':bmse,'candidate_mse':cmse,
        'baseline_bias':float(np.mean(b[mask]-y[mask])),'candidate_bias':float(np.mean(p[mask]-y[mask])),
        'overall_mse_change_contribution':float(np.sum((p[mask]-y[mask])**2-(b[mask]-y[mask])**2)/len(y))})
original_control=read_json(public/'original_head_control.json')
source_replay=read_json(public/'fixed_epoch2_price_replay.json')
assert original_control['passed'] and original_control['validation_n']==1601
assert source_replay['passed'] and source_replay['validation_n']==1601
assert source_replay['all8000_pixel_hashes_verified'] and source_replay['all_feature_chunks_verified']
audit={'passed':True,'scope':'Independent CPU all6399 bank membership/equation and all1601 branch/fixed-blend prices; hashes and cached norms. GPU and raw-pixel evidence referenced from producer checks, not independently repeated.',
    'source_sha256':digest(__file__),'source_results_sha256':digest(public/'results.json'),
    'head_sha256':digest(local/'head.npz'),'feature_manifest_sha256':digest(cache/'manifest.json'),
    'feature_chunk_manifest_sha256':digest(cache/'chunk_sha256.json'),
    'branch_predictions_sha256':digest(local/'coherent_branch/predictions.csv'),
    'final_predictions_sha256':digest(local/'predictions.csv'),
    'all6399_training_equation_max_price_error':float(abs(train_predictions-expected).max()),
    'all1601_query_replay_max_price_error':float(abs(query-branch.raw_prediction.to_numpy()).max()),
    'all1601_blend_replay_max_price_error':float(abs(p-final.prediction.to_numpy()).max()),
    'metrics':metrics,'price_slices':slices,'screen_gate_verified':True,'screen_gate_passed':result['screen_passed'],
    'original_control_sha256':digest(public/'original_head_control.json'),
    'producer_epoch2_replay_sha256':digest(public/'fixed_epoch2_price_replay.json'),
    'new_model_oof_generated':False,'three_seed_adoption_completed':False,'export_eligible':False}
write_json(public/'independent_artifact_audit.json',audit)
print({'passed':True,'metrics':metrics,'screen_gate_verified':result['screen_passed']})
