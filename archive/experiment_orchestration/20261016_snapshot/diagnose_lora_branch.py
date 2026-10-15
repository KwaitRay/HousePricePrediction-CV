"""Recover clipped query-branch prices from the fixed blend, without refitting."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
from common.local_paths import local_path
from common.config import read_json,write_json,digest
RUN=Path('20_price_lora/pe_giant_query_lora/20261014T112652Z_0fdd4154')
public=ROOT/'output/experiment_record'/RUN
root=local_path('records_root')
reg=read_json(public/'registration.json')
assert digest(local_path('split_path'))==reg['split_sha256']
frame=pd.read_csv(local_path('split_path'),dtype={'imageid':str})
valid=frame.loc[frame.partition.eq('validation')].reset_index(drop=True)
y=valid.price.to_numpy()
ensemble_path=root/reg['baseline_fusion']/'ensemble.json'
assert digest(ensemble_path)==reg['baseline_ensemble_sha256']
ensemble=read_json(ensemble_path)
own=[i for i,r in enumerate(ensemble['models']) if Path(r)==Path(reg['source_kernel'])]
assert len(own)==1
own=own[0]
weight=ensemble['weights'][own]
assert weight>0 and len(valid)==1601
other=np.zeros(len(valid))
inputs={str(ensemble_path.relative_to(root)):digest(ensemble_path)}
for i,(run,w) in enumerate(zip(ensemble['models'],ensemble['weights'])):
    if not w or i==own: continue
    file=root/run/'predictions.csv'
    assert digest(file)==ensemble['prediction_sha256'][i]
    table=pd.read_csv(file,dtype={'imageid':str})
    assert table.imageid.tolist()==valid.imageid.tolist()
    np.testing.assert_array_equal(table.price.to_numpy(),y)
    other+=w*table.prediction.to_numpy()
    inputs[str(file.relative_to(root))]=digest(file)
teacher_file=root/reg['source_kernel']/'predictions.csv'
assert digest(teacher_file)==ensemble['prediction_sha256'][own]
teacher=pd.read_csv(teacher_file,dtype={'imageid':str})
assert teacher.imageid.tolist()==valid.imageid.tolist()
np.testing.assert_array_equal(teacher.price.to_numpy(),y)
old=teacher.prediction.to_numpy()
baseline_file=root/reg['baseline_fusion']/'predictions.csv'
baseline=pd.read_csv(baseline_file,dtype={'imageid':str})
assert baseline.imageid.tolist()==valid.imageid.tolist()
np.testing.assert_allclose(other+weight*old,baseline.prediction.to_numpy(),rtol=1e-7,atol=1e-4)
inputs[str(teacher_file.relative_to(root))]=digest(teacher_file)
inputs[str(baseline_file.relative_to(root))]=digest(baseline_file)

def metrics(p):
    e=p-y
    return {'n':len(y),'mse':float(np.mean(e**2)),'mae':float(np.mean(abs(e))),'bias':float(e.mean()),
            'tail_mse':float(np.mean(e[y>1300]**2)),'low_mse':float(np.mean(e[y<=300]**2))}

epochs=[]
for row in read_json(public/'history.json'):
    file=root/RUN/f"epoch_{row['epoch']}"/'predictions.csv'
    table=pd.read_csv(file,dtype={'imageid':str})
    assert table.imageid.tolist()==valid.imageid.tolist()
    np.testing.assert_array_equal(table.price.to_numpy(),y)
    np.testing.assert_allclose(table.raw_prediction.to_numpy(),table.prediction.to_numpy(),rtol=1e-12,atol=1e-8)
    query=(table.prediction.to_numpy()-other)/weight
    assert np.isfinite(query).all() and (query>=-1e-7).all()
    query=np.maximum(query,0)
    np.testing.assert_allclose(other+weight*query,table.prediction.to_numpy(),rtol=1e-12,atol=1e-8)
    new=metrics(query)
    epochs.append({'epoch':row['epoch'],'clipped_query_branch':new,'fixed_blend_mse':row['mse'],
        'query_branch_mse_change':new['mse']-metrics(old)['mse'],'csv_sha256':digest(file)})
result={'scope':'Algebraic reconstruction of clipped query-branch prices from the existing unchanged-weight blend; no new features, head fitting, meta weights or model OOF. Unclipped query prices cannot be recovered.',
    'source_sha256':digest(__file__),'split_sha256':reg['split_sha256'],'input_sha256':inputs,
    'query_weight':weight,'fixed_teacher_branch':metrics(old),'epochs':epochs,
    'test_images_used':False,'model_or_optimizer_changed':False,'export_eligible':False}
write_json(public/'query_branch_diagnostic.json',result)
print({'fixed_teacher_branch':result['fixed_teacher_branch'],'epochs':epochs})
