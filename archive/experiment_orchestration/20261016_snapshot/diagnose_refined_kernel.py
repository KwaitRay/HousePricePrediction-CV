"""Retrospective paired price-bin audit; no fitting or model selection."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
from common.local_paths import local_path
from common.config import digest,read_json,write_json

RUN=Path('15_fixed_ensembles/pe_whole_refined_incremental_fusion/20261014T115345Z_60350891')
BASE=Path('15_fixed_ensembles/pe_giant_whole_incremental_fusion/20261014T082727Z_c5152cef')
root=local_path('records_root')
public=ROOT/'output/experiment_record'/RUN
assert read_json(public/'status.json')['status']=='completed'
frame=pd.read_csv(local_path('split_path'),dtype={'imageid':str,'group_id':str})
train,valid=[frame.loc[frame.partition.eq(p)].reset_index(drop=True) for p in ['train','validation']]
meta=pd.read_csv(root/RUN/'meta_crossfit_predictions.csv',dtype={'imageid':str})
oldmeta=pd.read_csv(root/BASE/'meta_crossfit_predictions.csv',dtype={'imageid':str})
previous=pd.read_csv(root/BASE/'predictions.csv',dtype={'imageid':str,'group_id':str})
current=pd.read_csv(root/RUN/'predictions.csv',dtype={'imageid':str,'group_id':str})
assert len(train)==6399 and len(valid)==1601
assert meta.imageid.tolist()==oldmeta.imageid.tolist()==train.imageid.tolist()
np.testing.assert_allclose(meta.baseline_meta_prediction,oldmeta.augmented_meta_prediction,rtol=1e-7,atol=1e-4)
assert previous.imageid.tolist()==current.imageid.tolist()==valid.imageid.tolist()
assert previous.group_id.tolist()==current.group_id.tolist()==valid.group_id.tolist()
np.testing.assert_array_equal(previous.price,valid.price)
np.testing.assert_array_equal(current.price,valid.price)

def compare(y,b,n):
    be,ne=b-y,n-y
    rows=[]
    masks=[('all',np.ones(len(y),bool)),('le300',y<=300),('300_700',(y>300)&(y<=700)),('700_1300',(y>700)&(y<=1300)),('gt1300',y>1300)]
    for name,mask in masks:
        rows.append({'band':name,'n':int(mask.sum()),'baseline_mse':float(np.mean(be[mask]**2)),
            'candidate_mse':float(np.mean(ne[mask]**2)),'baseline_bias':float(np.mean(be[mask])),
            'candidate_bias':float(np.mean(ne[mask])),'overall_mse_change_contribution':float(np.sum(ne[mask]**2-be[mask]**2)/len(y)),
            'fraction_absolute_errors_improved':float(np.mean(abs(ne[mask])<abs(be[mask])))})
    np.testing.assert_allclose(sum(r['overall_mse_change_contribution'] for r in rows[1:]),rows[0]['candidate_mse']-rows[0]['baseline_mse'],atol=1e-7)
    return rows

result={'scope':'Retrospective paired equal-weight original-price slices; existing selected models, no fit/calibration/test images',
    'source_sha256':digest(__file__),'split_sha256':digest(local_path('split_path')),
    'input_sha256':{str(p.relative_to(root)):digest(p) for p in [root/RUN/'meta_crossfit_predictions.csv',root/BASE/'meta_crossfit_predictions.csv',root/BASE/'predictions.csv',root/RUN/'predictions.csv']},
    'training_meta_crossfit':compare(train.price.to_numpy(),meta.baseline_meta_prediction.to_numpy(),meta.augmented_meta_prediction.to_numpy()),
    'development_validation':compare(valid.price.to_numpy(),previous.prediction.to_numpy(),current.prediction.to_numpy()),
    'test_images_used':False,'no_causal_or_unbiased_generalization_claim':True}
write_json(public/'paired_price_slice_diagnostic.json',result)
for scope in ['training_meta_crossfit','development_validation']:
    print(scope)
    for row in result[scope]: print(row)
