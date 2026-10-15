from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
import pandas as pd
import numpy as np
from common.local_paths import local_path
from common.config import write_json
root=local_path('records_root')
RUN='15_fixed_ensembles/pe_whole_pca_incremental_fusion/20261015T033810Z_8aa6ec62'
BASE='15_fixed_ensembles/pe_giant_whole_incremental_fusion/20261014T082727Z_c5152cef'
a=pd.read_csv(root/RUN/'predictions.csv',dtype={'imageid':str})
b=pd.read_csv(root/BASE/'predictions.csv',dtype={'imageid':str})
assert a.imageid.tolist()==b.imageid.tolist()
np.testing.assert_array_equal(a.price,b.price)
y=a.price.to_numpy();rows=[]
for name,mask in [('le300',y<=300),('300_700',(y>300)&(y<=700)),('700_1300',(y>700)&(y<=1300)),('gt1300',y>1300)]:
    r={'band':name,'n':int(mask.sum())}
    for label,t in [('baseline',b),('candidate',a)]:
        e=t.prediction.to_numpy()[mask]-y[mask]
        r[label+'_mse']=float(np.mean(e**2));r[label+'_bias']=float(e.mean())
    r['overall_mse_change_contribution']=float(mask.sum()/len(y)*(r['candidate_mse']-r['baseline_mse']))
    rows.append(r)
record={'scope':'Paired fixed validation price slices, no model fitting or subgroup selection','slices':rows,
    'low_mse_worsens_over5pct':rows[0]['candidate_mse']>rows[0]['baseline_mse']*1.05,
    'low_positive_bias_increases_over10pct':rows[0]['candidate_bias']>rows[0]['baseline_bias']*1.1}
write_json(ROOT/'output/experiment_record'/RUN/'paired_price_slices.json',record)
print(json.dumps(record))
