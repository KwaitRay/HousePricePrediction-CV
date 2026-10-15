"""Fixed teacher in-sample control, explicitly separate from OOF and final train scores."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
from common.local_paths import local_path
from common.config import read_json,write_json,digest
RUN=Path('20_price_lora/pe_giant_query_lora/20261014T112652Z_0fdd4154')
public=ROOT/'output/experiment_record'/RUN
reg=read_json(public/'registration.json')
assert digest(local_path('split_path'))==reg['split_sha256']
file=local_path('records_root')/reg['source_kernel']/'head.npz'
assert digest(file)==reg['source_head_sha256']
frame=pd.read_csv(local_path('split_path'),dtype={'imageid':str})
train=frame.loc[frame.partition.eq('train')].reset_index(drop=True)
y=train.price.to_numpy()
assert len(y)==6399
with np.load(file,allow_pickle=False) as head:
    x,c=head['train_features'],head['coefficients']
    alpha,gamma,mean=[float(head[k]) for k in ['alpha','gamma','target_mean']]
    assert x.shape==(6399,2560) and c.shape==(6399,) and alpha==1 and gamma==4
    np.testing.assert_allclose(y.mean()/1000,mean,rtol=1e-12)
    # Since (K+alpha*I)c=y/1000-mean, teacher Kc+mean = y/1000-alpha*c.
    raw=y-1000*alpha*c
    indices=np.sort(np.random.default_rng(2026).choice(len(y),12,replace=False))
    with threadpool_limits(limits=2):
        replay=(np.exp(-gamma*np.maximum(2-2*x[indices]@x.T,0))@c+mean)*1000
    np.testing.assert_allclose(replay,raw[indices],rtol=1e-7,atol=1e-4)
p=np.maximum(raw,0)
err=p-y
loss=float(np.mean(err**2)/1e6)
rows=[]
for name,mask in [('le300',y<=300),('300_700',(y>300)&(y<=700)),('700_1300',(y>700)&(y<=1300)),('gt1300',y>1300)]:
    rows.append({'band':name,'n':int(mask.sum()),'mse':float(np.mean(err[mask]**2)),'bias':float(np.mean(err[mask]))})
history=read_json(public/'history.json')
comparison=[{'epoch':r['epoch'],'mean_dynamic_training_price_loss':r['mean_training_price_loss'],
    'fixed_teacher_price_loss':loss,'dynamic_process_mean_minus_fixed_teacher':r['mean_training_price_loss']-loss,
    'relative_process_mean_change':r['mean_training_price_loss']/loss-1} for r in history]
result={'scope':'Fixed teacher prices on its own 6399 training anchors; in-sample diagnostic, not OOF or validation. Dynamic epoch process loss averages different adapter states and is not endpoint training MSE.',
    'source_sha256':digest(__file__),'head_sha256':digest(file),'split_sha256':digest(local_path('split_path')),
    'history_sha256_observed':digest(public/'history.json'),'equation_identity_probe_count':12,
    'equation_identity_max_error':float(np.max(abs(replay-raw[indices]))),
    'fixed_teacher_in_sample_mse':float(np.mean(err**2)),'fixed_teacher_normalized_price_loss':loss,
    'price_slices':rows,'process_comparison':comparison,'test_images_used':False,
    'model_or_optimizer_changed':False,'final_adapter_training_predictions_evaluated':False}
write_json(public/'teacher_in_sample_control_diagnostic.json',result)
print({k:result[k] for k in ['fixed_teacher_in_sample_mse','equation_identity_max_error','process_comparison']})
