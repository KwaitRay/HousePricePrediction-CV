"""CPU-only independent artifact checks; never modifies the running trainer."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
from common.local_paths import local_path

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def metrics(y, p):
    e = p-y
    return {'n': int(len(y)), 'mse': float(np.mean(e**2)),
            'mae': float(np.mean(abs(e))), 'bias': float(np.mean(e))}

def main(relative):
    public = ROOT/'output/experiment_record'/relative
    local = local_path('records_root')/relative
    reg = read(public/'registration.json')
    expected = {'pe_lora_price_screen.py': reg['source_sha256'], **reg['dependency_sha256']}
    for name, digest in expected.items():
        assert sha(ROOT/'scripts/experiments'/name) == digest, name
        assert sha(local/name) == digest, ('snapshot', name)
    assert sha(local_path('split_path')) == reg['split_sha256']
    assert sha(local_path('records_root')/reg['source_kernel']/'head.npz') == reg['source_head_sha256']
    assert sha(local_path('records_root')/reg['baseline_fusion']/'ensemble.json') == reg['baseline_ensemble_sha256']
    assert sha(ROOT/'output/experiment_record'/reg['source_kernel']/'registration.json') == reg['source_feature_registration_sha256']
    assert sha(ROOT/'output/experiment_record'/reg['preflight_reference']/'results.json') == reg['preflight_results_sha256']
    frame = pd.read_csv(local_path('split_path'), dtype={'imageid':str,'group_id':str})
    valid = frame.loc[frame.partition.eq('validation')].reset_index(drop=True)
    train = frame.loc[frame.partition.eq('train')]
    assert len(valid)==1601 and len(train)==6399
    assert not set(valid.group_id)&set(train.group_id)
    baseline = pd.read_csv(local_path('records_root')/reg['baseline_fusion']/'predictions.csv',dtype={'imageid':str,'group_id':str})
    assert baseline.imageid.tolist()==valid.imageid.tolist()
    y = valid.price.to_numpy()
    base = baseline.prediction.to_numpy()
    baseline_metrics = metrics(y,base)
    control = read(public/'zero_lora_control.json')
    np.testing.assert_allclose(baseline_metrics['mse'], control['baseline_validation_mse'],rtol=1e-10)
    assert control['passed'] and control['cached_validation_prices_replayed']==1601
    history = read(public/'history.json') if (public/'history.json').exists() else []
    epochs = []
    for row in history:
        folder = local/f"epoch_{row['epoch']}"
        csv = folder/'predictions.csv'
        t = pd.read_csv(csv,dtype={'imageid':str,'group_id':str})
        assert t.imageid.tolist()==valid.imageid.tolist() and t.group_id.tolist()==valid.group_id.tolist()
        np.testing.assert_array_equal(t.price.to_numpy(),y)
        p = t.prediction.to_numpy()
        assert np.isfinite(p).all() and (p>=0).all()
        np.testing.assert_allclose(p,np.maximum(t.raw_prediction.to_numpy(),0),rtol=1e-12)
        np.testing.assert_allclose(t.error.to_numpy(),p-y,rtol=1e-10,atol=1e-9)
        np.testing.assert_allclose(t.squared_error.to_numpy(),(p-y)**2,rtol=1e-10,atol=1e-7)
        full = metrics(y,p)
        for key in ['mse','mae','n']:
            np.testing.assert_allclose(full[key],row[key],rtol=1e-10,atol=1e-7)
        adapter = folder/'adapter.safetensors'
        assert sha(adapter)==row['adapter_sha256']
        tensors = load_file(str(adapter))
        keys = {f'blocks.{i}.attn.{leaf}.{ab}' for i in [48,49] for leaf in ['qkv','proj'] for ab in ['a','b']}
        assert set(tensors)==keys and sum(x.size for x in tensors.values())==73728
        assert all(x.dtype==np.float32 and np.isfinite(x).all() for x in tensors.values())
        slices = {}
        for name,mask in [('low_le300',y<=300),('mid_300_700',(y>300)&(y<=700)),('mid_700_1300',(y>700)&(y<=1300)),('high_gt1300',y>1300)]:
            slices[name] = {'baseline':metrics(y[mask],base[mask]),'candidate':metrics(y[mask],p[mask])}
        np.testing.assert_allclose(slices['high_gt1300']['candidate']['mse'],row['tail_mse'],rtol=1e-10)
        epochs.append({'epoch':row['epoch'],'metrics':full,'baseline_minus_candidate_mse':baseline_metrics['mse']-full['mse'],
                       'price_slices':slices,'csv_sha256':sha(csv),'adapter_sha256':sha(adapter)})
    result = {'scope':'Frozen sources, snapshots, split, head, baseline and completed epoch artifacts; no GPU feature or price formula replay claimed',
              'source_integrity_passed':True,'baseline':baseline_metrics,'completed_epochs_audited':epochs,
              'training_status_observed':read(public/'status.json'),'three_seed_adoption_verified':False,'export_eligible':False}
    if (public/'results.json').exists():
        actual = read(public/'results.json')
        best = min([baseline_metrics['mse'],*[e['metrics']['mse'] for e in epochs]])
        assert len(epochs)==reg['epochs']
        np.testing.assert_allclose(best,actual['best_validation_mse'],rtol=1e-10)
        assert actual['screen_passed']==(best<baseline_metrics['mse']*.99)
        result['screen_gate_independently_verified']=actual['screen_passed']
    (public/'independent_artifact_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'source_integrity_passed':True,'completed_epochs_audited':len(epochs),'training_status':result['training_status_observed']},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='20_price_lora/pe_giant_query_lora/20261014T112652Z_0fdd4154')
    main(parser.parse_args().run)
