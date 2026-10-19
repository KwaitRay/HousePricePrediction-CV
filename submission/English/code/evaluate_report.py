"""Recompute report metrics/selected-branch ablations from archived predictions."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts/experiments'))
from encoder_oof_fusion import fit_weights


def metrics(y,p):
    e=p-y
    return {'mse':float(np.mean(e**2)),'mae':float(np.mean(abs(e))),
            'rmse':float(np.sqrt(np.mean(e**2))),'bias':float(e.mean())}


def main():
    split=pd.read_csv(ROOT/'inputs/split_v1.csv')
    train=split[split.partition.eq('train')];valid=split[split.partition.eq('validation')]
    assert len(train)==6399 and len(valid)==1601
    assert not set(train.group_id)&set(valid.group_id)
    final=pd.read_csv(ROOT/'evidence/final_validation_predictions.csv')
    assert final.imageid.tolist()==valid.imageid.tolist()
    np.testing.assert_array_equal(final.price,valid.price)
    result={'validation':metrics(final.price.to_numpy(),final.prediction.to_numpy())}
    np.testing.assert_allclose(result['validation']['mse'],52201.21126766699,rtol=1e-12)
    meta=pd.read_csv(ROOT/'evidence/meta_crossfit_predictions.csv')
    assert meta.imageid.tolist()==train.imageid.tolist()
    result['meta_cv']={name:metrics(train.price.to_numpy(),meta[name].to_numpy()) for name in ['baseline_meta_prediction','augmented_meta_prediction']}
    ens=json.loads((ROOT/'evidence/final_fusion/ensemble.json').read_text(encoding='utf-8'))
    active=[(m,w) for m,w in zip(ens['models'],ens['weights']) if w>0]
    a,b=[],[]
    for model,w in active:
        folder=ROOT/'evidence/components'/model.replace('\\','/')
        t=pd.read_csv(folder/'selected_training_oof.csv');v=pd.read_csv(folder/'predictions.csv')
        assert t.imageid.tolist()==train.imageid.tolist() and v.imageid.tolist()==valid.imageid.tolist()
        a.append(t.oof_prediction.to_numpy()/1000);b.append(v.prediction.to_numpy())
    a=np.column_stack(a);b=np.column_stack(b)
    np.testing.assert_allclose(b@np.array([w for _,w in active]),final.prediction,atol=1e-6)
    result['selected_three_branch_ablations']={}
    for i,name in enumerate(['NaFlex','PE-L PLS','PE-G PCA']):
        keep=[j for j in range(3) if j!=i]
        weights=fit_weights(a[:,keep],train.price.to_numpy()/1000,1e-12)
        result['selected_three_branch_ablations']['without '+name]={**metrics(valid.price.to_numpy(),b[:,keep]@weights),'weights':weights.tolist()}
    result['selected_three_branch_ablations']['equal_weights']=metrics(valid.price.to_numpy(),b.mean(axis=1))
    out=ROOT/'checks';out.mkdir(exist_ok=True)
    (out/'report_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
