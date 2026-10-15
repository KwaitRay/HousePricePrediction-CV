"""Diagnose failed fixed-adapter replay; never fit or export a price model."""
from pathlib import Path
import sys, os, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
import numpy as np
import pandas as pd
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from safetensors.torch import load_file
from PIL import Image, ImageOps
from common.config import read_json,write_json,digest
from common.local_paths import local_path
from pe_lora_price_screen import QueryPrice
from threadpoolctl import threadpool_limits

RUN=Path('20_price_lora/pe_giant_lora_coherent_readout/20261014T144957Z_d58a2814')
public=ROOT/'output/experiment_record'/RUN
local=local_path('records_root')/RUN
reg=read_json(public/'registration.json')
binding=read_json(public/'execution_binding.json')
assert read_json(public/'status.json')['status']=='failed'
assert digest(local/'source_adapter.safetensors')==binding['adapter_sha256']
report=public/'reload_diagnostic'
assert not report.exists()
report.mkdir()
frame=pd.read_csv(local_path('split_path'),dtype={'imageid':str})
valid=np.flatnonzero(frame.partition.eq('validation').to_numpy())
train=np.flatnonzero(frame.partition.eq('train').to_numpy())
indices=[*train[:3],*valid[[944,1000,1143,1399]],*valid[:2]]
registration={'source_sha256':digest(__file__),'adapter_sha256':binding['adapter_sha256'],
    'upstream_failure_sha256':digest(public/'status.json'),'probe_imageids':frame.imageid.iloc[indices].tolist(),
    'probe_selection':'3 fixed first training + 4 known replay violations + 2 fixed first validation; diagnostic, no accuracy selection',
    'policies':['legacy_default','deterministic_math_tf32_off'], 'repeats':3,
    'feature_tolerance':{'rtol':1e-4,'atol':3e-6},'price_tolerance':{'rtol':1e-7,'atol':1e-4},
    'fit_price_model':False,'test_images_used':False,'export_eligible':False}
write_json(report/'registration.json',registration)
def status(state,**kw):write_json(report/'status.json',{'status':state,'pid':os.getpid(),**kw})
try:
    status('loading')
    manifest=read_json(public/'feature_manifest.json')
    cache=local_path('cache_root')/reg['feature_family']/manifest['cache_key']
    checks=read_json(cache/'chunk_sha256.json')
    assert len(checks)==80
    for name,sha in checks.items():assert digest(cache/name)==sha
    features=np.concatenate([np.load(cache/name,allow_pickle=False) for name in sorted(checks)])
    torch.set_num_threads(2)
    with np.load(local_path('records_root')/reg['original_teacher_kernel']/'head.npz') as head:
        model=QueryPrice(head)
    adapter=load_file(str(local/'source_adapter.safetensors'))
    params=dict(model.visual.named_parameters())
    with torch.no_grad():
        for name,value in adapter.items():params[name].copy_(value)
    model.visual.eval()
    previous=pd.read_csv(local_path('records_root')/reg['upstream']/'epoch_2/predictions.csv',dtype={'imageid':str})
    ensemble=read_json(local_path('records_root')/reg['baseline_fusion']/'ensemble.json')
    own=[i for i,r in enumerate(ensemble['models']) if Path(r)==Path(reg['original_teacher_kernel'])][0]
    other=np.zeros(1601)
    for i,(r,w) in enumerate(zip(ensemble['models'],ensemble['weights'])):
        if w and i!=own:other+=w*pd.read_csv(local_path('records_root')/r/'predictions.csv').prediction.to_numpy()
    results=[]
    for policy in registration['policies']:
        if policy=='deterministic_math_tf32_off':
            os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False
            torch.use_deterministic_algorithms(True)
        for repeat in range(3):
            status('probing',policy=policy,repeat=repeat)
            for idx in indices:
                with Image.open(local_path('data_root')/'train'/frame.imageid.iloc[idx]) as image:
                    rgb=ImageOps.exif_transpose(image).convert('RGB')
                with torch.no_grad():
                    if policy=='legacy_default':value=model.query(rgb)
                    else:
                        with sdpa_kernel(SDPBackend.MATH):value=model.query(rgb)
                    price=float(model.price(value).item())
                v=value.cpu().numpy().reshape(-1)
                record={'policy':policy,'repeat':repeat,'index':int(idx),'imageid':frame.imageid.iloc[idx],
                    'feature_max_abs_error_vs_cache':float(abs(v-features[idx]).max()),
                    'feature_matches_cache':bool(np.allclose(v,features[idx],rtol=1e-4,atol=3e-6)), 'raw_branch_price':price}
                matches=np.flatnonzero(valid==idx)
                if len(matches):
                    j=int(matches[0]);p=other[j]+ensemble['weights'][own]*max(price,0)
                    record.update(old_fusion_price=float(previous.prediction.iloc[j]),replayed_fusion_price=p,
                        price_error=p-float(previous.prediction.iloc[j]))
                results.append(record)
                write_json(report/'probes.json',results)
                print(record,flush=True)
    write_json(report/'results.json',{'completed':True,'probes':results,'fit_price_model':False,
        'export_eligible':False,'note':'Diagnostic only; no repair, relaxed tolerance, or price effectiveness claim'})
    status('completed')
except BaseException as exc:
    status('failed',error=repr(exc));raise
