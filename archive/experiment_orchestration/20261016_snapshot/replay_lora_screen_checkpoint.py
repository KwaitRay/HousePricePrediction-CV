"""Audit a completed LoRA screen; GPU replay only when the screen passes."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
from common.local_paths import local_path
from common.config import digest, read_json, write_json
from audit_lora_screen_result import main as artifact_audit
from pe_lora_price_screen import QueryPrice
from pe_core_giant_preflight import REVISION

RELATIVE = Path('20_price_lora/pe_giant_query_lora/20261014T112652Z_0fdd4154')

def main():
    public, local = ROOT/'output/experiment_record'/RELATIVE, local_path('records_root')/RELATIVE
    assert read_json(public/'status.json')['status']=='completed', 'Wait for actual training completion'
    artifact_audit(RELATIVE)
    result = read_json(public/'results.json')
    if not result['screen_passed']:
        write_json(public/'checkpoint_replay_audit.json',{'status':'not_required_screen_failed',
            'cpu_artifact_audit_completed':True,'gpu_replay_performed':False,'export_eligible':False})
        print('Screen failed; completed artifact audit, no further GPU replay',flush=True)
        return
    reg = read_json(public/'registration.json')
    selected = read_json(local/'selected.json')
    epoch = local/f"epoch_{selected['epoch']}"
    checkpoint = epoch/'adapter.safetensors'
    assert digest(checkpoint)==selected['adapter_sha256']
    weights = local_path('cache_root')/'pe_core_giant_weights'/REVISION
    provenance = read_json(weights/'download.json')
    assert digest(weights/'model.safetensors')==provenance['files']['model.safetensors']['sha256']
    assert provenance['files']['model.safetensors']['publisher_lfs_verified']
    frame = pd.read_csv(local_path('split_path'),dtype={'imageid':str,'group_id':str})
    valid = frame.loc[frame.partition.eq('validation')].reset_index(drop=True)
    saved = pd.read_csv(epoch/'predictions.csv',dtype={'imageid':str})
    assert saved.imageid.tolist()==valid.imageid.tolist()
    ensemble = read_json(local_path('records_root')/reg['baseline_fusion']/'ensemble.json')
    own = [i for i,m in enumerate(ensemble['models']) if Path(m)==Path(reg['source_kernel'])]
    assert len(own)==1
    own = own[0]
    other = np.zeros(len(valid))
    for i,(run,weight) in enumerate(zip(ensemble['models'],ensemble['weights'])):
        if not weight or i==own:
            continue
        file = local_path('records_root')/run/'predictions.csv'
        assert digest(file)==ensemble['prediction_sha256'][i]
        table = pd.read_csv(file,dtype={'imageid':str})
        assert table.imageid.tolist()==valid.imageid.tolist()
        other += weight*table.prediction.to_numpy()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    with np.load(local_path('records_root')/reg['source_kernel']/'head.npz',allow_pickle=False) as head:
        anchors, coefficients = head['train_features'].copy(),head['coefficients'].copy()
        gamma,mean = float(head['gamma']),float(head['target_mean'])
        model = QueryPrice(head)
    adapter = load_file(str(checkpoint),device='cpu')
    params = dict(model.visual.named_parameters())
    assert set(adapter)=={name for name,p in params.items() if p.requires_grad}
    with torch.no_grad():
        for name,value in adapter.items():
            assert value.shape==params[name].shape
            params[name].copy_(value)
    indices = np.sort(np.random.default_rng(2026).choice(len(valid),12,replace=False))
    checks=[]
    with torch.no_grad():
        for index in indices:
            name=valid.imageid.iloc[index]
            image_file=local_path('data_root')/'train'/name
            with Image.open(image_file) as image:
                image=ImageOps.exif_transpose(image).convert('RGB')
                query=model.query(image)
            q=query.cpu().numpy().reshape(-1)
            assert np.isfinite(q).all()
            np.testing.assert_allclose(np.linalg.norm(q),1,rtol=1e-10)
            cpu=float((np.exp(-gamma*np.maximum(2-2*anchors@q,0))@coefficients+mean)*1000)
            gpu=float(model.price(query).item())
            np.testing.assert_allclose(cpu,gpu,rtol=1e-10,atol=1e-7)
            blend=other[index]+ensemble['weights'][own]*max(cpu,0)
            np.testing.assert_allclose(blend,saved.prediction.iloc[index],rtol=1e-7,atol=1e-4)
            checks.append({'imageid':name,'provided_image_file_sha256':digest(image_file),
                'cpu_price':cpu,'gpu_price':gpu,'blend_saved_error':abs(blend-saved.prediction.iloc[index])})
    write_json(public/'checkpoint_replay_audit.json',{'status':'passed','probe_count':12,
        'scope':'Twelve fixed validation-image fresh GPU queries after adapter reload plus independent CPU RBF and blend price replay; not all 1601 fresh images',
        'adapter_sha256':digest(checkpoint),'source_sha256':digest(__file__),
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'probes':checks,
        'test_images_used':False,'three_seed_adoption_completed':False,'export_eligible':False})
    print('Selected checkpoint reload and 12-image price replay passed',flush=True)

if __name__=='__main__':
    main()
