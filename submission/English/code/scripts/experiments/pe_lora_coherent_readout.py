"""Refit a coherent RBF bank/head on a fixed supervised LoRA encoder; no OOF."""
import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file
from scipy.linalg import solve
from threadpoolctl import threadpool_limits
import timm.models.eva as eva_module
from common.config import digest, read_json, write_json
from common.local_paths import local_path, REPOSITORY
from common.metrics import save_predictions
from preparation.split import pixel_hash
from pe_lora_price_screen import QueryPrice
from pe_core_giant_preflight import REVISION
from visual_kernel_submission import features_for

def paths(out):
    out=out.resolve()
    return out,REPOSITORY/'output/experiment_record'/out.relative_to(local_path('records_root'))

def dependencies():
    folder=Path(__file__).parent
    names=['pe_lora_price_screen.py','pe_core_giant_preflight.py','pe_giant_lora_preflight.py',
           'pe_giant_whole_view.py','visual_kernel_submission.py','common/config.py',
           'common/local_paths.py','common/metrics.py','preparation/split.py']
    return {name:folder/name for name in names}

def bind(out):
    out,public=paths(out)
    reg=read_json(public/'registration.json')
    assert read_json(public/'status.json')['status']=='preregistered_conditional'
    assert str(Path(reg['execution_source_path']))==str(Path(__file__).relative_to(REPOSITORY))
    up=REPOSITORY/'output/experiment_record'/reg['upstream']
    assert read_json(up/'status.json')['status']=='completed'
    queue=read_json(up/'checkpoint_audit_command_queue.json')
    assert queue['status']=='completed' and queue['exit_code']==0
    result=read_json(up/'results.json')
    if result['screen_passed']:
        write_json(public/'status.json',{'status':'not_started_upstream_passed','reason':reg['start_condition']})
        print('Upstream passed: prefer its registered three-seed workflow',flush=True)
        return
    assert digest(up/'registration.json')==reg['upstream_registration_sha256']
    upstream_reg=read_json(up/'registration.json')
    assert digest(Path(__file__).with_name('pe_lora_price_screen.py'))==reg['upstream_source_sha256']
    for name,sha in reg['upstream_dependency_sha256'].items():
        assert digest(Path(__file__).with_name(name))==sha
    assert digest(local_path('split_path'))==reg['split_sha256']
    rows=[r for r in read_json(up/'history.json') if r['epoch']==reg['source_checkpoint_epoch']]
    assert len(rows)==1 and rows[0]['n']==1601
    source=local_path('records_root')/reg['upstream']/f"epoch_{reg['source_checkpoint_epoch']}"
    assert digest(source/'adapter.safetensors')==rows[0]['adapter_sha256']
    target=out/'source_adapter.safetensors'
    with target.open('xb') as f: f.write((source/'adapter.safetensors').read_bytes())
    weights=local_path('cache_root')/'pe_core_giant_weights'/REVISION
    provenance=read_json(weights/'download.json')
    assert provenance['files']['model.safetensors']['publisher_lfs_verified']
    assert digest(weights/'model.safetensors')==provenance['files']['model.safetensors']['sha256']
    binding={'registration_sha256':digest(public/'registration.json'),'source_sha256':digest(__file__),
        'dependency_sha256':{name:digest(path) for name,path in dependencies().items()},
        'eva_implementation_sha256':digest(eva_module.__file__),'adapter_sha256':digest(target),
        'upstream_epoch_predictions_sha256':digest(source/'predictions.csv'),
        'upstream_results_sha256':digest(up/'results.json'),
        'upstream_artifact_audit_sha256':digest(up/'independent_artifact_audit.json'),
        'source_feature_registration_sha256':upstream_reg['source_feature_registration_sha256'],
        'base_checkpoint_sha256':provenance['files']['model.safetensors']['sha256'],
        'config_sha256':digest(weights/'config.json')}
    write_json(public/'execution_binding.json',binding)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    for name,path in dependencies().items():
        destination=out/'dependency_snapshots'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(path.read_bytes())
    write_json(public/'status.json',{'status':'bound_ready'})
    print(out,flush=True)

def fit_head(x,y,gamma,alpha):
    mean=float(y.mean()/1000)
    kernel=np.exp(-gamma*np.maximum(2-2*x@x.T,0))
    kernel.flat[::len(x)+1]+=alpha
    c=solve(kernel,y/1000-mean,assume_a='pos',check_finite=False,overwrite_a=True)
    return c,mean

def predict(x,anchors,c,mean,gamma):
    values=[]
    for start in range(0,len(x),128):
        values.extend(((np.exp(-gamma*np.maximum(2-2*x[start:start+128]@anchors.T,0))@c+mean)*1000).tolist())
    return np.asarray(values)

def run(out):
    out,public=paths(out)
    reg=read_json(public/'registration.json')
    binding=read_json(public/'execution_binding.json')
    assert read_json(public/'status.json')['status']=='bound_ready'
    assert digest(public/'registration.json')==binding['registration_sha256']
    assert digest(__file__)==binding['source_sha256']
    for name,path in dependencies().items():
        assert digest(path)==binding['dependency_sha256'][name]
        assert digest(out/'dependency_snapshots'/name)==binding['dependency_sha256'][name]
    assert digest(eva_module.__file__)==binding['eva_implementation_sha256']
    assert digest(out/'source_adapter.safetensors')==binding['adapter_sha256']
    assert digest(local_path('split_path'))==reg['split_sha256']
    assert reg['gamma']==4 and reg['alpha']==1 and reg['price_scale']==1000
    def status(state,**extra):
        write_json(public/'status.json',{'status':state,'pid':os.getpid(),**extra})
    began=time.perf_counter()
    root=local_path('records_root')
    try:
        status('checking_control')
        frame=pd.read_csv(local_path('split_path'),dtype={'imageid':str,'group_id':str})
        train,valid=[frame.partition.eq(p).to_numpy() for p in ['train','validation']]
        assert len(frame)==8000 and train.sum()==6399 and valid.sum()==1601
        assert not set(frame.loc[train,'group_id'])&set(frame.loc[valid,'group_id'])
        y=frame.loc[train,'price'].to_numpy()
        baseline_root=root/reg['baseline_fusion']
        assert digest(baseline_root/'ensemble.json')==reg['baseline_ensemble_sha256']
        ensemble=read_json(baseline_root/'ensemble.json')
        own=[i for i,r in enumerate(ensemble['models']) if Path(r)==Path(reg['original_teacher_kernel'])]
        assert len(own)==1
        own=own[0]
        other=np.zeros(1601)
        for i,(reference,w) in enumerate(zip(ensemble['models'],ensemble['weights'])):
            if not w or i==own: continue
            file=root/reference/'predictions.csv'
            assert digest(file)==ensemble['prediction_sha256'][i]
            t=pd.read_csv(file,dtype={'imageid':str})
            assert t.imageid.tolist()==frame.loc[valid,'imageid'].tolist()
            np.testing.assert_array_equal(t.price.to_numpy(),frame.loc[valid,'price'].to_numpy())
            other+=w*t.prediction.to_numpy()
        previous=pd.read_csv(baseline_root/'predictions.csv',dtype={'imageid':str})
        assert previous.imageid.tolist()==frame.loc[valid,'imageid'].tolist()
        source_reg_file=REPOSITORY/'output/experiment_record'/reg['original_teacher_kernel']/'registration.json'
        assert digest(source_reg_file)==binding['source_feature_registration_sha256']
        old_features=features_for(read_json(source_reg_file),frame)
        with threadpool_limits(limits=2):
            c0,mean0=fit_head(old_features[train],y,4.,1.)
            base=predict(old_features[valid],old_features[train],c0,mean0,4.)
        control=other+ensemble['weights'][own]*np.maximum(base,0)
        np.testing.assert_allclose(control,previous.prediction.to_numpy(),rtol=1e-7,atol=1e-4)
        mse=float(np.mean((control-frame.loc[valid,'price'].to_numpy())**2))
        np.testing.assert_allclose(mse,reg['baseline_validation_mse'],rtol=1e-10,atol=1e-6)
        write_json(public/'original_head_control.json',{'passed':True,'validation_n':1601,'mse':mse,
            'max_price_error':float(abs(control-previous.prediction.to_numpy()).max()),
            'scope':'Independent same-gamma/alpha refit of the original cached teacher features and unchanged blend; no GPU zero-adapter claim'})
        del old_features,c0
        gc.collect()
        head_path=root/reg['original_teacher_kernel']/'head.npz'
        assert digest(head_path)==reg['original_teacher_head_sha256']
        weights=local_path('cache_root')/'pe_core_giant_weights'/REVISION
        assert digest(weights/'config.json')==binding['config_sha256']
        assert digest(weights/'model.safetensors')==binding['base_checkpoint_sha256']
        torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats()
        status('loading_gpu')
        with np.load(head_path,allow_pickle=False) as head:
            old_anchors,old_c=head['train_features'].copy(),head['coefficients'].copy()
            old_mean=float(head['target_mean'])
            model=QueryPrice(head)
        adapter=load_file(str(out/'source_adapter.safetensors'),device='cpu')
        params=dict(model.visual.named_parameters())
        assert set(adapter)=={name for name,p in params.items() if p.requires_grad}
        assert sum(v.numel() for v in adapter.values())==73728
        with torch.no_grad():
            for name,value in adapter.items():
                assert value.shape==params[name].shape and value.dtype==torch.float32 and torch.isfinite(value).all()
                params[name].copy_(value)
        model.visual.eval()
        manifest={'feature_family':reg['feature_family'],'feature_dim':2560,'dtype':'float64',
            'split_sha256':reg['split_sha256'],'imageids':frame.imageid.tolist(),'pixel_hashes':frame.pixel_hash.tolist(),
            'adapter_sha256':binding['adapter_sha256'],'execution_binding_sha256':digest(public/'execution_binding.json'),
            'supervised_encoder':True,'encoder_training_partition':'6399 train only','new_model_oof_generated':False,
            'test_images_used':False,'mllm_components':False,'views':'Same official and448 whole two-view QueryPrice normalization; unchanged final epoch2 adapter'}
        key=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
        cache=local_path('cache_root')/reg['feature_family']/key
        cache.mkdir(parents=True,exist_ok=False)
        write_json(cache/'manifest.json',manifest)
        write_json(public/'feature_manifest.json',{**manifest,'cache_key':key})
        features=np.empty((8000,2560),dtype=np.float64)
        chunks={}
        def image_for(name):
            with Image.open(local_path('data_root')/'train'/name) as image:
                return ImageOps.exif_transpose(image).convert('RGB')
        with torch.no_grad():
            for i,row in enumerate(frame.itertuples()):
                if pixel_hash(local_path('data_root')/'train'/row.imageid)!=row.pixel_hash:
                    raise ValueError('Provided image pixels changed: '+row.imageid)
                value=model.query(image_for(row.imageid)).cpu().numpy()
                assert value.shape==(1,2560) and value.dtype==np.float64 and np.isfinite(value).all()
                np.testing.assert_allclose(np.linalg.norm(value),1,rtol=1e-10,atol=1e-12)
                features[i]=value[0]
                if (i+1)%100==0:
                    file=cache/f'{i-99:05d}_100.npy'
                    with file.open('xb') as f: np.save(f,features[i-99:i+1],allow_pickle=False)
                    chunks[file.name]=digest(file)
                    write_json(cache/'chunk_sha256.json',chunks)
                    status('extracting',images=i+1,total=8000)
                    print(f'Fixed epoch2 LoRA features: {i+1}/8000',flush=True)
            probes=np.sort(np.random.default_rng(2026).choice(8000,12,replace=False))
            errors=[]
            for index in probes:
                actual=model.query(image_for(frame.imageid.iloc[index])).cpu().numpy().reshape(-1)
                np.testing.assert_allclose(actual,features[index],rtol=1e-4,atol=3e-6)
                errors.append(float(abs(actual-features[index]).max()))
        peak=torch.cuda.max_memory_allocated()/2**30
        del model,adapter
        torch.cuda.empty_cache()
        for name,sha in chunks.items(): assert digest(cache/name)==sha
        np.testing.assert_array_equal(np.concatenate([np.load(cache/name,allow_pickle=False) for name in sorted(chunks)]),features)
        source_csv=root/reg['upstream']/f"epoch_{reg['source_checkpoint_epoch']}"/'predictions.csv'
        assert digest(source_csv)==binding['upstream_epoch_predictions_sha256']
        source_prices=pd.read_csv(source_csv,dtype={'imageid':str})
        assert source_prices.imageid.tolist()==frame.loc[valid,'imageid'].tolist()
        with threadpool_limits(limits=2):
            original_query=predict(features[valid],old_anchors,old_c,old_mean,4.)
            old_blend=other+ensemble['weights'][own]*np.maximum(original_query,0)
            np.testing.assert_allclose(old_blend,source_prices.prediction.to_numpy(),rtol=1e-7,atol=1e-4)
            write_json(public/'fixed_epoch2_price_replay.json',{'passed':True,'validation_n':1601,
                'max_price_error':float(abs(old_blend-source_prices.prediction.to_numpy()).max()),
                'all8000_pixel_hashes_verified':True,'all_feature_chunks_verified':True,
                'gpu_feature_probe_n':12,'max_gpu_feature_replay_error':max(errors),
                'adapter_sha256':binding['adapter_sha256'],'cache_key':key})
            status('fitting_coherent_head')
            c,mean=fit_head(features[train],y,4.,1.)
            query=predict(features[valid],features[train],c,mean,4.)
        branch=out/'coherent_branch'
        branch.mkdir()
        branch_metrics=save_predictions(branch,frame.loc[valid],query,1300)
        np.savez_compressed(out/'head.npz',train_features=features[train],coefficients=c,gamma=4.,alpha=1.,target_mean=mean)
        final=other+ensemble['weights'][own]*np.maximum(query,0)
        metrics=save_predictions(out,frame.loc[valid],final,1300)
        with np.load(out/'head.npz',allow_pickle=False) as head:
            with threadpool_limits(limits=2): replay=predict(features[valid],head['train_features'],head['coefficients'],float(head['target_mean']),4.)
        np.testing.assert_allclose(replay,query,rtol=1e-10,atol=1e-7)
        result={**metrics,'branch_metrics':branch_metrics,'baseline_validation_mse':mse,
            'fixed_epoch2_asymmetric_validation_mse':float(np.mean((old_blend-frame.loc[valid,'price'].to_numpy())**2)),
            'relative_gain':1-metrics['mse']/mse,'screen_passed':metrics['mse']<mse*.99,
            'fixed_blend_weights':True,'new_model_oof_generated':False,'meta_weights_relearned':False,
            'three_seed_adoption_completed':False,'export_eligible':False,'full_refit_adapter_supported':False,
            'cache_key':key,'adapter_sha256':binding['adapter_sha256'],'head_sha256':digest(out/'head.npz'),
            'seconds':time.perf_counter()-began,'peak_allocated_gib':peak,'test_images_used':False}
        write_json(public/'results.json',result)
        status('completed',screen_passed=result['screen_passed'])
        print(result,flush=True)
    except BaseException as exc:
        status('failed',error=repr(exc))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['bind','run'],required=True)
    parser.add_argument('--protocol-dir',type=Path,required=True)
    args=parser.parse_args()
    (bind if args.phase=='bind' else run)(args.protocol_dir)
