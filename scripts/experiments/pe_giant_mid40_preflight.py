"""Two-process numerical/resource preflight, no price fitting or test images."""
import argparse,os,time,uuid
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
from common.config import read_json,write_json,digest
from common.local_paths import REPOSITORY,local_path
from pe_giant_mid40_encoder import make_encoder,MODEL,REVISION,CHECKPOINT_SHA
import torch
import timm.models.eva as eva

def main(phase,out):
    if phase=='probe':
        relative=Path('19_image_embedding_reference/pe_giant_mid40_preflight')/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
        out=local_path('records_root')/relative;out.mkdir(parents=True,exist_ok=False)
        public=REPOSITORY/'output/experiment_record'/relative
        frame=pd.read_csv(local_path('split_path'),dtype={'imageid':str})
        train=np.flatnonzero(frame.partition.eq('train').to_numpy())
        extra=np.random.default_rng(2026).choice(train[3:],12,replace=False)
        indices=np.concatenate([train[:3],extra])
        reg={'source_sha256':digest(__file__),'model':MODEL,'revision':REVISION,'checkpoint_sha256':CHECKPOINT_SHA,
            'split_sha256':digest(local_path('split_path')),'probe_imageids':frame.imageid.iloc[indices].tolist(),
            'probe_selection':'First3 training +12 fixed seed2026 randomly selected other training rows, labels never consulted',
            'feature_dim':3072,'block_zero_based':39,'pooling':'Raw patch tokens float32 mean, no final layer norm/attention pool/learned projection, unit per-view then concatenated unit',
            'views':'Unchanged official crop and gray448 whole-frame, batch1',
            'precision':'FrozenFP16 base; deterministic MATH SDPA, TF32off, fixed2 neutral warmups, FP32 per-view pooling/unit and FP32 storage-rounding before final FP64 unit',
            'feature_tolerance':{'rtol':1e-4,'atol':3e-6},'price_fitting':False,'test_images_used':False,
            'eva_implementation_sha256':digest(eva.__file__),
            'dependency_sha256':{name:digest(Path(__file__).with_name(name)) for name in ['pe_giant_mid40_encoder.py','pe_core_giant_preflight.py','pe_giant_whole_view.py']}}
        write_json(public/'registration.json',reg)
        (out/'source.py').write_bytes(Path(__file__).read_bytes())
        for name in reg['dependency_sha256']:(out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
        print(str(out),flush=True)
    else:
        out=out.resolve();public=REPOSITORY/'output/experiment_record'/out.relative_to(local_path('records_root'))
        assert read_json(public/'status.json')['status']=='probe_completed'
        reg=read_json(public/'registration.json')
    def status(state,**kw):write_json(public/'status.json',{'status':state,'pid':os.getpid(),**kw})
    try:
        assert digest(__file__)==reg['source_sha256'] and digest(eva.__file__)==reg['eva_implementation_sha256']
        assert digest(local_path('split_path'))==reg['split_sha256']
        for name,sha in reg['dependency_sha256'].items():assert digest(Path(__file__).with_name(name))==sha
        def image(name):
            with Image.open(local_path('data_root')/'train'/name) as im:return ImageOps.exif_transpose(im).convert('RGB')
        status('loading',phase=phase);torch.cuda.reset_peak_memory_stats()
        encode,hook_control,processor=make_encoder();images=[image(n) for n in reg['probe_imageids']]
        before=time.perf_counter();first=encode(images);second=encode(images)
        np.testing.assert_allclose(first,second,**reg['feature_tolerance'])
        assert first.shape==(15,3072) and np.isfinite(first).all()
        hook_errors=hook_control(images[:3])
        repeat_error=float(abs(first-second).max())
        if phase=='probe':
            np.save(out/'probe_features.npy',first,allow_pickle=False)
            write_json(public/'probe_results.json',{'passed':True,'repeat_max_error':repeat_error,'full_forward_hook_errors':hook_errors,
                'probe_features_sha256':digest(out/'probe_features.npy'),'processor':processor,'seconds':time.perf_counter()-before,
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'price_fitting':False})
            status('probe_completed')
        else:
            old=read_json(public/'probe_results.json')
            assert digest(out/'probe_features.npy')==old['probe_features_sha256'] and processor==old['processor']
            expected=np.load(out/'probe_features.npy',allow_pickle=False)
            np.testing.assert_allclose(first,expected,**reg['feature_tolerance'])
            write_json(public/'results.json',{'passed':True,'fresh_process_replay_n':15,
                'fresh_process_max_error':float(abs(first-expected).max()),'repeat_max_error':repeat_error,
                'full_forward_hook_errors':hook_errors,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                'price_fitting':False,'full_extraction_completed':False,'test_images_used':False})
            status('completed')
        print(read_json(public/('probe_results.json' if phase=='probe' else 'results.json')),flush=True)
    except BaseException as exc:status('failed',phase=phase,error=repr(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['probe','verify'],required=True);p.add_argument('--protocol-dir',type=Path)
    a=p.parse_args();main(a.phase,a.protocol_dir)
