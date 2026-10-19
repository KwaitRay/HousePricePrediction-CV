"""Registered mid40 extraction and separate fresh-process GPU/pixel/cache audit."""
import argparse,hashlib,json,os,time,uuid
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
from common.config import read_json,write_json,digest
from common.local_paths import REPOSITORY,local_path
from preparation.split import pixel_hash
from pe_giant_mid40_encoder import make_encoder,MODEL,REVISION,CHECKPOINT_SHA
import torch
import timm.models.eva as eva

FAMILY='pe_giant_mid40_features'
PREFLIGHT='19_image_embedding_reference/pe_giant_mid40_preflight/20261015T035323Z_f4c60c8d'
DEPS=['pe_giant_mid40_encoder.py','pe_core_giant_preflight.py','pe_giant_whole_view.py',
      'frozen_kernel_search.py','common/config.py','common/local_paths.py','preparation/split.py']

def paths(out):
    out=out.resolve()
    return out,REPOSITORY/'output/experiment_record'/out.relative_to(local_path('records_root'))

def register():
    pre=REPOSITORY/'output/experiment_record'/PREFLIGHT
    assert read_json(pre/'status.json')['status']=='completed' and read_json(pre/'results.json')['passed']
    preg=read_json(pre/'registration.json')
    for name,sha in preg['dependency_sha256'].items():assert digest(Path(__file__).parent/name)==sha
    config=local_path('cache_root')/'pe_core_giant_weights'/REVISION/'config.json'
    assert digest(config)=='f3d4cd28623bd6dc7dcea522411c48ebfea44aae844aa5cb7a8401a3601d5b6f'
    relative=Path('19_image_embedding_reference/pe_giant_mid40')/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
    out=local_path('records_root')/relative;out.mkdir(parents=True,exist_ok=False)
    public=REPOSITORY/'output/experiment_record'/relative
    reg={'method':'Frozen PE-G block40 token means, fixed crop+whole views','source_sha256':digest(__file__),
        'preflight':PREFLIGHT,'preflight_results_sha256':digest(pre/'results.json'),'preflight_registration_sha256':digest(pre/'registration.json'),
        'dependency_sha256':{name:digest(Path(__file__).parent/name) for name in DEPS},
        'eva_implementation_sha256':digest(eva.__file__),'split_sha256':digest(local_path('split_path')),
        'model':MODEL,'revision':REVISION,'checkpoint_sha256':CHECKPOINT_SHA,'config_sha256':digest(config),'cache_family':FAMILY,'feature_dim':3072,
        'batch_size':1,'dtype':'float64 after explicit FP32 feature storage rounding in encoder',
        'hypothesis':'Intermediate patch averages may retain complementary visible scene detail to the final learned attention pooling and semantic projection; layer39 zero-based fixed before any price evaluation',
        'encoder_policy':preg['precision'],'pooling':preg['pooling'],'views':preg['views'],
        'baseline_fusion':'15_fixed_ensembles/pe_whole_pca_incremental_fusion/20261015T033810Z_8aa6ec62',
        'kernel_grid':{'gammas':[1.,4.,16.],'alphas':[.01,.1,1.,10.],'group_folds':3},
        'gate':{'minimum_meta_cv_relative_gain':.01,'minimum_folds_won':2,'require_full_validation_improvement':True},
        'source_scope':'Generic image tower, provided images/labels only; no MLLM, external house-price data or new model training',
        'supervised_encoder':False,'mllm_components':False,'test_images_used':False,'automatic_test_export':False}
    write_json(public/'registration.json',reg);write_json(public/'status.json',{'status':'preregistered'})
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    for name in DEPS:
        target=out/'dependency_snapshots'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((Path(__file__).parent/name).read_bytes())
    print(str(out),flush=True)

def run(phase,out):
    out,public=paths(out);reg=read_json(public/'registration.json')
    assert read_json(public/'status.json')['status']==('preregistered' if phase=='extract' else 'features_extracted')
    assert digest(__file__)==reg['source_sha256'] and digest(eva.__file__)==reg['eva_implementation_sha256']
    assert digest(local_path('split_path'))==reg['split_sha256']
    assert digest(local_path('cache_root')/'pe_core_giant_weights'/REVISION/'config.json')==reg['config_sha256']
    for name,sha in reg['dependency_sha256'].items():assert digest(Path(__file__).parent/name)==sha
    def status(s,**kw):write_json(public/'status.json',{'status':s,'pid':os.getpid(),'phase':phase,**kw})
    began=time.perf_counter()
    try:
        frame=pd.read_csv(local_path('split_path'),dtype={'imageid':str,'group_id':str})
        train=frame.partition.eq('train').to_numpy()
        assert len(frame)==8000 and train.sum()==6399 and (~train).sum()==1601
        assert not set(frame.loc[train,'group_id'])&set(frame.loc[~train,'group_id'])
        if phase=='extract':
            manifest={**reg,'imageids':frame.imageid.tolist(),'pixel_hashes':frame.pixel_hash.tolist(),
                'preprocess':read_json(REPOSITORY/'output/experiment_record'/PREFLIGHT/'probe_results.json')['processor'],
                'encoder_adapter_source_sha256':reg['dependency_sha256']['pe_giant_mid40_encoder.py']}
            key=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
            cache=local_path('cache_root')/FAMILY/key;cache.mkdir(parents=True,exist_ok=False)
            write_json(cache/'manifest.json',manifest)
            reg.update(cache_key=key,feature_cache_key=key,manifest_sha256=digest(cache/'manifest.json'))
            write_json(public/'registration.json',reg)
        else:
            key=reg['cache_key'];cache=local_path('cache_root')/FAMILY/key
            assert digest(cache/'manifest.json')==reg['manifest_sha256']
        status('loading_gpu');torch.cuda.reset_peak_memory_stats()
        encode,_,processor=make_encoder();manifest=read_json(cache/'manifest.json')
        assert processor==manifest['preprocess']
        def image_for(index):
            path=local_path('data_root')/'train'/frame.imageid.iloc[index]
            assert pixel_hash(path)==frame.pixel_hash.iloc[index]
            with Image.open(path) as image:return ImageOps.exif_transpose(image).convert('RGB')
        if phase=='extract':
            features=np.empty((8000,3072),dtype=np.float64);chunks={}
            for i in range(8000):
                value=encode([image_for(i)])
                assert value.shape==(1,3072) and value.dtype==np.float64 and np.isfinite(value).all()
                np.testing.assert_allclose(np.linalg.norm(value),1,rtol=1e-10,atol=1e-12)
                features[i]=value[0]
                if (i+1)%100==0:
                    file=cache/f'{i-99:05d}_100.npy'
                    with file.open('xb') as f:np.save(f,features[i-99:i+1],allow_pickle=False)
                    chunks[file.name]=digest(file);write_json(cache/'chunk_sha256.json',chunks)
                    status('extracting',images=i+1,total=8000);print(f'Mid40: {i+1}/8000',flush=True)
            write_json(public/'extraction_results.json',{'all8000_pixel_hashes_verified':True,'cache_key':key,
                'seconds':time.perf_counter()-began,'feature_shape':[8000,3072],
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'price_fitting':False})
            status('features_extracted',images=8000,cache_key=key)
        else:
            checks=read_json(cache/'chunk_sha256.json');assert len(checks)==80
            parts=[]
            for start in range(0,8000,100):
                name=f'{start:05d}_100.npy';assert digest(cache/name)==checks[name]
                value=np.load(cache/name,allow_pickle=False)
                assert value.shape==(100,3072) and value.dtype==np.float64 and np.isfinite(value).all()
                np.testing.assert_allclose(np.linalg.norm(value,axis=1),1,rtol=1e-10,atol=1e-12);parts.append(value)
            features=np.concatenate(parts)
            for i in range(8000):
                assert pixel_hash(local_path('data_root')/'train'/frame.imageid.iloc[i])==frame.pixel_hash.iloc[i]
            probes=np.sort(np.random.default_rng(2026).choice(8000,12,replace=False))
            actual=encode([image_for(i) for i in probes])
            np.testing.assert_allclose(actual,features[probes],rtol=1e-4,atol=3e-6)
            audit={'passed':True,'feature_manifest_sha256':digest(cache/'manifest.json'),'cache_key':key,
                'all8000_pixel_hashes_verified':True,'all_feature_chunk_hashes_verified':True,'fresh_gpu_probes':12,
                'fresh_gpu_probe_max_abs_error':float(abs(actual-features[probes]).max()),
                'encoder_adapter_source_sha256':digest(Path(__file__).with_name('pe_giant_mid40_encoder.py')),
                'source_sha256':digest(__file__),'test_images_used':False,'price_fitting':False}
            write_json(public/'encoder_adapter_gpu_audit.json',audit);status('completed',cache_key=key);print(audit,flush=True)
    except BaseException as exc:status('failed',error=repr(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['register','extract','audit'],required=True);p.add_argument('--protocol-dir',type=Path)
    a=p.parse_args();register() if a.phase=='register' else run(a.phase,a.protocol_dir)
