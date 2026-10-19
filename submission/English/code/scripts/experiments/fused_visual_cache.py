"""Equal-normalized block fusion of already extracted CLIP and BLIP image features."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from common.local_paths import local_path
from common.config import digest,write_json

def main():
    base=local_path('cache_root');split=local_path('split_path');frame=pd.read_csv(split,dtype={'imageid':str})
    b16=base/'clip_features/780d60dc5a9918db07be9fc116a60967e4c60c7759338a241a2eb76da3390760'
    blip=base/'blip_features/bab1d08edf868e4350bb353f4dac15f6b52b71395d78d2542659f691584a6bdd'
    multi=base/'clip_multiview/c518c3a14cac4fe8296d031904963a637c9b1caea151ebf5b108f5afcf92e34d'
    original_manifest=local_path('records_root')/'03_clip_reference/runs/20261013T043141Z_c9f899f2/data_manifest.json'
    om=json.loads(original_manifest.read_text());original=base/'clip_features'/hashlib.sha256(json.dumps(om,sort_keys=True).encode()).hexdigest()
    blocks=[];provenance=[]
    for folder in [b16,blip,multi,original]:
        mp=folder/'manifest.json' if folder!=original else original_manifest
        manifest=json.loads(mp.read_text());assert manifest['imageids']==frame.imageid.tolist() and manifest['split_sha256']==digest(split)
        files=sorted(folder.glob('*.npy'));values=np.concatenate([np.load(p,allow_pickle=False) for p in files]).astype(np.float64)
        assert len(values)==8000 and np.isfinite(values).all()
        blocks.append(values);provenance.append({'cache':str(folder.relative_to(base)),'manifest_sha256':digest(mp),'chunks_sha256':[digest(p) for p in files]})
    assert blocks[0].shape==(8000,512) and blocks[1].shape==(8000,768) and blocks[2].shape==(8000,2,512) and blocks[3].shape==(8000,512)
    joined_multi=np.concatenate([blocks[3],blocks[2].reshape(8000,1024)],axis=1)
    units=[blocks[0],blocks[1],joined_multi]
    units=[x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12) for x in units]
    result=np.concatenate(units,axis=1);result/=np.maximum(np.linalg.norm(result,axis=1,keepdims=True),1e-12)
    manifest={'imageids':frame.imageid.tolist(),'split_sha256':digest(split),'sources':provenance,'feature_dim':2816,
        'fusion':'Each of B16 image, BLIP image, B32 three-view block normalized to unit norm, concatenated and globally L2-normalized; equal block similarity weight',
        'source_sha256':digest(__file__),'test_images_used':False,'course_scope':'Exploratory pretrained visual features, eligibility unresolved'}
    key=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest();folder=base/'fused_features'/key;folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'manifest.json',manifest)
    np.save(folder/'00000_8000.npy',result.astype(np.float32))
    print(key,flush=True)

if __name__=='__main__':main()
