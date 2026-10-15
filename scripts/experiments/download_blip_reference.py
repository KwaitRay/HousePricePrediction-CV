"""Fetch pinned official BLIP-base files and verify publisher LFS checksums."""
import os,json,hashlib,argparse,re
from pathlib import Path
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import HfApi,hf_hub_download
from common.local_paths import local_path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',default='Salesforce/blip-image-captioning-base');parser.add_argument('--cache-folder',default='blip_weights');args=parser.parse_args()
    repo=args.repo;info=HfApi().model_info(repo,files_metadata=True)
    siblings={s.rfilename:s for s in info.siblings}
    sharded='model.safetensors' not in siblings and 'model.safetensors.index.json' in siblings
    weight='model.safetensors.index.json' if sharded else 'model.safetensors' if 'model.safetensors' in siblings else 'pytorch_model.bin'
    destination=local_path('cache_root')/args.cache_folder/info.sha;destination.mkdir(parents=True,exist_ok=True)
    record={'repo':repo,'revision':info.sha,'weight_file':weight,'files':{},'scope':'Download only; downstream uses vision tensors, no text modules',
        'model_card':f'https://huggingface.co/{repo}','license':info.card_data.to_dict().get('license') if info.card_data is not None else None}
    (destination/'download.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    def fetch(filename):
        path=Path(hf_hub_download(repo,filename,revision=info.sha,local_dir=destination))
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
        value=h.hexdigest();lfs=siblings[filename].lfs
        if lfs is not None:
            expected=getattr(lfs,'sha256',None) or (lfs.get('sha256') if isinstance(lfs,dict) else None)
            if expected is None or expected!=value:raise ValueError('Publisher checksum mismatch')
        record['files'][filename]={'sha256':value,'bytes':path.stat().st_size,'publisher_lfs_verified':lfs is not None}
        (destination/'download.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(filename,value,flush=True)
        return path
    for filename in ['config.json']+(['preprocessor_config.json'] if 'preprocessor_config.json' in siblings else [])+[weight]+(['README.md'] if 'README.md' in siblings else []):
        fetch(filename)
    if sharded:
        index=json.loads((destination/weight).read_text(encoding='utf-8'))
        vision_map={k:v for k,v in index['weight_map'].items() if k.startswith('vision_model.')}
        shards=sorted(set(vision_map.values()))
        if not shards or any(not re.fullmatch(r'model-\d{5}-of-\d{5}\.safetensors',name) or name not in siblings for name in shards):
            raise ValueError('Invalid official vision shard index')
        record.update(sharded=True,vision_weight_shards=shards,vision_tensor_count=len(vision_map),
            shard_scope='Only shards containing vision_model tensors; a required shard may also contain unused text tensors')
        (destination/'download.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        for filename in shards:
            fetch(filename)
    print(destination,flush=True)

if __name__=='__main__':main()
