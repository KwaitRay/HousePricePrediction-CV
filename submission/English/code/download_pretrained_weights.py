"""Download pinned publisher files; local provenance is already included."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def valid(path,entry):
    if not path.is_file() or path.stat().st_size!=entry['bytes']:return False
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()==entry['sha256']

def main():
    from huggingface_hub import hf_hub_download
    for model in json.loads((ROOT/'pretrained_downloads.json').read_text(encoding='utf-8')):
        for entry in model['files']:
            path=ROOT/model['local_dir']/entry['filename']
            if valid(path,entry):continue
            if entry.get('origin')=='bundled_local_provenance':
                raise ValueError(f'Missing/corrupt bundled provenance: {path}')
            path=Path(hf_hub_download(repo_id=model['model'],revision=model['revision'],
                filename=entry['filename'],local_dir=str(ROOT/model['local_dir'])))
            if not valid(path,entry):raise ValueError(f'Publisher checksum mismatch: {path}')
        print('Verified',model['model'],model['revision'],flush=True)

if __name__=='__main__':main()
