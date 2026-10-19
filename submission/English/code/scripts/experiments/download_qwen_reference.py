"""Pin and verify an official Qwen checkpoint for local-only exploration."""
import hashlib
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from huggingface_hub import HfApi, hf_hub_download
from common.config import write_json
from common.local_paths import local_path


def download(repo='Qwen/Qwen3.5-2B'):
    if not repo.startswith('Qwen/'):
        raise ValueError('Only official Qwen repositories are registered')
    info = HfApi().model_info(repo, files_metadata=True)
    destination = local_path('cache_root') / 'qwen_weights' / repo.split('/')[1] / info.sha
    destination.mkdir(parents=True, exist_ok=True)
    filenames = [s for s in info.siblings if '/' not in s.rfilename and
                 (s.rfilename.endswith(('.json', '.safetensors', '.jinja', '.txt')) or s.rfilename == 'README.md')]
    if not any(s.rfilename.endswith('.safetensors') for s in filenames):
        raise ValueError('Expected official safetensors weights')
    record = {'repo': repo, 'revision': info.sha, 'model_card': f'https://huggingface.co/{repo}',
              'scope': 'User-authorized independent local exploration; NOT course eligible; NO Kaggle export',
              'license': info.card_data.to_dict().get('license') if info.card_data else None, 'files': {}}
    write_json(destination / 'download.json', record)
    for sibling in sorted(filenames, key=lambda s: (s.rfilename.endswith('.safetensors'), s.rfilename)):
        name = sibling.rfilename
        path = Path(hf_hub_download(repo, name, revision=info.sha, local_dir=destination))
        hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(block)
        value = hasher.hexdigest()
        lfs = sibling.lfs
        if lfs is not None:
            expected = getattr(lfs, 'sha256', None) or (lfs.get('sha256') if isinstance(lfs, dict) else None)
            if value != expected:
                raise ValueError(f'Publisher SHA256 mismatch: {name}')
        elif name.endswith('.safetensors'):
            raise ValueError(f'Weight lacks publisher LFS checksum: {name}')
        record['files'][name] = {'sha256': value, 'bytes': path.stat().st_size,
                                 'publisher_lfs_verified': lfs is not None}
        write_json(destination / 'download.json', record)
        print(f'Qwen verified {name} ({path.stat().st_size} bytes)', flush=True)
    return destination


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='Qwen/Qwen3.5-2B')
    print(download(parser.parse_args().repo), flush=True)
