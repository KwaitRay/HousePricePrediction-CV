"""Download immutable official OpenAI CLIP weights with SHA256 verification."""
import argparse
import hashlib
import time
import urllib.request
from pathlib import Path
import open_clip
from common.config import digest, write_json
from common.local_paths import local_path, REPOSITORY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['ViT-L-14-336'], required=True)
    args = parser.parse_args()
    url = open_clip.get_pretrained_cfg(args.model, 'openai')['url']
    expected = url.split('/')[-2]
    destination = local_path('cache_root') / 'clip_weights' / (args.model + '.pt')
    destination.parent.mkdir(parents=True, exist_ok=True)
    public = REPOSITORY / 'output/experiment_record/13_clip_optimization' / (args.model.replace('-', '_') + '_download.json')
    registration = {'model': args.model, 'url': url, 'expected_sha256': expected,
                    'scope': 'Frozen visual encoder reference; official 336px checkpoint plus input preprocessing, not a pure resolution-only ablation',
                    'test_images_used': False}
    write_json(public, {**registration, 'status': 'downloading'})
    try:
        if destination.exists():
            if digest(destination) != expected:
                raise ValueError('Existing official checkpoint hash differs; refusing to overwrite it')
        else:
            partial = destination.with_suffix('.pt.partial')
            # A partial from an earlier failed transfer is retained and safely resumed.
            offset = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                    raise ValueError('Server did not honor exact checkpoint resume range')
                last = time.perf_counter()
                with partial.open('ab' if offset else 'wb') as stream:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        stream.write(block)
                        offset += len(block)
                        if time.perf_counter() - last >= 10:
                            print(f'CLIP336 download {offset // (1024**2)} MiB', flush=True)
                            last = time.perf_counter()
            if digest(partial) != expected:
                raise ValueError('Downloaded checkpoint SHA256 mismatch')
            partial.replace(destination)
        write_json(public, {**registration, 'status': 'completed', 'bytes': destination.stat().st_size,
                            'verified_sha256': expected})
        print('Verified official checkpoint:', destination, flush=True)
    except BaseException as exc:
        write_json(public, {**registration, 'status': 'failed', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
