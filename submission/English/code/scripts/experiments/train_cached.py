"""Verified opt-in cache adapter; invoke exactly like run.py train."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
from pathlib import Path
from common.config import resolve, write_json, digest
from training.cached_input import verify, CachedHouseImages


def main():
    if len(sys.argv) < 3 or sys.argv[1] != 'train':
        raise ValueError('Only development training is supported')
    meta = verify(resolve(Path(sys.argv[2])))
    import run
    import training.engine as engine
    engine.HouseImages = CachedHouseImages
    original = run.new_run
    def recorded(config, kind='development'):
        out = original(config, kind)
        write_json(out / 'runtime_cache.json', {
            'cache_key': meta['key'], 'inputs_sha256': meta['inputs_sha256'],
            'adapter_sha256': digest(Path(__file__).parent / 'training/cached_input.py'),
            'migration_evidence': os.environ.get('CV_MIGRATION_EVIDENCE'),
            'note': 'New repository runtime; separately verified against historical implementation.'})
        return out
    run.new_run = recorded
    run.main(sys.argv[1:])


if __name__ == '__main__':
    main()
