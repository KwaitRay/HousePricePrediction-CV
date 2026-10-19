"""Original-resolution stochastic training; verified cached deterministic evaluation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
from copy import deepcopy
from pathlib import Path
from common.config import resolve, write_json
from preparation.images import HouseImages
from training.cached_input import CachedHouseImages, verify


class AugmentedHouseImages(HouseImages):
    def __init__(self, root, frame, config, stats=None, training=False, partition='train'):
        super().__init__(root, frame, config, stats, training, partition)
        self.cached = None
        if not training:
            plain = deepcopy(config)
            plain['augmentation'] = {}
            self.cached = CachedHouseImages(root, frame, plain, stats, False, partition)

    def __getitem__(self, index):
        return self.cached[index] if self.cached is not None else super().__getitem__(index)


def main():
    if len(sys.argv) != 3 or sys.argv[1] != 'train':
        raise ValueError('Development train command required')
    plain = resolve(Path(sys.argv[2])); plain['augmentation'] = {}
    meta = verify(plain)
    import run
    import training.engine as engine
    engine.HouseImages = AugmentedHouseImages
    original = run.new_run
    def recorded(config, kind='development'):
        out = original(config, kind)
        write_json(out / 'runtime_cache.json', {'cache_key': meta['key'], 'inputs_sha256': meta['inputs_sha256'],
                   'training': 'Original image -> augmentation -> resize -> normalize',
                   'evaluation': 'Verified unaugmented stretch cache'})
        return out
    run.new_run = recorded
    run.main(sys.argv[1:])


if __name__ == '__main__':
    main()
