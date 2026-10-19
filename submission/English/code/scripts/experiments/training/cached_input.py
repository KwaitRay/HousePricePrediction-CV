"""Explicit, verified reuse of a local lossless stretch-input cache."""
import os
from pathlib import Path
import importlib.metadata
import numpy as np
import torch
from common.config import read_json, digest, path_at_root
from preparation.images import HouseImages


def directory():
    return Path(os.environ['CV_STRETCH_CACHE_DIRECTORY'])


def verify(config):
    meta = read_json(directory() / 'manifest.json')
    spec = meta['specification']
    if meta['status'] != 'complete' or config['preprocess'] != spec['preprocess'] or config['augmentation']:
        raise ValueError('Cache configuration mismatch')
    if path_at_root(config['data']['root']) != Path(spec['root']):
        raise ValueError('Cache data root mismatch')
    if digest(directory() / 'inputs.npy') != meta['inputs_sha256']:
        raise ValueError('Cache checksum mismatch')
    for package, version in spec['packages'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f'Cache package mismatch: {package}')
    for row in spec['images']:
        if digest(Path(spec['root']) / 'train' / row['imageid']) != row['sha256']:
            raise ValueError('Source image checksum mismatch')
    array = np.load(directory() / 'inputs.npy', mmap_mode='r', allow_pickle=False)
    if array.shape != (len(spec['images']), 3, 224, 224) or array.dtype != np.float32:
        raise ValueError('Cache shape or dtype mismatch')
    return meta


class CachedHouseImages(HouseImages):
    def __init__(self, root, frame, config, stats=None, training=False, partition='train'):
        super().__init__(root, frame, config, stats, training, partition)
        self.cache_directory = directory()
        meta = read_json(self.cache_directory / 'manifest.json')
        spec = meta['specification']
        if (partition != 'train' or config['augmentation'] or config['preprocess'] != spec['preprocess']
                or self.stats != spec['stats'] or self.root.resolve() != Path(spec['root'])):
            raise ValueError('Unsupported cached dataset')
        self.cache_index = {row['imageid']: i for i, row in enumerate(spec['images'])}
        if not set(frame.imageid).issubset(self.cache_index):
            raise ValueError('Missing cached images')
        self._cache_array = None

    def __getstate__(self):
        result = self.__dict__.copy()
        result['_cache_array'] = None
        return result

    def __getitem__(self, index):
        if self._cache_array is None:
            self._cache_array = np.load(self.cache_directory / 'inputs.npy', mmap_mode='r', allow_pickle=False)
        row = self.frame.iloc[index]
        x = torch.from_numpy(np.array(self._cache_array[self.cache_index[row.imageid]], copy=True))
        return x, float(row.get('price', 0)), row.imageid, '{}'
