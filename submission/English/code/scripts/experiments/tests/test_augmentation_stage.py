"""Ensure augmentation remains stochastic across persistent-worker epochs."""
from pathlib import Path
import sys
import tempfile
import unittest
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import resolve, validate, differences
from train_augmented import AugmentedHouseImages
from preparation.images import HouseImages
from training.loading import LoaderPool
from training.engine import loader, worker_seed


class AugmentationContracts(unittest.TestCase):
    def test_registered_candidates_only_change_augmentation(self):
        base = resolve(ROOT / 'configs/selected/training.json')
        for name in ('augment_01_flip','augment_05_rotation','augment_06_translation','augment_07_crop','augment_08_blur','augment_09_noise','augment_10_erase'):
            c = validate(resolve(ROOT / f'configs/03_augmentation/{name}.json'))
            self.assertEqual(len(c['augmentation']),1)
            self.assertTrue(all(k.startswith('augmentation.') for k in differences(base,c)))

    def test_persistent_workers_deliver_geometry_epochs(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp); (root/'train').mkdir()
            rng = np.random.default_rng(7)
            for i in range(6):
                Image.fromarray(rng.integers(0,256,(48,71,3),dtype=np.uint8)).save(root/'train'/f'{i}.png')
            frame = pd.DataFrame({'imageid':[f'{i}.png' for i in range(6)],'price':[100.]*6})
            c = resolve(ROOT/'configs/03_augmentation/augment_05_rotation.json')
            c['augmentation']['rotation']['probability'] = 1.0
            c['training'].update(workers=2,batch_size=4,loader_mode='persistent')
            ds = AugmentedHouseImages(root,frame,c,training=True)
            raw = HouseImages(root,frame,c,training=True)
            with LoaderPool(c,loader,worker_seed) as pool:
                batches = pool.get(ds,training=True)
                seen = []
                for epoch in (1,2):
                    ds.epoch = raw.epoch = epoch
                    observed = {name:(x,events) for batch in batches for name,x,events in zip(batch[2],batch[0],batch[3])}
                    self.assertEqual(len(observed),6)
                    for i in range(6):
                        x,_,name,events = raw[i]
                        self.assertTrue(torch.equal(x,observed[name][0]))
                        self.assertEqual(events,observed[name][1])
                    seen.append(observed['0.png'][0])
                self.assertFalse(torch.equal(*seen))


if __name__ == '__main__': unittest.main()
