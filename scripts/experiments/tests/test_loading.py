"""Real multiprocessing contracts for epoch delivery and unchanged training RNG."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import resolve, read_json
from preparation.images import HouseImages
from training.engine import loader, worker_seed, train_epoch, EarlyStop, fit_neural, predict
from models.alexnet import AlexNetRegressor
from training.loading import LoaderPool


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(.5)
        self.fc = torch.nn.Linear(3, 1)

    def forward(self, x):
        return self.fc(self.dropout(x.mean((-1, -2)))).squeeze(-1)


class LoadingContracts(unittest.TestCase):
    def test_persistent_epochs_match_legacy_and_cleanup(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp).resolve()
            self.assertEqual(path.parent, ROOT.resolve())
            (path/'train').mkdir()
            rng = np.random.default_rng(17)
            for i in range(9):
                Image.fromarray(rng.integers(0, 256, (33, 47, 3), dtype=np.uint8)).save(path/'train'/f'{i}.png')
            frame = pd.DataFrame({'imageid': [f'{i}.png' for i in range(9)], 'price': np.arange(9)+100.})
            cfg = resolve(ROOT/'configs/base.json')
            cfg['preprocess']['size'] = 64
            cfg['augmentation'] = {'noise': {'probability': 1., 'std': .01}}
            cfg['training'].update(batch_size=4, accumulation=2, loader_mode='persistent')
            for workers in (0, 2):
                cfg['training']['workers'] = workers
                old_ds = HouseImages(path, frame, cfg, training=True)
                new_ds = HouseImages(path, frame, cfg, training=True)
                old_model = TinyNet()
                new_model = deepcopy(old_model)
                old_opt = torch.optim.SGD(old_model.parameters(), lr=.01)
                new_opt = torch.optim.SGD(new_model.parameters(), lr=.01)
                with LoaderPool(cfg, loader, worker_seed) as pool:
                    batches = pool.get(new_ds, training=True)
                    processes = []
                    for epoch in (1, 2, 4):
                        old_ds.epoch = new_ds.epoch = epoch
                        before = torch.get_rng_state().clone()
                        old = list(loader(old_ds, cfg, True))
                        new = list(batches)
                        self.assertTrue(torch.equal(before, torch.get_rng_state()))
                        self.assertEqual([i for b in old for i in b[2]], [i for b in new for i in b[2]])
                        self.assertEqual(len(set(i for b in new for i in b[2])), 9)
                        for a, b in zip(old, new):
                            torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
                            torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
                            self.assertEqual(a[3], b[3])
                        torch.manual_seed(200+epoch)
                        train_epoch(old_model, old_ds, cfg, old_opt, torch.device('cpu'))
                        old_rng = torch.get_rng_state().clone()
                        torch.manual_seed(200+epoch)
                        train_epoch(new_model, new_ds, cfg, new_opt, torch.device('cpu'), batches)
                        self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
                        for a, b in zip(old_model.parameters(), new_model.parameters()):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                    if workers:
                        processes = list(batches._iterator._workers)
                        self.assertTrue(all(p.is_alive() for p in processes))
                    eval_ds = HouseImages(path, frame, cfg)
                    cached = pool.get(eval_ds, probe=True)
                    fresh = list(loader(eval_ds, cfg))
                    for a, b in zip(cached, fresh):
                        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
                        self.assertEqual(a[2], b[2])
                self.assertTrue(all(not p.is_alive() for p in processes))

    def test_stopping_boundaries_recovery_disabled_nonfinite(self):
        config = {'enabled': True, 'min_epochs': 4, 'patience': 2, 'relative_improvement': .001}
        stopper = EarlyStop(config)
        self.assertEqual([stopper.update(v, i) for i, v in enumerate([100,100,100,100],1)], [False,False,False,True])
        stopper = EarlyStop(config)
        self.assertEqual([stopper.update(v, i) for i, v in enumerate([100,100,99,99,99],1)], [False,False,False,False,True])
        stopper = EarlyStop({**config,'enabled':False})
        self.assertFalse(any(stopper.update(100,i) for i in range(1,10)))
        with self.assertRaises(FloatingPointError):
            stopper.update(float('nan'), 10)

    def test_early_stop_records_and_returns_selected_checkpoint(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp).resolve()
            self.assertEqual(path.parent, ROOT.resolve())
            (path/'train').mkdir()
            rng = np.random.default_rng(31)
            for i in range(12):
                Image.fromarray(rng.integers(0,256,(70,90,3),dtype=np.uint8)).save(path/'train'/f'{i}.png')
            frame = pd.DataFrame({'imageid':[f'{i}.png' for i in range(12)],'price':[100.,900.,1700.]*4})
            frame['group_id']=frame.imageid
            cfg = resolve(ROOT/'configs/base.json')
            cfg['device']='cpu'
            cfg['model']['head']='mean'
            cfg['preprocess']['size']=64
            cfg['training'].update(workers=0,loader_mode='persistent',epochs=5,cpu_threads=2)
            cfg['training']['early_stopping'].update(enabled=True,min_epochs=2,patience=1,relative_improvement=.99)
            out=path/'run';out.mkdir()
            train,val=frame.iloc[:9],frame.iloc[9:]
            returned=fit_neural(cfg,path,train,val,1300.,out)
            cost=read_json(out/'cost.json')
            self.assertEqual(cost['epochs'],2)
            self.assertTrue(cost['stopped_early'])
            self.assertEqual(read_json(out/'stopping.json')['reason'],'early_stopping')
            saved=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
            history=pd.read_csv(out/'history.csv')
            self.assertEqual(saved['epoch'],int(history.loc[history.val_mse.idxmin(),'epoch']))
            model=AlexNetRegressor(cfg['model']);model.load_state_dict(saved['model'])
            expected=predict(model,HouseImages(path,val,cfg,saved['stats']),cfg,torch.device('cpu'))
            np.testing.assert_array_equal(returned,expected)


if __name__ == '__main__':
    unittest.main()
