from pathlib import Path
import sys
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.alexnet import initialize
from common.config import resolve,validate,differences


class InitializationContracts(unittest.TestCase):
    def test_only_convolution_weights_change(self):
        def model():
            m=torch.nn.Module()
            m.features=torch.nn.Sequential(torch.nn.Conv2d(3,4,3),torch.nn.ReLU())
            m.regressor=torch.nn.Sequential(torch.nn.Linear(4,3),torch.nn.ReLU())
            m.output=torch.nn.Linear(3,1)
            return m
        a,b=model(),model();initialize(a,2026,.7,'default');initialize(b,2026,.7,'xavier_convolution')
        changed=[name for name,p in a.state_dict().items() if not torch.equal(p,b.state_dict()[name])]
        self.assertEqual(changed,['features.0.weight'])

    def test_only_registered_config_factor_changes(self):
        a=resolve(ROOT/'configs/selected/training.json')
        b=validate(resolve(ROOT/'configs/02_training/tune_08_initialization.json'))
        self.assertEqual(set(differences(a,b)),{'training.initialization'})
        self.assertEqual(b['training']['early_stopping'],a['training']['early_stopping'])


if __name__=='__main__':unittest.main()
