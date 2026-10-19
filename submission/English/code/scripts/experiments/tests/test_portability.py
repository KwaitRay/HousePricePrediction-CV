"""Portable path contracts and extended optimizer CPU checks."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.local_paths import local_path, REPOSITORY
from common.config import stage_directory, resolve
from training.optimizers import optimizer_for


class Portability(unittest.TestCase):
    def test_relative_setting_independent_of_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with patch.dict(os.environ, {'CV_DATA_ROOT': '.local/path-test'}):
                    self.assertEqual(local_path('data_root'), REPOSITORY / '.local/path-test')
            finally:
                os.chdir(previous)

    def test_traditional_owns_stage_three(self):
        config = resolve(ROOT / 'configs/base.json')
        config['experiment']['stage'] = '01_baselines'
        config['method'] = 'traditional'
        self.assertEqual(stage_directory(config).name, '03_traditional')

    def test_extended_optimizers_update_and_restore_on_cpu(self):
        for name in ('adam', 'adagrad', 'muon'):
            model = torch.nn.Module()
            model.regressor = torch.nn.Sequential(
                torch.nn.Identity(), torch.nn.Linear(4, 4), torch.nn.ReLU(),
                torch.nn.Identity(), torch.nn.Linear(4, 2), torch.nn.ReLU())
            model.output = torch.nn.Linear(2, 1)
            config = {'optimizer': name, 'lr': .001, 'weight_decay': .0001}
            optimizer = optimizer_for(model, config)
            before = [p.detach().clone() for p in model.parameters()]
            sum(p.square().sum() for p in model.parameters()).backward()
            optimizer.step()
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
            restored = optimizer_for(model, config)
            restored.load_state_dict(optimizer.state_dict())
            restored.zero_grad(set_to_none=True)
            self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
