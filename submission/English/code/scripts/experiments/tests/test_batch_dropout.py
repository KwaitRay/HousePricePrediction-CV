from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import resolve, differences
from batch_dropout import configuration


class BatchDropoutContracts(unittest.TestCase):
    def test_only_registered_factor_changes(self):
        base = resolve(ROOT / 'configs/selected/schedule.json')
        for batch in (16, 32, 64):
            c = configuration(base, 'batch', batch)
            self.assertLessEqual(set(differences(base, c)), {'training.accumulation'})
            self.assertEqual(c['training']['batch_size'], 4)
            self.assertEqual(c['training']['accumulation'] * 4, batch)
            for dropout in (0, .2, .5):
                d = configuration(c, 'dropout', dropout)
                self.assertLessEqual(set(differences(c, d)), {'model.dropout'})
                self.assertEqual(d['training'], c['training'])


if __name__ == '__main__': unittest.main()
