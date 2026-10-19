import tempfile
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import write_json, resolve, validate
from continue_training import decide


class ContinuationContracts(unittest.TestCase):
    def test_selection_requires_mean_and_two_paired_wins(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            controls = [root / f'b{i}' for i in range(3)]
            candidates = [root / f'a{i}' for i in range(3)]
            for p in controls: write_json(p / 'metrics.json', {'mse': 10})
            for values, expected in [([9, 9, 11], True), ([1, 11, 11], False), ([9, 9, 20], False), ([10, 10, 10], False)]:
                for p, mse in zip(candidates, values): write_json(p / 'metrics.json', {'mse': mse})
                self.assertEqual(decide(candidates, controls, 'test')['accepted'], expected)

    def test_adam_cosine_keeps_stopping_protocol(self):
        c = resolve(ROOT / 'configs/04_training_recorded/tune_02_adam_0p001.json')
        c = {k: v for k, v in c.items() if not k.startswith('_')}
        c['_parent_source'] = None
        c['experiment']['enforce_changes'] = False
        c['training']['schedule'] = 'cosine'
        validate(c)
        self.assertEqual(c['training']['epochs'], 60)
        self.assertEqual(c['training']['early_stopping']['min_epochs'], 40)
        self.assertEqual(c['training']['early_stopping']['patience'], 15)


if __name__ == '__main__': unittest.main()
