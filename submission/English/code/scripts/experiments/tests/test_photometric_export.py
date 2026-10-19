"""Small deterministic checks for the aggregate photometric evidence exporter."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_photometric_screen import metrics, price_slices, portable_config


class PhotometricEvidence(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(dict(imageid=list('abcd'), price=[300., 700., 1300., 1500.],
                                       prediction=[301., 702., 1303., 1504.]))

    def test_original_units_and_strict_tail(self):
        result = metrics(self.frame)
        self.assertEqual(result['mse'], 7.5)
        self.assertEqual(result['mae'], 2.5)
        self.assertEqual(result['tail_n'], 1)
        self.assertEqual(result['tail_mse'], 16.)

    def test_bin_boundaries_and_contributions(self):
        slices = price_slices(self.frame, 'control')
        self.assertEqual([r['n'] for r in slices], [1, 1, 1, 1])
        self.assertEqual([r['mse'] for r in slices], [1., 4., 9., 16.])
        self.assertEqual(sum(r['contribution_to_overall_mse'] for r in slices), 7.5)

    def test_nonfinite_duplicate_and_empty_rejected(self):
        for bad in (self.frame.iloc[:0], self.frame.assign(prediction=np.nan),
                    self.frame.assign(imageid='duplicate')):
            with self.assertRaises(ValueError):
                metrics(bad)

    def test_path_redaction_does_not_mutate_original(self):
        source = dict(_source='D:/private/config.json', output_root='D:/private/output',
                      data=dict(root='D:/private/data', split='D:/private/split.csv'),
                      augmentation={'brightness': {'probability': 1., 'range': [.9, 1.1]}})
        public = portable_config(source)
        self.assertNotIn('_source', public)
        self.assertNotIn('D:/private', str(public))
        self.assertEqual(source['data']['root'], 'D:/private/data')
        self.assertEqual(public['augmentation'], source['augmentation'])


if __name__ == '__main__':
    unittest.main()
