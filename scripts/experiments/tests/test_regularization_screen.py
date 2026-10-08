from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from regularization_screen import shortlist


class ScreeningContracts(unittest.TestCase):
    def test_top_two_must_beat_both_controls(self):
        rows=[{'index':i,'mse':v} for i,v in enumerate([10,9,8,7,float('nan'),11])]
        self.assertEqual([r['index'] for r in shortlist(rows,10,9)],[3,2])

    def test_none_or_one_finalist_without_forcing_replications(self):
        self.assertEqual(shortlist([{'index':1,'mse':10}],10,10),[])
        self.assertEqual(len(shortlist([{'index':1,'mse':9}],10,10)),1)


if __name__=='__main__':unittest.main()
