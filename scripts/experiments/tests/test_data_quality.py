"""Verify weighted gradients and the train-population weighting definition."""
from pathlib import Path
import sys, unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diagnostic_training import weighted_sum
from data_quality_stage import mild_weights, fixed_views

class QualityContracts(unittest.TestCase):
    def test_weighted_gradients_match_real_uneven_accumulation(self):
        p=torch.tensor([.5,.7,1.1,1.5,.8],requires_grad=True)
        target=torch.tensor([.2,.4,.9,1.8,.6]);w=torch.tensor([1.2,.8,.9,1.2,.8])
        (weighted_sum(p,target,w)/5).backward();full=p.grad.clone()
        p.grad=None
        for a,b in ((0,2),(2,5)):weighted_sum(p[a:b],target[a:b],w[a:b]).backward()
        self.assertTrue(torch.allclose(p.grad/5,full))
        p.grad=None
        (weighted_sum(p,target,torch.ones_like(w))/5).backward()
        self.assertTrue(torch.allclose(p.grad,2*(p.detach()-target)/5))
    def test_training_weights_mean_one_and_unknown_bands_rejected(self):
        prices=np.array([300]*7+[700]*31+[1300]*20+[1800]*6)
        spec=mild_weights(prices)
        self.assertEqual(spec['counts'],[7,31,20,6])
        self.assertAlmostEqual(np.average(spec['weights'],weights=spec['counts']),1.)
        with self.assertRaises(ValueError):mild_weights(np.array([500,800]))
    def test_fixed_local_views_keep_context_reference_separate(self):
        x=torch.arange(3*100*150).reshape(3,100,150)
        views=fixed_views(x)
        self.assertEqual(len(views),4)
        self.assertTrue(all(v.shape==(3,60,90) for v in views))
        self.assertTrue(torch.equal(views[0],x[:,:60,:90]))
        self.assertTrue(torch.equal(views[-1],x[:,40:,60:]))

if __name__=='__main__':unittest.main()
